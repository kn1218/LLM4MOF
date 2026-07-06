"""Phase 4a: Deterministic feature extraction from parsed XYZ data.

Computes Layer 1 facts: formula, molecular weight, atom counts, metals,
degree of unsaturation, connection point details, bond graph summary,
rigidity, planarity, and edge/node-specific properties.

No LLM involvement. Every value is traceable to the XYZ + bond block.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import networkx as nx

from .config import ATOMIC_WEIGHTS, METAL_SET, HALOGEN_SET, DUMMY_ELEMENT
from .xyz_parser import ParsedXYZ, Atom, Bond
from .smiles_generator import generate_smiles
from .schema import (
    Layer1Facts,
    ConnectionPoint,
    ConnectionPointsSummary,
    BondGraphSummary,
    MetalCoordination,
)


def extract_layer1(parsed: ParsedXYZ) -> Layer1Facts:
    """Extract all deterministic Layer 1 facts from a parsed XYZ file.

    Args:
        parsed: Output from xyz_parser.parse_xyz().

    Returns:
        Layer1Facts dataclass with all computed fields.
    """
    # ── Atom counts (exclude dummy) ──
    element_counts = Counter(a.element for a in parsed.real_atoms)
    total_atoms = len(parsed.real_atoms)

    # ── Formula (Hill notation: C first, H second, rest alphabetical) ──
    formula = _hill_formula(element_counts)

    # ── Molecular weight ──
    molecular_weight = sum(
        ATOMIC_WEIGHTS.get(a.element, 0.0) for a in parsed.real_atoms
    )
    molecular_weight = round(molecular_weight, 3)

    # ── Metals ──
    metals = sorted(set(a.element for a in parsed.real_atoms if a.element in METAL_SET))
    has_metal = len(metals) > 0

    # ── Connection points ──
    connection_points = _extract_connection_points(parsed)

    # ── Bond graph (exclude X atoms and their bonds) ──
    bond_graph = _build_bond_graph_summary(parsed)

    # ── Rigidity ──
    is_rigid = _compute_rigidity(parsed)

    # ── Degree of unsaturation ──
    dou = _degree_of_unsaturation(element_counts)

    # ── Planarity ──
    is_planar = _check_planarity(parsed.real_atoms)

    # ── Edge-specific ──
    length_angstroms = None
    topological_distance = None
    if parsed.bb_type == "edge" and len(parsed.dummy_atoms) >= 2:
        length_angstroms = _max_dummy_distance(parsed.dummy_atoms)
        topological_distance = _topological_distance_matrix(parsed)

    # ── Node-specific ──
    nuclearity = None
    geometry_inferred = None
    metal_coordination = None
    if parsed.bb_type == "node":
        nuclearity = sum(1 for a in parsed.real_atoms if a.element in METAL_SET)
        geometry_inferred = _infer_geometry(parsed.dummy_atoms)
        if has_metal:
            metal_coordination = _extract_metal_coordination(parsed)

    # ── SMILES / SELFIES ──
    smiles_result = generate_smiles(parsed)

    return Layer1Facts(
        formula=formula,
        molecular_weight=molecular_weight,
        atom_counts=dict(sorted(element_counts.items())),
        total_atoms=total_atoms,
        connection_points=connection_points,
        bond_graph=bond_graph,
        metals=metals,
        has_metal=has_metal,
        is_rigid=is_rigid,
        degree_of_unsaturation=dou,
        num_rings=bond_graph.num_rings,
        is_planar=is_planar,
        smiles=smiles_result.smiles,
        smiles_canonical=smiles_result.smiles_canonical,
        selfies=smiles_result.selfies,
        smiles_method=smiles_result.method,
        net_charge=smiles_result.net_charge,
        length_angstroms=length_angstroms,
        topological_distance=topological_distance,
        nuclearity=nuclearity,
        geometry_inferred=geometry_inferred,
        metal_coordination=metal_coordination,
    )


# ── Private helpers ────────────────────────────────────────────────────

def _hill_formula(counts: Counter) -> str:
    """Build molecular formula in Hill notation (C first, H second, rest alpha)."""
    parts = []
    # Carbon first
    if "C" in counts:
        parts.append(f"C{counts['C']}" if counts["C"] > 1 else "C")
    # Hydrogen second
    if "H" in counts:
        parts.append(f"H{counts['H']}" if counts["H"] > 1 else "H")
    # Rest alphabetical
    for elem in sorted(counts.keys()):
        if elem in ("C", "H"):
            continue
        parts.append(f"{elem}{counts[elem]}" if counts[elem] > 1 else elem)
    return "".join(parts)


def _extract_connection_points(parsed: ParsedXYZ) -> ConnectionPointsSummary:
    """Extract connection point details from dummy atoms and their bonds."""
    points = []
    for dummy in parsed.dummy_atoms:
        # Find what this dummy atom is bonded to
        bonded_to_idx = None
        bonded_to_elem = None
        bond_type = None
        for b in parsed.bonds:
            if b.atom1 == dummy.index:
                partner_idx = b.atom2
            elif b.atom2 == dummy.index:
                partner_idx = b.atom1
            else:
                continue
            # Partner should be a real atom
            partner = parsed.atoms[partner_idx]
            if not partner.is_dummy:
                bonded_to_idx = partner_idx
                bonded_to_elem = partner.element
                bond_type = b.bond_type
                break

        points.append(ConnectionPoint(
            atom_index=dummy.index,
            coordinates=[dummy.x, dummy.y, dummy.z],
            bonded_to_index=bonded_to_idx if bonded_to_idx is not None else -1,
            bonded_to_element=bonded_to_elem if bonded_to_elem is not None else "unknown",
            bond_type=bond_type if bond_type is not None else "S",
        ))

    # Connection chemistry: element at each anchor
    connection_chemistry = [p.bonded_to_element for p in points]

    # Distance matrix between dummy atoms
    distance_matrix = _compute_distance_matrix(parsed.dummy_atoms)

    return ConnectionPointsSummary(
        count=len(points),
        points=points,
        connection_chemistry=connection_chemistry,
        distance_matrix=distance_matrix,
    )


def _compute_distance_matrix(atoms: tuple[Atom, ...]) -> list[list[float]]:
    """Compute pairwise Euclidean distance matrix."""
    n = len(atoms)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt(
                (atoms[i].x - atoms[j].x) ** 2 +
                (atoms[i].y - atoms[j].y) ** 2 +
                (atoms[i].z - atoms[j].z) ** 2
            )
            d = round(d, 4)
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


def _build_real_graph(parsed: ParsedXYZ) -> nx.Graph:
    """Build a NetworkX graph of real atoms (excludes dummy atoms and their bonds)."""
    dummy_indices = {a.index for a in parsed.dummy_atoms}
    G = nx.Graph()
    # Add all real atoms as nodes
    for a in parsed.real_atoms:
        G.add_node(a.index, element=a.element)
    # Add real bonds
    for b in parsed.bonds:
        if b.atom1 not in dummy_indices and b.atom2 not in dummy_indices:
            G.add_edge(b.atom1, b.atom2, bond_type=b.bond_type)
    return G


def _build_bond_graph_summary(parsed: ParsedXYZ) -> BondGraphSummary:
    """Build bond graph summary excluding dummy atoms and their bonds."""
    G = _build_real_graph(parsed)

    # Bond type counts
    type_counts = {"S": 0, "D": 0, "T": 0, "A": 0}
    for _, _, data in G.edges(data=True):
        bt = data.get("bond_type", "S")
        type_counts[bt] = type_counts.get(bt, 0) + 1

    # Ring detection via NetworkX cycle basis
    try:
        cycles = nx.cycle_basis(G)
    except Exception:
        cycles = []
    ring_sizes = sorted(len(c) for c in cycles)

    # Check connectivity
    is_connected = nx.is_connected(G) if len(G) > 0 else True

    return BondGraphSummary(
        num_bonds=G.number_of_edges(),
        bond_type_counts=type_counts,
        num_rings=len(cycles),
        ring_sizes=ring_sizes,
        is_connected=is_connected,
    )


def _compute_rigidity(parsed: ParsedXYZ) -> bool:
    """Determine if the building block is rigid.

    Rigid = no rotatable single bonds in the real atom graph.
    A single bond is rotatable if:
      - It is type "S" (single)
      - Neither endpoint is H
      - Neither endpoint is a metal (metal coordination is rigid)
      - Neither endpoint is a halogen (terminal, no rotational freedom)
      - Neither endpoint is a terminal atom (degree=1, no conformational change)
      - The bond is NOT part of a ring (NetworkX cycle basis)
    """
    G = _build_real_graph(parsed)

    h_indices = {a.index for a in parsed.atoms if a.element == "H"}
    metal_indices = {a.index for a in parsed.atoms if a.element in METAL_SET}
    halogen_indices = {a.index for a in parsed.atoms if a.element in HALOGEN_SET}

    # A bond is in a ring iff it is NOT a bridge.
    # Bridge = removing it disconnects the graph.
    bridge_edges = set()
    try:
        for u, v in nx.bridges(G):
            bridge_edges.add((min(u, v), max(u, v)))
    except Exception:
        pass

    # Check each single bond
    for u, v, data in G.edges(data=True):
        if data.get("bond_type") != "S":
            continue
        # Skip bonds involving H
        if u in h_indices or v in h_indices:
            continue
        # Skip metal coordination bonds
        if u in metal_indices or v in metal_indices:
            continue
        # Skip bonds involving halogens (terminal, no rotation)
        if u in halogen_indices or v in halogen_indices:
            continue
        # Skip bonds where either atom has degree ≤ 2
        # (degree=1: terminal, degree=2: no conformational change from rotation)
        if G.degree(u) <= 2 or G.degree(v) <= 2:
            continue
        # Skip ring bonds (non-bridge = in a ring)
        pair = (min(u, v), max(u, v))
        if pair not in bridge_edges:
            continue
        # This is a rotatable single bond (bridge, non-terminal, non-H/metal/halogen)
        return False

    return True


def _degree_of_unsaturation(counts: Counter) -> float:
    """Compute degree of unsaturation (DoU) from element counts.

    Standard formula for organics: DoU = (2C + 2 - H + N - Halogen) / 2
    Extended: also accounts for P (+1 like N), S (neutral like C).
    """
    c = counts.get("C", 0)
    h = counts.get("H", 0)
    n = counts.get("N", 0)
    p = counts.get("P", 0)
    halogens = sum(counts.get(x, 0) for x in HALOGEN_SET)

    dou = (2 * c + 2 - h + n + p - halogens) / 2.0
    return round(dou, 1)


def _check_planarity(atoms: tuple[Atom, ...]) -> bool:
    """Check if all real atoms are approximately coplanar.

    Simple heuristic: if the range of z-coordinates is < 0.5 Å,
    consider the molecule planar.
    """
    if len(atoms) < 3:
        return True
    z_values = [a.z for a in atoms]
    z_range = max(z_values) - min(z_values)
    return z_range < 0.5


def _max_dummy_distance(dummies: tuple[Atom, ...]) -> float:
    """Max Euclidean distance between any two dummy atoms."""
    max_dist = 0.0
    for i in range(len(dummies)):
        for j in range(i + 1, len(dummies)):
            d = math.sqrt(
                (dummies[i].x - dummies[j].x) ** 2 +
                (dummies[i].y - dummies[j].y) ** 2 +
                (dummies[i].z - dummies[j].z) ** 2
            )
            max_dist = max(max_dist, d)
    return round(max_dist, 4)


def _topological_distance_matrix(parsed: ParsedXYZ) -> Optional[list[list[int]]]:
    """Pairwise shortest graph paths between all dummy atom anchors.

    Returns an NxN matrix where N = number of connection points.
    Each entry is the shortest hop count through real atoms only.
    -1 indicates unreachable (disconnected).
    """
    n_dummies = len(parsed.dummy_atoms)
    if n_dummies < 2:
        return None

    dummy_indices = {a.index for a in parsed.dummy_atoms}

    # Map each dummy to its anchor (the real atom it bonds to)
    dummy_to_anchor: dict[int, int] = {}
    for d in parsed.dummy_atoms:
        for b in parsed.bonds:
            if b.atom1 == d.index and b.atom2 not in dummy_indices:
                dummy_to_anchor[d.index] = b.atom2
                break
            elif b.atom2 == d.index and b.atom1 not in dummy_indices:
                dummy_to_anchor[d.index] = b.atom1
                break

    # Build adjacency for real atoms
    adj: dict[int, set[int]] = {}
    for b in parsed.bonds:
        if b.atom1 in dummy_indices or b.atom2 in dummy_indices:
            continue
        adj.setdefault(b.atom1, set()).add(b.atom2)
        adj.setdefault(b.atom2, set()).add(b.atom1)

    # BFS from each anchor
    anchors = [dummy_to_anchor.get(d.index) for d in parsed.dummy_atoms]
    matrix = [[-1] * n_dummies for _ in range(n_dummies)]

    for i in range(n_dummies):
        matrix[i][i] = 0
        if anchors[i] is None:
            continue
        # BFS from anchor[i]
        dist_map = _bfs_distances(anchors[i], adj)
        for j in range(n_dummies):
            if i == j or anchors[j] is None:
                continue
            matrix[i][j] = dist_map.get(anchors[j], -1)

    return matrix


def _bfs_distances(start: int, adj: dict[int, set[int]]) -> dict[int, int]:
    """BFS from start node, return distance map to all reachable nodes."""
    visited = {start: 0}
    queue = [(start, 0)]
    qi = 0
    while qi < len(queue):
        node, dist = queue[qi]
        qi += 1
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                visited[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
    return visited


def _infer_geometry(dummies: tuple[Atom, ...]) -> Optional[str]:
    """Infer geometry from the spatial arrangement of connection points.

    Uses connection count and coplanarity/angle analysis.
    """
    n = len(dummies)
    if n == 0:
        return None

    coords = [(d.x, d.y, d.z) for d in dummies]

    if n == 2:
        return "linear"

    if n == 3:
        if _are_coplanar(coords):
            return "trigonal_planar"
        return "trigonal_pyramidal"

    if n == 4:
        if _are_coplanar(coords):
            return "square_planar"
        return "tetrahedral"

    if n == 5:
        return "5-c_generic"

    if n == 6:
        # Check if roughly octahedral (3 orthogonal axes)
        return "octahedral"

    if n == 7:
        return "7-c_generic"

    if n == 8:
        return "8-c_generic"

    return f"{n}-c_generic"


def _are_coplanar(coords: list[tuple[float, float, float]], tol: float = 0.5) -> bool:
    """Check if a set of 3D points are approximately coplanar."""
    if len(coords) <= 3:
        return True

    z_values = [c[2] for c in coords]
    z_range = max(z_values) - min(z_values)
    if z_range < tol:
        return True

    # More robust: check if all points lie near a single plane
    # using SVD or cross-product normal vector
    # For now, use the simple z-range heuristic
    # (Most PorMake BBs are oriented with the principal plane in xy)
    return False


def _extract_metal_coordination(parsed: ParsedXYZ) -> list[MetalCoordination]:
    """Extract coordination environment for each metal atom."""
    dummy_indices = {a.index for a in parsed.dummy_atoms}
    metal_atoms = [a for a in parsed.real_atoms if a.element in METAL_SET]

    coordinations = []
    for metal in metal_atoms:
        bonded = []
        for b in parsed.bonds:
            partner_idx = None
            if b.atom1 == metal.index:
                partner_idx = b.atom2
            elif b.atom2 == metal.index:
                partner_idx = b.atom1

            if partner_idx is not None and partner_idx not in dummy_indices:
                partner = parsed.atoms[partner_idx]
                bonded.append({
                    "index": partner_idx,
                    "element": partner.element,
                    "bond_type": b.bond_type,
                })

        coordinations.append(MetalCoordination(
            metal_index=metal.index,
            element=metal.element,
            bonded_atoms=bonded,
            coordination_number=len(bonded),
        ))

    return coordinations
