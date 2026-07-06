"""Deterministic node-specific enrichment: sbu_type and ligand_chemistry.

Derives SBU classification and ligand donor chemistry from existing
Layer 1 metal_coordination data. All results are deterministic — no LLM.
"""

from __future__ import annotations

from collections import Counter

from .config import METAL_SET
from .schema import BBRecord, BBType


def enrich_node_fields(record: BBRecord) -> BBRecord:
    """Derive sbu_type and ligand_chemistry for node BBs.

    Modifies record.layer2_semantics in-place and returns the record.
    Only applies to nodes (bb_type == "node"). Edges are left unchanged.
    """
    if record.bb_type != BBType.NODE:
        return record

    facts = record.layer1_facts
    l2 = record.layer2_semantics

    # ── Ligand chemistry ──
    l2.ligand_chemistry = _derive_ligand_chemistry(facts)

    # ── SBU type ──
    l2.sbu_type = _derive_sbu_type(facts)

    return record


def _derive_ligand_chemistry(facts) -> list[str]:
    """Extract unique non-metal donor atom types from metal_coordination.

    For each metal center, collect the elements of bonded atoms that are
    NOT themselves metals. Returns sorted unique list.
    E.g., ["N", "O"] for a metal coordinated by nitrogen and oxygen donors.
    """
    if not facts.metal_coordination:
        return []

    donor_elements = set()
    for mc in facts.metal_coordination:
        for bonded in mc.bonded_atoms:
            elem = bonded["element"]
            # Exclude metal-metal bonds and dummy atoms
            if elem not in METAL_SET and elem != "X":
                donor_elements.add(elem)

    return sorted(donor_elements)


def _derive_sbu_type(facts) -> str:
    """Build a systematic SBU type string from Layer 1 facts.

    Format for metal nodes:
        {Metal}{count}[_{Metal2}{count2}]_{DonorFormula}_{geometry}_{connectivity}c
        e.g., "Zn4_O12_tetrahedral_4c", "Fe1_N4Cl1_tetrahedral_4c"

    Format for organic nodes:
        Organic_{geometry}_{connectivity}c
        e.g., "Organic_square_planar_4c"
    """
    connectivity = facts.connection_points.count
    geometry = facts.geometry_inferred or "unknown"

    if not facts.has_metal or not facts.metal_coordination:
        return f"Organic_{geometry}_{connectivity}c"

    # ── Metal formula part ──
    # Count metals by element
    metal_counts = Counter()
    for mc in facts.metal_coordination:
        metal_counts[mc.element] += 1

    # Format: sorted by element symbol, e.g., "Cu4W1" or "Zn4"
    metal_parts = []
    for elem in sorted(metal_counts.keys()):
        count = metal_counts[elem]
        metal_parts.append(f"{elem}{count}")
    metal_str = "".join(metal_parts)

    # ── Donor formula part ──
    # Count non-metal donor atoms across all metal centers (unique bonds)
    donor_counts = Counter()
    seen_bonds = set()  # avoid double-counting shared ligands
    for mc in facts.metal_coordination:
        for bonded in mc.bonded_atoms:
            elem = bonded["element"]
            idx = bonded["index"]
            if elem not in METAL_SET and elem != "X":
                bond_key = (mc.metal_index, idx)
                if bond_key not in seen_bonds:
                    seen_bonds.add(bond_key)
                    donor_counts[elem] += 1

    # Format: Hill-like ordering (C first, then alphabetical), e.g., "N4O2"
    donor_parts = []
    for elem in sorted(donor_counts.keys()):
        count = donor_counts[elem]
        donor_parts.append(f"{elem}{count}" if count > 1 else elem)
    donor_str = "".join(donor_parts) if donor_parts else "none"

    return f"{metal_str}_{donor_str}_{geometry}_{connectivity}c"
