"""Phase 4: Metal node enrichment from QMOF source JSONs.

Extracts metal composition, nuclearity, geometry, oxidation states,
and functional groups from qmof_global_jsons_v2/ analysis files.

These 20,372 source JSONs contain rich metal node data that was
completely IGNORED in v1 (metal_node: {} was always empty).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from .schema_v2 import MetalNode, IssueLogEntry


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
SOURCE_JSON_DIR = BASE_DIR / "qmof" / "_source_data" / "qmof_global_jsons_v2"

# Known metal elements for formula parsing (subset for cross-validation)
FORMULA_METALS = frozenset({
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "Rb", "Sr", "Y", "Zr", "Nb",
    "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "Ac", "Th", "Pa", "U", "Np", "Pu",
})

# Regex for parsing element symbols from chemical formulas
# Matches uppercase letter optionally followed by lowercase letter
ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")


# =============================================================================
# Source JSON loader
# =============================================================================

def find_source_json(qmof_id: str) -> Optional[Path]:
    """Find the source analysis JSON for a given QMOF ID.

    Returns the Path if found, None otherwise.
    """
    json_path = SOURCE_JSON_DIR / f"{qmof_id}_analysis.json"
    if json_path.exists():
        return json_path
    return None


def load_source_json(qmof_id: str) -> tuple[Optional[dict], Optional[str]]:
    """Load source JSON and compute SHA256 hash.

    Returns (data_dict, sha256_hex) or (None, None) if not found.
    """
    json_path = find_source_json(qmof_id)
    if json_path is None:
        return None, None

    try:
        raw_bytes = json_path.read_bytes()
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        data = json.loads(raw_bytes.decode("utf-8"))
        return data, sha256
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None, None


# =============================================================================
# Metal node extraction
# =============================================================================

def extract_metal_node(
    source_json: dict,
    issues: list[IssueLogEntry],
    qmof_id: str,
) -> MetalNode:
    """Extract MetalNode from source JSON's metal_node field.

    Defensively accesses nested fields with .get() — never crashes on
    missing or malformed data. Logs issues for any missing expected fields.

    Args:
        source_json: Parsed JSON dict from source analysis file.
        issues: Mutable list to append IssueLogEntry objects to.
        qmof_id: For issue logging context.

    Returns:
        Populated MetalNode dataclass.
    """
    metal_node_data = source_json.get("metal_node", {})

    if not metal_node_data:
        issues.append(IssueLogEntry(
            field="metal_node",
            severity="warning",
            message="Source JSON has no metal_node data",
            source="source_json",
        ))
        return MetalNode()

    # --- Composition ---
    composition = metal_node_data.get("composition", {})
    metals = composition.get("metals", [])
    nuclearity = composition.get("nuclearity", 0)

    if not metals:
        issues.append(IssueLogEntry(
            field="metal_node.composition.metals",
            severity="warning",
            message="No metals found in source JSON composition",
            source="source_json",
        ))

    # Ensure metals is a list of strings
    if isinstance(metals, list):
        metals = [str(m) for m in metals]
    else:
        metals = []
        issues.append(IssueLogEntry(
            field="metal_node.composition.metals",
            severity="warning",
            message=f"metals field is not a list: {type(metals).__name__}",
            source="source_json",
        ))

    if not isinstance(nuclearity, (int, float)):
        nuclearity = 0

    # --- Topology Features ---
    topo = metal_node_data.get("topology_features", {})
    connectivity = topo.get("connectivity_points")
    geometry = topo.get("geometry")
    sbu_type = topo.get("sbu_type")
    ligand_chemistry = topo.get("ligand_chemistry", [])

    if isinstance(connectivity, (int, float)):
        connectivity = int(connectivity)
    else:
        connectivity = None

    if isinstance(geometry, str):
        geometry = _normalize_geometry(geometry)
    else:
        geometry = None

    if isinstance(sbu_type, str):
        sbu_type = sbu_type
    else:
        sbu_type = metal_node_data.get("value")  # fallback: top-level value

    if isinstance(ligand_chemistry, list):
        ligand_chemistry = [str(lc) for lc in ligand_chemistry]
    else:
        ligand_chemistry = []

    # --- Chemistry ---
    chemistry = metal_node_data.get("chemistry", {})
    net_charge = chemistry.get("net_charge", 0)
    oxidation_states = chemistry.get("oxidation_states")
    has_open_metal_sites = chemistry.get("has_open_metal_sites")
    spin_state = chemistry.get("spin_state")

    if not isinstance(net_charge, (int, float)):
        net_charge = 0
    else:
        net_charge = int(net_charge)

    if isinstance(oxidation_states, dict):
        # Ensure keys are strings, values are ints
        oxidation_states = {
            str(k): int(v) for k, v in oxidation_states.items()
            if isinstance(v, (int, float))
        }
    else:
        oxidation_states = None

    if not isinstance(has_open_metal_sites, bool):
        has_open_metal_sites = None

    if isinstance(spin_state, str):
        spin_state = spin_state
    else:
        spin_state = None

    # --- Functional Groups (from metal coordination sphere) ---
    fg_data = chemistry.get("functional_groups", {})
    coordinating_groups = _safe_list(fg_data.get("coordinating_groups", []))
    linker_substituents = _safe_list(fg_data.get("linker_substituents", []))
    linker_backbone = _safe_list(fg_data.get("linker_backbone", []))
    metal_terminal_ligands = _safe_list(fg_data.get("metal_terminal_ligands", []))

    return MetalNode(
        metals=sorted(metals),
        nuclearity=int(nuclearity),
        connectivity=connectivity,
        geometry=geometry,
        sbu_type=sbu_type,
        oxidation_states=oxidation_states,
        ligand_chemistry=sorted(ligand_chemistry),
        has_open_metal_sites=has_open_metal_sites,
        spin_state=spin_state,
        net_charge=net_charge,
        coordinating_groups=sorted(coordinating_groups),
        linker_substituents=sorted(linker_substituents),
        linker_backbone=sorted(linker_backbone),
        metal_terminal_ligands=sorted(metal_terminal_ligands),
    )


# =============================================================================
# Cross-validation
# =============================================================================

def extract_metals_from_formula(formula: str) -> list[str]:
    """Extract metal element symbols from a chemical formula string.

    Parses element symbols (uppercase + optional lowercase) and filters
    for known metals. Does NOT parse counts.

    Example: "Cu2C14H24N4O6" → ["Cu"]
    """
    if not formula:
        return []

    elements = ELEMENT_RE.findall(formula)
    metals = sorted(set(e for e in elements if e in FORMULA_METALS))
    return metals


def cross_validate_metals(
    formula_metals: list[str],
    json_metals: list[str],
    qmof_id: str,
    issues: list[IssueLogEntry],
) -> bool:
    """Compare metals from formula vs source JSON.

    Returns True if consistent, False if discrepancy found.
    """
    formula_set = set(formula_metals)
    json_set = set(json_metals)

    if formula_set == json_set:
        return True

    # Log discrepancy
    in_formula_only = formula_set - json_set
    in_json_only = json_set - formula_set

    parts = []
    if in_formula_only:
        parts.append(f"in formula only: {sorted(in_formula_only)}")
    if in_json_only:
        parts.append(f"in source JSON only: {sorted(in_json_only)}")

    issues.append(IssueLogEntry(
        field="metal_composition",
        severity="warning",
        message=f"Metal discrepancy: {'; '.join(parts)}",
        source="cross_validation",
    ))
    return False


# =============================================================================
# Helpers
# =============================================================================

def _safe_list(val) -> list[str]:
    """Ensure value is a list of strings."""
    if isinstance(val, list):
        return [str(v) for v in val if v]
    return []


def _normalize_geometry(geometry: str) -> str:
    """Normalize geometry labels for consistency.

    Converts underscore-heavy notation to cleaner form while preserving
    the coordination number prefix.

    Examples:
        "CN_5_Geometry" → "CN5_Generic"
        "CN_4_Tetrahedral" → "CN4_Tetrahedral"
        "CN_6_Octahedral" → "CN6_Octahedral"
    """
    if not geometry:
        return geometry

    # Pattern: CN_X_Name or CN_XX_Name
    m = re.match(r"CN_?(\d+)_?(.*)", geometry)
    if m:
        cn = m.group(1)
        name = m.group(2).strip("_").replace("_", " ").strip()
        if not name or name.lower() == "geometry":
            name = "Generic"
        return f"CN{cn}_{name.replace(' ', '_')}"

    return geometry


# =============================================================================
# Main enrichment function
# =============================================================================

def enrich_metal_node(
    qmof_id: str,
    formula: str = "",
) -> tuple[MetalNode, Optional[str], Optional[str], list[IssueLogEntry]]:
    """Full metal node enrichment for a single QMOF record.

    Loads source JSON, extracts metal node data, cross-validates against
    formula, and returns results with issue log.

    Args:
        qmof_id: QMOF identifier (e.g., "qmof-0000295")
        formula: Chemical formula from CSV for cross-validation

    Returns:
        (metal_node, source_json_filename, sha256, issues)
    """
    issues: list[IssueLogEntry] = []

    # Load source JSON
    source_json, sha256 = load_source_json(qmof_id)

    if source_json is None:
        issues.append(IssueLogEntry(
            field="source_json",
            severity="warning",
            message=f"Source JSON not found for {qmof_id}",
            source="source_json",
        ))
        return MetalNode(), None, None, issues

    source_filename = f"{qmof_id}_analysis.json"

    # Extract metal node
    metal_node = extract_metal_node(source_json, issues, qmof_id)

    # Cross-validate metals with formula
    if formula:
        formula_metals = extract_metals_from_formula(formula)
        if formula_metals or metal_node.metals:
            cross_validate_metals(
                formula_metals, metal_node.metals, qmof_id, issues
            )

    return metal_node, source_filename, sha256, issues
