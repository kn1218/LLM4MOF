"""RASPA3 utilities shared between run_raspa.py and test_analysis.py."""

import os
import re
from collections import defaultdict
from typing import Dict, Optional


_FLOAT_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

_LENGTHS_RE = re.compile(rf"Lengths:\s*({_FLOAT_RE})\s+({_FLOAT_RE})\s+({_FLOAT_RE})")
_FRAMEWORK_MASS_RE = re.compile(rf"Framework Mass:\s*({_FLOAT_RE})")
_GENERIC_MASS_AMU_RE = re.compile(rf"mass:\s*({_FLOAT_RE})\s+amu")
_NUM_UNIT_CELLS_RE = re.compile(r"NumberOfUnitCells\s+(\d+)")
_FRAMEWORK_DENSITY_RE = re.compile(
    rf"Framework Density:\s*({_FLOAT_RE})\s*\[kg/m\^3\]"
    rf"(?:\s+({_FLOAT_RE})\s*\[cm\^3/g\])?"
)
_AVG_ABS_MOLECULES_RE = re.compile(
    rf"Average loading absolute \[molecules/unit cell\]\s*({_FLOAT_RE})"
)
_AVG_ABS_MOL_KG_RE = re.compile(
    r"Abs\. loading average\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\+/-?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\[mol/kg[/-]?framework\]"
)
_AVG_ABS_MG_G_RE = re.compile(
    r"Abs\. loading average\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\+/-?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\[mg/g[/-]?framework\]"
)
_ABS_STEP_RE = re.compile(
    rf"absolute adsorption:.*?({_FLOAT_RE})"
    rf"(?:\s+\(avg\.\s+{_FLOAT_RE}\))?\s+\[mol/uc\],\s*"
    rf"({_FLOAT_RE})(?:\s+\(avg\.\s+{_FLOAT_RE}\))?\s+\[mol/kg\],\s*"
    rf"({_FLOAT_RE})(?:\s+\(avg\.\s+{_FLOAT_RE}\))?\s+\[mg/g\]"
)


def _merge_element_counts(
    target: Dict[str, float], source: Dict[str, float], factor: float = 1.0
) -> None:
    """Merge element counts into target with a multiplier."""
    for symbol, count in source.items():
        target[symbol] += count * factor


def _parse_formula_group(tokens, start_idx=0):
    """Parse one formula group recursively.

    Supports:
    - element symbols: H, C, Na, Fe ...
    - integer multipliers: H2, (OH)2
    - nested brackets: (), [], {}
    """
    counts = defaultdict(float)
    i = start_idx

    opening_to_closing = {"(": ")", "[": "]", "{": "}"}
    closing_tokens = {")", "]", "}"}

    while i < len(tokens):
        tok = tokens[i]

        if tok in closing_tokens:
            return counts, i + 1

        if tok in opening_to_closing:
            subgroup_counts, i = _parse_formula_group(tokens, i + 1)

            multiplier = 1
            if i < len(tokens) and tokens[i].isdigit():
                multiplier = int(tokens[i])
                i += 1

            _merge_element_counts(counts, subgroup_counts, multiplier)
            continue

        if re.fullmatch(r"[A-Z][a-z]?", tok):
            symbol = tok
            i += 1

            multiplier = 1
            if i < len(tokens) and tokens[i].isdigit():
                multiplier = int(tokens[i])
                i += 1

            counts[symbol] += multiplier
            continue

        raise ValueError(f"Unexpected token in formula: {tok}")

    return counts, i


def parse_formula(formula: str) -> Dict[str, float]:
    """Parse a chemical formula string into element counts.

    Examples:
        H2 -> {"H": 2}
        C6H6O -> {"C": 6, "H": 6, "O": 1}
        Mg(OH)2 -> {"Mg": 1, "O": 2, "H": 2}
        Al2(SO4)3 -> {"Al": 2, "S": 3, "O": 12}

    Also supports hydrate-style separators:
        CuSO4·5H2O
        CuSO4.5H2O
    """
    if not formula or not isinstance(formula, str):
        raise ValueError("Formula must be a non-empty string.")

    formula = formula.strip().replace(" ", "")
    if not formula:
        raise ValueError("Formula must be a non-empty string.")

    # Split hydrate-like parts: CuSO4·5H2O
    parts = re.split(r"[·.]", formula)

    total_counts = defaultdict(float)
    token_pattern = re.compile(r"[A-Z][a-z]?|\d+|[()\[\]{}]")

    for part in parts:
        if not part:
            continue

        # Allow leading coefficient for a whole part: 5H2O
        leading_multiplier = 1
        m = re.match(r"^(\d+)(.*)$", part)
        if m:
            leading_multiplier = int(m.group(1))
            part = m.group(2)
            if not part:
                raise ValueError(f"Invalid formula segment: {formula}")

        tokens = token_pattern.findall(part)
        if "".join(tokens) != part:
            raise ValueError(f"Unsupported formula format: {formula}")

        part_counts, end_idx = _parse_formula_group(tokens, 0)
        if end_idx != len(tokens):
            raise ValueError(f"Failed to parse full formula: {formula}")

        _merge_element_counts(total_counts, part_counts, leading_multiplier)

    return dict(total_counts)


def get_molecular_weight_from_formula(formula: str) -> Optional[float]:
    """Calculate molecular weight from a chemical formula.

    Returns:
        Molecular weight in g/mol, or None if ASE is unavailable.
    """
    try:
        from ase.data import atomic_masses, atomic_numbers
    except ImportError:
        return None

    try:
        element_counts = parse_formula(formula)
    except Exception:
        return None

    mw = 0.0
    for symbol, count in element_counts.items():
        if symbol not in atomic_numbers:
            return None
        atomic_number = atomic_numbers[symbol]
        mw += float(atomic_masses[atomic_number]) * count

    return mw


DEFAULT_ADSORBATE_FORMULA = "H2"
DEFAULT_ADSORBATE_MW_G_MOL = get_molecular_weight_from_formula(
    DEFAULT_ADSORBATE_FORMULA
)
if DEFAULT_ADSORBATE_MW_G_MOL is None:
    DEFAULT_ADSORBATE_MW_G_MOL = 2.016


def get_density_from_cif(cif_path: str) -> Optional[float]:
    """Calculate crystal density from CIF using ASE."""
    try:
        from ase.io import read
    except ImportError:
        return None

    try:
        atoms = read(cif_path)
        if isinstance(atoms, list):
            atoms = atoms[0]

        mass = atoms.get_masses().sum()
        volume = atoms.get_volume()

        if volume > 0:
            return mass / volume * 1.66053906660
    except Exception:
        pass

    return None


def _find_output_file(search_dir: str) -> Optional[str]:
    """Find the newest RASPA output file in a directory."""
    if not os.path.isdir(search_dir):
        return None

    candidates = []
    for filename in os.listdir(search_dir):
        if filename.startswith("output_") and (
            filename.endswith(".txt") or filename.endswith(".data")
        ):
            candidates.append(os.path.join(search_dir, filename))

    if not candidates:
        return None

    return max(candidates, key=os.path.getmtime)


def parse_output(
    output_dir: str,
    cif_path: Optional[str] = None,
    adsorbate_mw_g_mol: float = DEFAULT_ADSORBATE_MW_G_MOL,
):
    """Parse a RASPA output file and extract simulation results.

    Args:
        output_dir: Directory containing RASPA output files.
        cif_path: Optional CIF path used as a fallback for framework density.
        adsorbate_mw_g_mol: Adsorbate molecular weight in g/mol.
            Default is 2.016 for H2.

    Returns:
        Dictionary with parsed results or None if parsing fails.

    Notes:
        - loading_mol_kg is gravimetric uptake: mol adsorbate / kg framework
        - loading_g_L is volumetric uptake: g adsorbate / L framework
          computed as:
              loading_mol_kg * adsorbate_mw_g_mol * framework_density_kg_L
        - framework_density_g_cm3 and framework_density_kg_L are numerically equal
        - RASPA's Framework Density (kg/m³) is often incorrect due to supercell mismatch.
          Use get_density_from_cif() for accurate density values.
    """
    output_subdir = os.path.join(output_dir, "output")
    search_dir = output_subdir if os.path.isdir(output_subdir) else output_dir

    output_file = _find_output_file(search_dir)
    if not output_file or not os.path.exists(output_file):
        return None

    with open(output_file, "r") as f:
        lines = f.readlines()

    result = {}

    unit_cell_a = unit_cell_b = unit_cell_c = None
    framework_mass_amu = None
    num_unit_cells = None

    framework_density_kg_m3 = None
    framework_specific_volume_cm3_g = None

    avg_loading_molecules = None
    avg_loading_mol_kg = None
    avg_loading_mg_g = None

    last_step_molecules = None
    last_step_mol_kg = None
    last_step_mg_g = None

    for line in lines:
        if unit_cell_a is None:
            match = _LENGTHS_RE.search(line)
            if match:
                unit_cell_a = float(match.group(1))
                unit_cell_b = float(match.group(2))
                unit_cell_c = float(match.group(3))

        if framework_mass_amu is None:
            match = _FRAMEWORK_MASS_RE.search(line)
            if match:
                framework_mass_amu = float(match.group(1))
            else:
                if "[-]" not in line:
                    match = _GENERIC_MASS_AMU_RE.search(line)
                    if match:
                        framework_mass_amu = float(match.group(1))

        if num_unit_cells is None:
            match = _NUM_UNIT_CELLS_RE.search(line)
            if match:
                num_unit_cells = int(match.group(1))

        match = _FRAMEWORK_DENSITY_RE.search(line)
        if match:
            framework_density_kg_m3 = float(match.group(1))
            if match.group(2) is not None:
                framework_specific_volume_cm3_g = float(match.group(2))

        match = _AVG_ABS_MOLECULES_RE.search(line)
        if match:
            avg_loading_molecules = float(match.group(1))

        match = _AVG_ABS_MOL_KG_RE.search(line)
        if match:
            avg_loading_mol_kg = float(match.group(1))

        match = _AVG_ABS_MG_G_RE.search(line)
        if match:
            avg_loading_mg_g = float(match.group(1))

        match = _ABS_STEP_RE.search(line)
        if match:
            last_step_molecules = float(match.group(1))
            last_step_mol_kg = float(match.group(2))
            last_step_mg_g = float(match.group(3))

    # Prefer average values over last step values
    if avg_loading_molecules is not None:
        result["loading_molecules"] = avg_loading_molecules
    elif last_step_molecules is not None:
        result["loading_molecules"] = last_step_molecules

    if avg_loading_mol_kg is not None:
        result["loading_mol_kg"] = avg_loading_mol_kg
    elif last_step_mol_kg is not None:
        result["loading_mol_kg"] = last_step_mol_kg

    if avg_loading_mg_g is not None:
        result["loading_mg_g"] = avg_loading_mg_g
    elif last_step_mg_g is not None:
        result["loading_mg_g"] = last_step_mg_g
    elif result.get("loading_mol_kg") is not None:
        # mol/kg * g/mol = g/kg, which is numerically equal to mg/g
        result["loading_mg_g"] = result["loading_mol_kg"] * adsorbate_mw_g_mol

    # Calculate density: prefer CIF-based over RASPA-reported
    # RASPA's Framework Density often has issues with supercell calculations
    framework_density_g_cm3 = None

    if cif_path:
        # Use CIF-based density (more reliable)
        framework_density_g_cm3 = get_density_from_cif(cif_path)
        if framework_density_g_cm3 is not None:
            framework_density_kg_m3 = framework_density_g_cm3 * 1000.0

    # Fallback to RASPA-reported density if CIF not available
    if framework_density_g_cm3 is None and framework_density_kg_m3 is not None:
        # Note: RASPA's density may be unreliable due to supercell mismatch
        framework_density_g_cm3 = framework_density_kg_m3 / 1000.0

    if framework_density_g_cm3 is not None:
        result["framework_density_g_cm3"] = framework_density_g_cm3
        result["framework_density_kg_L"] = framework_density_g_cm3

    if framework_density_kg_m3 is not None:
        result["framework_density_kg_m3"] = framework_density_kg_m3

    if framework_specific_volume_cm3_g is None and framework_density_g_cm3:
        framework_specific_volume_cm3_g = 1.0 / framework_density_g_cm3

    if framework_specific_volume_cm3_g is not None:
        result["framework_specific_volume_cm3_g"] = framework_specific_volume_cm3_g

    # Calculate volumetric loading (g/L)
    if result.get("loading_mol_kg") is not None and framework_density_g_cm3 is not None:
        result["loading_g_L"] = (
            result["loading_mol_kg"] * adsorbate_mw_g_mol * framework_density_g_cm3
        )

    if unit_cell_a is not None and unit_cell_b is not None and unit_cell_c is not None:
        result["unit_cell"] = {
            "a": unit_cell_a,
            "b": unit_cell_b,
            "c": unit_cell_c,
        }

    if num_unit_cells is not None:
        result["num_unit_cells"] = num_unit_cells

    if framework_mass_amu is not None:
        result["framework_mass_amu"] = framework_mass_amu

    return result if result else None


# ======================================================================
# Xe/Kr mixture parsing
# ======================================================================

_COMPONENT_HEADER_RE = re.compile(r"Component\s+\d+\s+[\[\(](.+?)[\]\)]")
_ABS_STEP_MOLKG_RE = re.compile(
    rf"absolute adsorption:.*?({_FLOAT_RE})(?:\s+\(avg\.\s+{_FLOAT_RE}\))?\s+\[mol/uc\],\s*"
    rf"({_FLOAT_RE})(?:\s+\(avg\.\s+{_FLOAT_RE}\))?\s+\[mol/kg\]"
)
# RASPA3 multiline format: "    Abs. loading average   VALUE +/- ERR [mol/kg-framework]"
_ABS_LOADING_AVG_MOLKG_RE = re.compile(
    rf"Abs\.\s+loading\s+average\s+({_FLOAT_RE}).*\[mol/kg"
)


def parse_output_mixture(
    output_dir: str,
    xe_molfrac: float = 0.20,
    cif_path: Optional[str] = None,
) -> Optional[dict]:
    """Parse RASPA output for a 2-component Xe/Kr mixture and compute selectivity.

    Args:
        output_dir: Directory containing RASPA output files.
        xe_molfrac: Xe mole fraction in the gas phase (Kr = 1 - xe_molfrac).
        cif_path: Optional CIF path for framework density.

    Returns:
        Dictionary with:
            xe_loading_mol_kg, kr_loading_mol_kg (gravimetric)
            xe_loading_g_L, kr_loading_g_L (volumetric, if density available)
            selectivity_xe_kr = (N_Xe/N_Kr) / (y_Xe/y_Kr)
        or None if parsing fails.
    """
    output_subdir = os.path.join(output_dir, "output")
    search_dir = output_subdir if os.path.isdir(output_subdir) else output_dir

    output_file = _find_output_file(search_dir)
    if not output_file or not os.path.exists(output_file):
        return None

    with open(output_file, "r") as f:
        lines = f.readlines()

    # MW constants
    xe_mw = 131.29   # g/mol
    kr_mw = 83.798   # g/mol
    kr_molfrac = 1.0 - xe_molfrac

    # Per-component loading accumulators (last step values, updated per component block)
    loadings: Dict[str, float] = {}
    current_component: Optional[str] = None

    for line in lines:
        m = _COMPONENT_HEADER_RE.search(line)
        if m:
            current_component = m.group(1).strip()
            continue

        if current_component is not None:
            m = _ABS_STEP_MOLKG_RE.search(line)
            if m:
                loadings[current_component] = float(m.group(2))
            else:
                m = _ABS_LOADING_AVG_MOLKG_RE.search(line)
                if m:
                    loadings[current_component] = float(m.group(1))

    xe_mol_kg = loadings.get("Xe")
    kr_mol_kg = loadings.get("Kr")

    if xe_mol_kg is None or kr_mol_kg is None:
        return None

    result: dict = {
        "xe_loading_mol_kg": xe_mol_kg,
        "kr_loading_mol_kg": kr_mol_kg,
    }

    # Selectivity: (N_Xe/N_Kr) / (y_Xe/y_Kr)
    if kr_mol_kg > 0 and kr_molfrac > 0 and xe_molfrac > 0:
        result["selectivity_xe_kr"] = (xe_mol_kg / kr_mol_kg) / (xe_molfrac / kr_molfrac)

    # Volumetric loading if density is available
    density_g_cm3 = get_density_from_cif(cif_path) if cif_path else None
    if density_g_cm3 is not None:
        result["xe_loading_g_L"] = xe_mol_kg * xe_mw * density_g_cm3
        result["kr_loading_g_L"] = kr_mol_kg * kr_mw * density_g_cm3
        result["framework_density_g_cm3"] = density_g_cm3

    return result
