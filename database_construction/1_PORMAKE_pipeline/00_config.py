"""Constants, atomic data, and configuration for the BB pipeline."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # PORMAKE/
BBS_DIR = PROJECT_ROOT / "_source_data" / "bbs"
OLD_BB_DICT_PATH = PROJECT_ROOT / "_source_data" / "pormake_bb_dictionary_v4.json"
OLD_META_DIR = PROJECT_ROOT / "_legacy" / "BuildingBlock_meta_data_v7_20260305"
VOCABULARY_PATH = PROJECT_ROOT / "_legacy" / "combined_hierarchical_vocabulary_20260310.json"
OUTPUT_DIR = PROJECT_ROOT / "bb_metadata_v8"

# ── Schema Version ────────────────────────────────────────────────────
SCHEMA_VERSION = "1.0.0"
PIPELINE_VERSION = "1.0.0"

# ── Dummy Atom Config ─────────────────────────────────────────────────
DUMMY_ELEMENT = "X"
DUMMY_MASS = 0.0
# Lawrencium placeholder for RDKit (avoids valence errors on dummy atoms)
LR_ATOMIC_NUM = 103

# ── Bond Type Mapping ─────────────────────────────────────────────────
BOND_TYPE_MAP = {
    "S": "SINGLE",
    "D": "DOUBLE",
    "T": "TRIPLE",
    "A": "AROMATIC",
}

# ── Metals ─────────────────────────────────────────────────────────────
# Includes metalloids B and Si which appear in PorMake SBUs intentionally.
METAL_SET = frozenset({
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "Rb", "Sr", "Y", "Zr", "Nb",
    "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
    "Am", "B", "Si",
})

# ── Atomic Weights (IUPAC 2021) ────────────────────────────────────────
ATOMIC_WEIGHTS = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Tc": 97.0, "Ru": 101.07, "Rh": 102.906,
    "Pd": 106.42, "Ag": 107.868, "Cd": 112.414, "In": 114.818, "Sn": 118.710,
    "Sb": 121.760, "Te": 127.60, "I": 126.904, "Xe": 131.293, "Cs": 132.905,
    "Ba": 137.327, "La": 138.905, "Ce": 140.116, "Pr": 140.908, "Nd": 144.242,
    "Pm": 145.0, "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.925,
    "Dy": 162.500, "Ho": 164.930, "Er": 167.259, "Tm": 168.934, "Yb": 173.045,
    "Lu": 174.967, "Hf": 178.49, "Ta": 180.948, "W": 183.84, "Re": 186.207,
    "Os": 190.23, "Ir": 192.217, "Pt": 195.084, "Au": 196.967, "Hg": 200.592,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.980, "Po": 209.0, "At": 210.0,
    "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.038,
    "Pa": 231.036, "U": 238.029, "Np": 237.0, "Pu": 244.0, "Am": 243.0,
    "X": 0.0,  # Dummy atom
    "Lr": 266.0,  # Lawrencium placeholder
}

# ── Halogen Set (for DoU calculation) ──────────────────────────────────
HALOGEN_SET = frozenset({"F", "Cl", "Br", "I", "At"})

# ── Typical Coordination Numbers (for open metal site detection) ───────
# Used in Layer 2 rule-based enrichment
TYPICAL_COORDINATION = {
    "Li": 4, "Be": 4, "Na": 6, "Mg": 6, "Al": 6,
    "K": 6, "Ca": 8, "Sc": 6, "Ti": 6, "V": 6,
    "Cr": 6, "Mn": 6, "Fe": 6, "Co": 6, "Ni": 6,
    "Cu": 4, "Zn": 4, "Ga": 4, "Ge": 4,
    "Y": 8, "Zr": 8, "Nb": 6, "Mo": 6,
    "Ru": 6, "Rh": 6, "Pd": 4, "Ag": 2,
    "Cd": 6, "In": 6, "Sn": 6, "Sb": 6,
    "La": 9, "Ce": 9, "Pr": 9, "Nd": 9,
    "Sm": 9, "Eu": 9, "Gd": 9, "Tb": 9,
    "Dy": 9, "Ho": 9, "Er": 9, "Tm": 9,
    "Yb": 9, "Lu": 9, "Hf": 8, "Ta": 6,
    "W": 6, "Re": 6, "Os": 6, "Ir": 6,
    "Pt": 4, "Au": 4, "Hg": 2,
    "Pb": 6, "Bi": 6,
    "Th": 9, "U": 8,
    "B": 4, "Si": 4,
}
