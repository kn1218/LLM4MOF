"""
qmof_vocabulary_parser.py  –  Dynamically builds a degenerate (multiple-inheritance) mapping
from the QMOF-compatible Edge Vocabulary (V5).

It reads `0edge_hierarchy_v5.md` to map each specific functional group back to all
its topological/compositional ancestor arrays.

It also exports a SMARTS_TO_VOCABULARY bridge dictionary to translate lowercase RDKit keys 
into proper title-case Vocabulary keys.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MD_PATH = BASE_DIR / "Raw data" / "pormake data rework" / "0edge_hierarchy_v5.md"

def build_vocabulary_map(md_file: Path) -> dict:
    vocabulary_map = {}
    
    with open(md_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    stack = []  # Stores tuples of (depth, node_name)
    
    for line in lines:
        if not line.strip() or line.startswith("#") or line.startswith("This file") or line.startswith("Groups may"):
            continue
            
        leading_spaces = len(line) - len(line.lstrip())
        depth = leading_spaces // 2  # Assuming 2 spaces per indentation level
        
        # Clean markdown characters
        node_name = line.strip().lstrip("- ").replace("**", "").strip()
        
        if not node_name:
            continue
            
        # Pop off nodes that are at the same or deeper indentation level
        while stack and stack[-1][0] >= depth:
            stack.pop()
            
        stack.append((depth, node_name))
        
        # Collect all ancestors (including self)
        ancestors = {name for _, name in stack}
        
        # Degeneracy Merge: Union with existing ancestors if node already encountered
        if node_name in vocabulary_map:
            vocabulary_map[node_name].update(ancestors)
        else:
            vocabulary_map[node_name] = ancestors
            
    # Convert sets to sorted lists for clean JSON export later
    for k, v in vocabulary_map.items():
        vocabulary_map[k] = sorted(list(v))
        
    return vocabulary_map


VOCABULARY_MAP = build_vocabulary_map(MD_PATH)


# ============================================================================
# BRIDGE DICTIONARY: Translates lowercase SMARTS keys to V5 Vocabulary keys
# ============================================================================
SMARTS_TO_VOCABULARY = {
    # --- Carbon Rings ---
    "benzene": "Benzene",
    "biphenyl": "Biphenyl",
    "terphenyl": "Terphenyl",
    "naphthalene": "Naphthalene",
    "anthracene": "Anthracene",
    "phenanthrene": "Phenanthrene",
    "pyrene": "Pyrene",
    "fluorene": "Fluorene",
    "indane": "Indane",
    "cyclohexane": "Cyclohexane",
    "cyclopentane": "Cyclopentane",
    
    # --- Cages ---
    "cubane": "Cubane",
    "adamantane": "Adamantane",
    "barrelene": "Barrelene",     # Fallback if present
    "triptycene": "Triptycene",   # Fallback if present
    
    # --- Heterocycles ---
    "pyridine": "Pyridine_Ring",
    "pyrimidine": "Pyrimidine_Ring",
    "pyrazine": "Pyrazine_Ring",
    "pyridazine": "Pyridazine_Ring",
    "triazine": "Triazine_Ring",
    "tetrazine": "Tetrazine_Ring",
    "imidazole": "Imidazole_Ring",
    "pyrazole": "Pyrazole_Ring",
    "triazole_123": "Triazole_Ring",
    "triazole_124": "Triazole_Ring",
    "tetrazole": "Tetrazole_Ring",
    "thiophene": "Thiophene_Ring",
    "furan": "Furan_Ring",
    "thiazole": "Thiazole_Ring",
    "oxazole": "Oxazole_Ring",
    "thiadiazole": "Thiadiazole_Ring",
    "oxadiazole": "Oxadiazole_Ring",
    "isoxazole": "Isoxazole_Ring",
    "isothiazole": "Isothiazole_Ring",
    "benzimidazole": "Benzimidazole", # Defaults
    "benzothiazole": "Benzothiazole",
    "benzoxazole": "Benzoxazole",
    "quinoline": "Quinoline",
    "isoquinoline": "Isoquinoline",
    "indole": "Indole",
    "benzofuran": "Benzofuran",
    "benzothiophene": "Benzothiophene",
    "benzodioxole": "Benzodioxole",
    "piperazine": "Piperazine_Ring",
    "piperidine": "Piperidine_Ring",
    "morpholine": "Morpholine",
    "porphyrin_core": "Porphyrin_Ring",
    "imide_ring": "Diimide_Ring",
    "BODIPY": "BODIPY_Core",
    
    # --- Halogens ---
    "fluorine": "Fluoro",
    "chlorine": "Chloro",
    "bromine": "Bromo",
    "iodine": "Iodo",
    "trifluoromethyl": "Trifluoromethyl",
    "perfluoro": "Perfluoro", 
    
    # --- Nitrogen Groups ---
    "amine": "Primary_Amine",          # Base approximation
    "secondary_amine": "Secondary_Amine",
    "quaternary_N": "Quaternary_N",
    "pyridinium": "Pyridinium",
    "amide": "Amide",
    "nitro": "Nitro",
    "azo": "Azo",
    "azine": "Azine",
    "hydrazone": "Hydrazone",
    "imine": "Imine",
    "sulfonamide": "Sulfonamide",

    # --- Oxygen Groups ---
    "hydroxyl": "Hydroxyl",
    "methoxy": "Methoxy",
    "ether": "Ether",
    "carboxylic_acid": "Carboxylate",
    "ketone": "Ketone",
    "aldehyde": "Aldehyde",
    "ether_link": "Ether",

    # --- Sulfur Groups ---
    "thiol": "Thiol",
    "thioether": "Thioether",
    "thioether_link": "Thioether",
    "sulfonic_acid": "Sulfonic_Acid",
    "sulfonyl": "Sulfonyl",
    
    # --- Alkyl / Acyclic ---
    "methyl": "Methyl",
    "ethyl": "Ethyl",
    "isopropyl": "Isopropyl",
    "tert_butyl": "tert-Butyl",
    "methylene_bridge": "Alkane_Chain",
    "alkane_chain": "Alkane_Chain",
    
    # --- Alkene / Alkyne ---
    "alkyne": "Alkyne",
    "butadiyne": "Butadiyne",
    "hexatriyne": "Hexatriyne",
    "alkene": "Alkene",
    "butadiene": "Butadiene",
    
    # Metal falls back to Transition_Metal
    "metallolinker": "Transition_Metal"
}
