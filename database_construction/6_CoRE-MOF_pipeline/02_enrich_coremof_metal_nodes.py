#!/usr/bin/env python3
"""Enrich CoRE-MOF enriched_v2 JSONs with metal node metadata.

Uses RDKit to analyze node SMILES and extract:
  - nuclearity (metal atom count)
  - geometry (inferred from coordination number)
  - oxidation_states (common values from literature)
  - sbu_type (standardized label)
  - coordinating_groups (from non-metal atoms bonded to metals)
  - ligand_chemistry (donor atom types)

Also applies generic tag propagation to functional_groups.

Usage:
    python enrich_coremof_metal_nodes.py                  # dry run
    python enrich_coremof_metal_nodes.py --apply          # modify JSONs
"""

import json
import os
import sys
import time
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_PATH = Path(__file__).parent / "enrich_coremof_metal_nodes.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metal set and common oxidation states
# ---------------------------------------------------------------------------
METAL_SET = frozenset({
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "Rb", "Sr", "Y", "Zr", "Nb",
    "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "Hg", "Tl", "Pb", "Bi",
})

# Common MOF metal oxidation states (most frequent in MOF literature)
COMMON_OX_STATES = {
    "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1,
    "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2,
    "Sc": 3, "Y": 3, "La": 3, "Ce": 3, "Pr": 3, "Nd": 3, "Sm": 3,
    "Eu": 3, "Gd": 3, "Tb": 3, "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3,
    "Yb": 3, "Lu": 3,
    "Ti": 4, "Zr": 4, "Hf": 4,
    "V": 3, "Nb": 5, "Ta": 5,
    "Cr": 3, "Mo": 6, "W": 6,
    "Mn": 2, "Fe": 3, "Co": 2, "Ni": 2, "Cu": 2, "Zn": 2,
    "Ru": 3, "Rh": 3, "Pd": 2, "Ag": 1, "Cd": 2,
    "In": 3, "Sn": 4, "Sb": 3,
    "Ir": 3, "Pt": 2, "Au": 3, "Hg": 2,
    "Al": 3, "Ga": 3, "Bi": 3, "Pb": 2, "Tl": 1,
}

# Well-known SBU patterns (SMILES -> metadata override)
KNOWN_SBUS = {
    "[Zn][O]([Zn])([Zn])[Zn]": {
        "geometry": "tetrahedral_Zn4O",
        "sbu_type": "Zn4O_octahedral_6c",
        "connectivity": 6,
        "has_open_metal_sites": False,
    },
    "[Cu][Cu]": {
        "geometry": "paddlewheel",
        "sbu_type": "Cu2_paddlewheel_4c",
        "connectivity": 4,
        "has_open_metal_sites": True,
    },
    "[Zn][Zn]": {
        "geometry": "paddlewheel",
        "sbu_type": "Zn2_paddlewheel_4c",
        "connectivity": 4,
        "has_open_metal_sites": True,
    },
    "[Co][Co]": {
        "geometry": "paddlewheel",
        "sbu_type": "Co2_paddlewheel_4c",
        "connectivity": 4,
        "has_open_metal_sites": True,
    },
}

# Generic tag hierarchy (same as PORMAKE/hMOF enrichment)
TAG_HIERARCHY = {
    "benzene_ring": {"aromatic", "aryl", "ring"},
    "naphthalene": {"aromatic", "aryl", "ring"},
    "biphenyl": {"aromatic", "aryl", "ring"},
    "anthracene": {"aromatic", "aryl", "ring"},
    "fluorene": {"aromatic", "aryl", "ring"},
    "terphenyl": {"aromatic", "aryl", "ring"},
    "pyrene": {"aromatic", "aryl", "ring"},
    "pyridine": {"heterocycle", "ring", "nitrogen", "aromatic"},
    "imidazole": {"heterocycle", "ring", "nitrogen", "aromatic", "azole", "azolate"},
    "thiophene": {"heterocycle", "ring", "sulfur", "aromatic"},
    "furan": {"heterocycle", "ring", "oxygen", "aromatic"},
    "pyrazole": {"heterocycle", "ring", "nitrogen", "aromatic", "azole"},
    "triazole": {"heterocycle", "ring", "nitrogen", "aromatic", "azole", "azolate"},
    "triazole_any": {"heterocycle", "ring", "nitrogen", "aromatic", "azole", "azolate"},
    "triazole_1_2_4": {"heterocycle", "ring", "nitrogen", "aromatic", "azole", "azolate"},
    "tetrazole": {"heterocycle", "ring", "nitrogen", "aromatic", "azole", "azolate"},
    "pyrimidine": {"heterocycle", "ring", "nitrogen", "aromatic"},
    "pyrazine": {"heterocycle", "ring", "nitrogen", "aromatic"},
    "triazine": {"heterocycle", "ring", "nitrogen", "aromatic"},
    "oxadiazole": {"heterocycle", "ring", "nitrogen", "oxygen", "aromatic"},
    "thiadiazole": {"heterocycle", "ring", "nitrogen", "sulfur", "aromatic"},
    "thiazole": {"heterocycle", "ring", "nitrogen", "sulfur", "aromatic"},
    "quinoline": {"heterocycle", "ring", "nitrogen", "aromatic"},
    "isoquinoline": {"heterocycle", "ring", "nitrogen", "aromatic"},
    "benzimidazole": {"heterocycle", "ring", "nitrogen", "aromatic", "azole"},
    "piperazine": {"heterocycle", "ring", "nitrogen"},
    "cyclohexane": {"ring", "aliphatic_ring"},
    "cyclopentane": {"ring", "aliphatic_ring"},
    "amine_any": {"nitrogen"}, "primary_amine": {"nitrogen", "amine"},
    "secondary_amine": {"nitrogen", "amine"}, "tertiary_amine": {"nitrogen", "amine"},
    "amide": {"nitrogen", "oxygen", "carbonyl"},
    "imine": {"nitrogen"}, "imine_any": {"nitrogen"},
    "nitro": {"nitrogen", "oxygen"}, "nitrile": {"nitrogen"}, "nitrile_sub": {"nitrogen"},
    "azo": {"nitrogen"},
    "carboxyl_any": {"oxygen", "carbonyl"}, "carboxylate": {"oxygen", "carbonyl"},
    "carboxylic_acid": {"oxygen", "carbonyl"}, "hydroxyl": {"oxygen"},
    "phenol": {"oxygen", "aromatic"}, "ether": {"oxygen"}, "methoxy": {"oxygen"},
    "ester": {"oxygen", "carbonyl"}, "aldehyde": {"oxygen", "carbonyl"},
    "ketone": {"oxygen", "carbonyl"},
    "phosphonate": {"oxygen", "phosphorus"}, "sulfonate": {"oxygen", "sulfur"},
    "thiol": {"sulfur"}, "thioether": {"sulfur"},
    "fluorine": {"halogen"}, "chlorine": {"halogen"}, "bromine": {"halogen"},
    "iodine": {"halogen"}, "trifluoromethyl": {"halogen"},
    "vinyl": {"carbon_framework"}, "acetylene": {"carbon_framework"},
    "butadiyne": {"carbon_framework"}, "methyl": set(),
}

# Donor atom element -> coordinating group name
DONOR_MAP = {
    "O": "Oxygen", "N": "Nitrogen", "S": "Sulfur", "P": "Phosphorus",
    "F": "Fluorine", "Cl": "Chlorine",
}


def analyze_node_smiles(smi: str) -> dict:
    """Analyze a node SMILES string using RDKit to extract metal node metadata."""
    try:
        from rdkit import Chem
    except ImportError:
        return {}

    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return {}

    metals = []
    non_metal_neighbors = set()

    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in METAL_SET:
            metals.append(sym)
            # Check what non-metals are bonded to this metal
            for neighbor in atom.GetNeighbors():
                nsym = neighbor.GetSymbol()
                if nsym not in METAL_SET:
                    non_metal_neighbors.add(nsym)

    if not metals:
        return {}

    nuclearity = len(metals)
    metal_set = sorted(set(metals))

    # Oxidation states from common values
    oxidation_states = {}
    for m in metal_set:
        if m in COMMON_OX_STATES:
            oxidation_states[m] = COMMON_OX_STATES[m]

    # Coordinating groups from donor atoms
    coordinating_groups = []
    ligand_chemistry = []
    for elem in sorted(non_metal_neighbors):
        if elem in DONOR_MAP:
            ligand_chemistry.append(DONOR_MAP[elem])
    # Infer coordinating group type
    if "O" in non_metal_neighbors:
        coordinating_groups.append("Carboxylate")
    if "N" in non_metal_neighbors:
        coordinating_groups.append("Nitrogen")
    if "S" in non_metal_neighbors:
        coordinating_groups.append("Sulfur")

    # Check for known SBU override
    known = KNOWN_SBUS.get(smi)
    if known:
        return {
            "nuclearity": nuclearity,
            "connectivity": known.get("connectivity"),
            "geometry": known["geometry"],
            "sbu_type": known["sbu_type"],
            "oxidation_states": oxidation_states,
            "has_open_metal_sites": known.get("has_open_metal_sites"),
            "coordinating_groups": coordinating_groups or ["Carboxylate"],
            "ligand_chemistry": ligand_chemistry or ["Oxygen"],
        }

    # Infer geometry from nuclearity
    if nuclearity == 1:
        geometry = None  # can't determine without coordination number
        sbu_type = f"{metal_set[0]}1_mononuclear"
    elif nuclearity == 2:
        geometry = "dinuclear"
        sbu_type = f"{''.join(metal_set)}{nuclearity}_dinuclear"
    elif nuclearity == 3:
        geometry = "trinuclear"
        sbu_type = f"{''.join(metal_set)}{nuclearity}_trinuclear"
    elif nuclearity == 4:
        geometry = "tetranuclear"
        sbu_type = f"{''.join(metal_set)}{nuclearity}_tetranuclear"
    elif nuclearity == 6 and "Zr" in metal_set:
        geometry = "octahedral_Zr6O8"
        sbu_type = "Zr6O8_UiO_12c"
    elif nuclearity >= 6:
        geometry = f"polynuclear_{nuclearity}"
        sbu_type = f"{''.join(metal_set)}{nuclearity}_polynuclear"
    else:
        geometry = f"cluster_{nuclearity}"
        sbu_type = f"{''.join(metal_set)}{nuclearity}_cluster"

    return {
        "nuclearity": nuclearity,
        "connectivity": None,
        "geometry": geometry,
        "sbu_type": sbu_type,
        "oxidation_states": oxidation_states,
        "has_open_metal_sites": None,
        "coordinating_groups": coordinating_groups,
        "ligand_chemistry": ligand_chemistry,
    }


def enrich_tags(fgs: list) -> list:
    """Add generic parent tags from TAG_HIERARCHY."""
    enriched = set(fgs)
    for tag in fgs:
        generic = TAG_HIERARCHY.get(tag.lower().strip(), set())
        enriched.update(generic)
    return sorted(enriched)


def enrich_record(data: dict) -> tuple[bool, str]:
    """Enrich a single CoRE-MOF record in place."""
    l1 = data.get("layer1_facts", {})
    l2 = data.get("layer2_semantics", {})
    mn = l1.get("metal_node", {})
    fg = l2.get("functional_groups", {})
    nodes = l1.get("smiles_nodes", [])
    changed = False

    # --- Metal node enrichment ---
    if nodes and mn.get("nuclearity") is None:
        node_smi = nodes[0]
        analysis = analyze_node_smiles(node_smi)
        if analysis:
            for field, value in analysis.items():
                current = mn.get(field)
                if current is None or current == [] or current == {}:
                    mn[field] = value
                    changed = True

    # --- Generic tag propagation ---
    rule_based = fg.get("rule_based", [])
    enriched_rb = enrich_tags(rule_based)
    if len(enriched_rb) > len(rule_based):
        fg["rule_based"] = enriched_rb
        changed = True

    backbone = fg.get("backbone", [])
    enriched_bb = enrich_tags(backbone)
    if len(enriched_bb) > len(backbone):
        fg["backbone"] = enriched_bb
        changed = True

    return changed, "enriched"


def main():
    parser = argparse.ArgumentParser(description="Enrich CoRE-MOF metal_node + generic tags")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry run)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--input-dir", type=str, default=None)
    args = parser.parse_args()

    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = Path(__file__).parent.parent / "CoRE-MOF" / "coremof_enriched_v2"

    log.info("=" * 60)
    log.info("CoRE-MOF Metal Node + Generic Tag Enrichment")
    log.info("=" * 60)
    log.info(f"  Input:     {input_dir}")
    log.info(f"  Mode:      {'APPLY' if args.apply else 'DRY RUN'}")
    log.info(f"  Limit:     {args.limit or 'all'}")
    log.info(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    log.info("")

    files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix == ".json" and not f.name.startswith("_")
    ])
    if args.limit:
        files = files[:args.limit]

    total = len(files)
    enriched = 0
    skipped = 0
    no_nodes = 0
    errors = 0
    t0 = time.time()

    for i, fpath in enumerate(files):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            ch, msg = enrich_record(data)
            if ch:
                enriched += 1
                if args.apply:
                    with open(fpath, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                skipped += 1

        except Exception as e:
            errors += 1
            log.error(f"  ERROR [{fpath.name}]: {e}")

        if (i + 1) % 200 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            log.info(f"  [{i+1:>5}/{total}] {elapsed:.1f}s | enriched={enriched} skipped={skipped} err={errors}")

    elapsed = time.time() - t0
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"  Total:    {total}")
    log.info(f"  Enriched: {enriched}")
    log.info(f"  Skipped:  {skipped}")
    log.info(f"  Errors:   {errors}")
    log.info(f"  Time:     {elapsed:.1f}s")

    if not args.apply:
        log.info("\n  DRY RUN -- no files modified. Use --apply to write.")


if __name__ == "__main__":
    main()
