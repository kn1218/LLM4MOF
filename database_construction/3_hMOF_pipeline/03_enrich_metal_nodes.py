#!/usr/bin/env python3
"""Enrich hMOF enriched_v2 JSONs with metal node metadata.

The hMOF pipeline originally left metal_node fields (nuclearity, geometry,
sbu_type, oxidation_states, coordinating_groups, ligand_chemistry) as None.
This script fills them using a deterministic lookup table derived from
RDKit analysis of the 19 unique node SMILES patterns found in hMOF.

Usage:
    python enrich_hmof_metal_nodes.py                    # dry run (report only)
    python enrich_hmof_metal_nodes.py --apply             # modify JSONs in place
    python enrich_hmof_metal_nodes.py --apply --limit 100 # process first 100 only

Log output written to: enrich_hmof_metal_nodes.log
"""

import json
import os
import sys
import time
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_PATH = Path(__file__).parent / "enrich_hmof_metal_nodes.log"
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
# Deterministic lookup table: node_smiles -> metal_node metadata
#
# Derived from RDKit analysis of all 19 unique node SMILES in hMOF.
# Oxidation states based on common MOF chemistry literature:
#   Zn4O cluster: Zn(II), Zr6 cluster: Zr(IV), Cu paddlewheel: Cu(II),
#   V-oxo: V(III/IV) mixed, mononuclear: standard oxidation states.
# ---------------------------------------------------------------------------
NODE_LOOKUP = {
    # --- Zn nodes ---
    "[Zn][O]([Zn])([Zn])[Zn]": {
        "nuclearity": 4,
        "connectivity": 6,
        "geometry": "tetrahedral_Zn4O",
        "sbu_type": "Zn4O_octahedral_6c",
        "oxidation_states": {"Zn": 2},
        "has_open_metal_sites": False,
        "coordinating_groups": ["Carboxylate"],
        "ligand_chemistry": ["Oxygen"],
    },
    "[Zn][Zn]": {
        "nuclearity": 2,
        "connectivity": 4,
        "geometry": "paddlewheel",
        "sbu_type": "Zn2_paddlewheel_4c",
        "oxidation_states": {"Zn": 2},
        "has_open_metal_sites": True,
        "coordinating_groups": ["Carboxylate"],
        "ligand_chemistry": ["Oxygen"],
    },
    "[Zn]": {
        "nuclearity": 1,
        "connectivity": None,
        "geometry": None,
        "sbu_type": "Zn1_mononuclear",
        "oxidation_states": {"Zn": 2},
        "has_open_metal_sites": None,
        "coordinating_groups": [],
        "ligand_chemistry": [],
    },
    # --- Cu nodes ---
    "[Cu][Cu]": {
        "nuclearity": 2,
        "connectivity": 4,
        "geometry": "paddlewheel",
        "sbu_type": "Cu2_paddlewheel_4c",
        "oxidation_states": {"Cu": 2},
        "has_open_metal_sites": True,
        "coordinating_groups": ["Carboxylate"],
        "ligand_chemistry": ["Oxygen"],
    },
    "[Cu]": {
        "nuclearity": 1,
        "connectivity": None,
        "geometry": None,
        "sbu_type": "Cu1_mononuclear",
        "oxidation_states": {"Cu": 2},
        "has_open_metal_sites": None,
        "coordinating_groups": [],
        "ligand_chemistry": [],
    },
    # --- Zr nodes ---
    "[O]12[Zr]34[O]5[Zr]62[O]2[Zr]71[O]4[Zr]14[O]3[Zr]35[O]6[Zr]2([O]71)[O]43": {
        "nuclearity": 6,
        "connectivity": 12,
        "geometry": "octahedral_Zr6O8",
        "sbu_type": "Zr6O8_UiO_12c",
        "oxidation_states": {"Zr": 4},
        "has_open_metal_sites": False,
        "coordinating_groups": ["Carboxylate", "Oxygen"],
        "ligand_chemistry": ["Oxygen"],
    },
    "[Zr]12[O]3[Zr]4[O]2[Zr]2[O]4[Zr]4[O]5[Zr]3[O]1[Zr]5[O]24": {
        "nuclearity": 6,
        "connectivity": 12,
        "geometry": "octahedral_Zr6O8",
        "sbu_type": "Zr6O8_UiO_12c",
        "oxidation_states": {"Zr": 4},
        "has_open_metal_sites": False,
        "coordinating_groups": ["Carboxylate", "Oxygen"],
        "ligand_chemistry": ["Oxygen"],
    },
    "[Zr]12[O]3[Zr]4[O]5[Zr]6[O]1[Zr]17[O]2[Zr]23[O]4[Zr]5([O]12)[O]67": {
        "nuclearity": 6,
        "connectivity": 12,
        "geometry": "octahedral_Zr6O8",
        "sbu_type": "Zr6O8_UiO_12c",
        "oxidation_states": {"Zr": 4},
        "has_open_metal_sites": False,
        "coordinating_groups": ["Carboxylate", "Oxygen"],
        "ligand_chemistry": ["Oxygen"],
    },
    "[Zr]": {
        "nuclearity": 1,
        "connectivity": None,
        "geometry": None,
        "sbu_type": "Zr1_mononuclear",
        "oxidation_states": {"Zr": 4},
        "has_open_metal_sites": None,
        "coordinating_groups": [],
        "ligand_chemistry": [],
    },
    "[Zr]1[O][Zr][O]1": {
        "nuclearity": 2,
        "connectivity": None,
        "geometry": "dinuclear_oxo",
        "sbu_type": "Zr2O2_dinuclear",
        "oxidation_states": {"Zr": 4},
        "has_open_metal_sites": None,
        "coordinating_groups": ["Oxygen"],
        "ligand_chemistry": ["Oxygen"],
    },
    "[Zr][O][Zr]": {
        "nuclearity": 2,
        "connectivity": None,
        "geometry": "dinuclear_oxo",
        "sbu_type": "Zr2O_dinuclear",
        "oxidation_states": {"Zr": 4},
        "has_open_metal_sites": None,
        "coordinating_groups": ["Oxygen"],
        "ligand_chemistry": ["Oxygen"],
    },
    # --- V nodes ---
    "[V]": {
        "nuclearity": 1,
        "connectivity": None,
        "geometry": None,
        "sbu_type": "V1_mononuclear",
        "oxidation_states": {"V": 3},
        "has_open_metal_sites": None,
        "coordinating_groups": [],
        "ligand_chemistry": [],
    },
    "[V]O[V]": {
        "nuclearity": 2,
        "connectivity": None,
        "geometry": "dinuclear_oxo",
        "sbu_type": "V2O_dinuclear",
        "oxidation_states": {"V": 4},
        "has_open_metal_sites": None,
        "coordinating_groups": ["Oxygen"],
        "ligand_chemistry": ["Oxygen"],
    },
    "[O][V]O[V]": {
        "nuclearity": 2,
        "connectivity": None,
        "geometry": "dinuclear_oxo",
        "sbu_type": "V2O2_dinuclear",
        "oxidation_states": {"V": 4},
        "has_open_metal_sites": None,
        "coordinating_groups": ["Oxygen"],
        "ligand_chemistry": ["Oxygen"],
    },
    "[V]1[V]O1": {
        "nuclearity": 2,
        "connectivity": None,
        "geometry": "dinuclear_oxo",
        "sbu_type": "V2O_dinuclear",
        "oxidation_states": {"V": 4},
        "has_open_metal_sites": None,
        "coordinating_groups": ["Oxygen"],
        "ligand_chemistry": ["Oxygen"],
    },
    "[V]O[V]O[V]": {
        "nuclearity": 3,
        "connectivity": None,
        "geometry": "trinuclear_oxo",
        "sbu_type": "V3O2_trinuclear",
        "oxidation_states": {"V": 3},
        "has_open_metal_sites": None,
        "coordinating_groups": ["Oxygen"],
        "ligand_chemistry": ["Oxygen"],
    },
    "[O]1[V]2[O][V]([O]2)O1": {
        "nuclearity": 2,
        "connectivity": None,
        "geometry": "dinuclear_oxo",
        "sbu_type": "V2O4_dinuclear",
        "oxidation_states": {"V": 4},
        "has_open_metal_sites": False,
        "coordinating_groups": ["Oxygen"],
        "ligand_chemistry": ["Oxygen"],
    },
    "[O]1[V][O][V]1": {
        "nuclearity": 2,
        "connectivity": None,
        "geometry": "dinuclear_oxo",
        "sbu_type": "V2O2_dinuclear",
        "oxidation_states": {"V": 4},
        "has_open_metal_sites": None,
        "coordinating_groups": ["Oxygen"],
        "ligand_chemistry": ["Oxygen"],
    },
    "[V]1[O][V][O]1": {
        "nuclearity": 2,
        "connectivity": None,
        "geometry": "dinuclear_oxo",
        "sbu_type": "V2O2_dinuclear",
        "oxidation_states": {"V": 4},
        "has_open_metal_sites": None,
        "coordinating_groups": ["Oxygen"],
        "ligand_chemistry": ["Oxygen"],
    },
}


def enrich_record(data: dict) -> tuple[bool, str]:
    """Enrich a single hMOF record's metal_node in place.

    Returns (changed: bool, message: str).
    """
    l1 = data.get("layer1_facts", {})
    mn = l1.get("metal_node", {})
    nodes = l1.get("smiles_nodes", [])

    if not nodes:
        return False, "no smiles_nodes"

    # Use first node SMILES for lookup
    node_smi = nodes[0]
    lookup = NODE_LOOKUP.get(node_smi)

    if lookup is None:
        return False, f"unknown node SMILES: {node_smi}"

    # Only update fields that are currently None/empty
    changed = False
    for field, value in lookup.items():
        current = mn.get(field)
        if current is None or current == [] or current == {}:
            mn[field] = value
            changed = True

    return changed, f"enriched from lookup ({node_smi[:30]}...)"


def main():
    parser = argparse.ArgumentParser(description="Enrich hMOF metal_node fields")
    parser.add_argument("--apply", action="store_true", help="Write changes to files (default: dry run)")
    parser.add_argument("--limit", type=int, default=None, help="Max records to process")
    parser.add_argument("--input-dir", type=str, default=None, help="Input directory (default: hMOF/hmof_enriched_v2)")
    args = parser.parse_args()

    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = Path(__file__).parent.parent / "hMOF" / "hmof_enriched_v2"

    log.info("=" * 60)
    log.info("hMOF Metal Node Enrichment")
    log.info("=" * 60)
    log.info(f"  Input:     {input_dir}")
    log.info(f"  Mode:      {'APPLY (write files)' if args.apply else 'DRY RUN (report only)'}")
    log.info(f"  Limit:     {args.limit or 'all'}")
    log.info(f"  Lookup:    {len(NODE_LOOKUP)} unique node patterns")
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
    unknown = 0
    errors = 0
    unknown_smiles = set()

    t0 = time.time()

    for i, fpath in enumerate(files):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            changed, msg = enrich_record(data)

            if changed:
                enriched += 1
                if args.apply:
                    with open(fpath, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
            elif "unknown" in msg:
                unknown += 1
                smi = msg.split(": ", 1)[1] if ": " in msg else msg
                unknown_smiles.add(smi)
            else:
                skipped += 1

        except Exception as e:
            errors += 1
            log.error(f"  ERROR [{fpath.name}]: {e}")

        if (i + 1) % 5000 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            log.info(
                f"  [{i+1:>6}/{total}] "
                f"{elapsed:.1f}s, {rate:.0f} rec/s | "
                f"enriched={enriched} skipped={skipped} unknown={unknown} err={errors}"
            )

    elapsed = time.time() - t0
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"  Total processed:  {total}")
    log.info(f"  Enriched:         {enriched}")
    log.info(f"  Skipped (no change): {skipped}")
    log.info(f"  Unknown SMILES:   {unknown}")
    log.info(f"  Errors:           {errors}")
    log.info(f"  Time:             {elapsed:.1f}s ({total/elapsed:.0f} rec/s)" if elapsed > 0 else "")

    if unknown_smiles:
        log.warning(f"\n  Unknown node SMILES ({len(unknown_smiles)}):")
        for s in sorted(unknown_smiles):
            log.warning(f"    {s}")

    if not args.apply:
        log.info("\n  DRY RUN -- no files were modified. Use --apply to write changes.")


if __name__ == "__main__":
    main()
