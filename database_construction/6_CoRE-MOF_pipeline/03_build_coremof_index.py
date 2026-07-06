#!/usr/bin/env python3
"""
build_coremof_index.py — Build flat CoRE-MOF index from enriched v2 JSONs.

Produces coremof_index_v1.json matching the flat format used by QMOF and hMOF,
plus CoRE-MOF-specific fields (stability, catenation, doi, refcode).

Usage:
    python build_coremof_index.py
    python build_coremof_index.py --dry-run
"""

import json
import os
import re
import sys
import time
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

LOG_PATH = Path(__file__).parent / "build_coremof_index.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

TOPOLOGY_INVALID = {"ERROR", "NA.NA", "NA.NAno_mof", "unstable"}


def safe_get(d, *keys, default=None):
    current = d
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k, default)
    return current


def extract_record(data):
    l1 = data.get("layer1_facts") or {}
    l2 = data.get("layer2_semantics") or {}
    mn = l1.get("metal_node") or {}
    fg = l2.get("functional_groups") or {}
    vr = data.get("validation_report") or {}
    stab = l1.get("stability") or {}

    topo = l1.get("topology")
    if topo and topo in TOPOLOGY_INVALID:
        topo = None

    record = {
        # ID
        "coremof_id": data.get("coremof_id"),
        # Common fields (same as QMOF/hMOF)
        "metals": mn.get("metals") or [],
        "functional_groups": fg.get("rule_based") or [],
        "topology": topo,
        "surface_area_m2g": l1.get("surface_area_m2g"),
        "void_fraction": l1.get("void_fraction"),
        "density": l1.get("density"),
        "pld": l1.get("pld"),
        "lcd": l1.get("lcd"),
        "synthesized": l1.get("synthesized"),
        "is_valid": vr.get("status") != "fail",
        "readable_name": l2.get("readable_name"),
        "has_open_metal_sites": mn.get("has_open_metal_sites"),
        # Metal node enrichment fields
        "nuclearity": mn.get("nuclearity"),
        "geometry": mn.get("geometry"),
        "sbu_type": mn.get("sbu_type"),
        "oxidation_states": mn.get("oxidation_states"),
        "coordinating_groups": mn.get("coordinating_groups") or [],
        "ligand_chemistry": mn.get("ligand_chemistry") or [],
        # CoRE-MOF specific
        "thermal_stability_C": stab.get("thermal_stability_C"),
        "water_stability": stab.get("water_stability"),
        "solvent_stability": stab.get("solvent_stability"),
        "catenation": l1.get("catenation"),
        "doi": l1.get("doi"),
        "refcode": l1.get("refcode"),
        "natoms": l1.get("natoms"),
        "spacegroup_number": l1.get("spacegroup_number"),
        "crystal_system": l1.get("crystal_system"),
        # Functional groups categorized
        "functional_groups_categorized": {
            "backbone": fg.get("backbone") or [],
            "substituents": fg.get("substituents") or [],
            "rule_based": fg.get("rule_based") or [],
            "rule_based_counts": fg.get("rule_based_counts") or {},
        },
    }
    return record


def main():
    parser = argparse.ArgumentParser(description="Build CoRE-MOF flat index")
    parser.add_argument("--source-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    source_dir = Path(args.source_dir) if args.source_dir else base / "CoRE-MOF" / "coremof_enriched_v2"
    output_dir = Path(args.output_dir) if args.output_dir else base / "CoRE-MOF"
    output_index = output_dir / "coremof_index_v1.json"
    output_sample = output_dir / "coremof_index_v1_pretty_sample.json"

    log.info("=" * 60)
    log.info("CoRE-MOF Index Builder")
    log.info("=" * 60)
    log.info(f"  Source:    {source_dir}")
    log.info(f"  Output:    {output_index}")
    log.info(f"  Mode:      {'DRY RUN' if args.dry_run else 'WRITE'}")
    log.info(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    log.info("")

    files = sorted([
        f for f in source_dir.iterdir()
        if f.suffix == ".json" and not f.name.startswith("_")
    ])
    total = len(files)
    log.info(f"Found {total} CoRE-MOF JSON files")

    records = []
    stats = Counter()
    errors = []
    t0 = time.time()

    for i, fpath in enumerate(files):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            rec = extract_record(data)
            records.append(rec)
            stats["total"] += 1
            if rec.get("nuclearity"): stats["has_nuclearity"] += 1
            if rec.get("geometry"): stats["has_geometry"] += 1
            if rec.get("thermal_stability_C"): stats["has_stability"] += 1
            if rec.get("topology"): stats["has_topology"] += 1
        except Exception as e:
            errors.append(f"{fpath.name}: {e}")

    # Sort by coremof_id
    records.sort(key=lambda r: r.get("coremof_id", ""))

    elapsed = time.time() - t0
    log.info(f"  Records:        {len(records)}")
    log.info(f"  Has nuclearity: {stats['has_nuclearity']} ({stats['has_nuclearity']/total*100:.1f}%)")
    log.info(f"  Has geometry:   {stats['has_geometry']} ({stats['has_geometry']/total*100:.1f}%)")
    log.info(f"  Has stability:  {stats['has_stability']} ({stats['has_stability']/total*100:.1f}%)")
    log.info(f"  Has topology:   {stats['has_topology']} ({stats['has_topology']/total*100:.1f}%)")
    log.info(f"  Errors:         {len(errors)}")
    log.info(f"  Time:           {elapsed:.1f}s")

    if args.dry_run:
        log.info("\n  DRY RUN -- no files written.")
        return

    log.info(f"\nWriting {output_index}...")
    with open(output_index, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, separators=(",", ":"))
    size_mb = output_index.stat().st_size / (1024 * 1024)
    log.info(f"  Written: {size_mb:.1f} MB, {len(records)} records")

    log.info(f"Writing {output_sample}...")
    with open(output_sample, "w", encoding="utf-8") as f:
        json.dump(records[:10], f, indent=2, ensure_ascii=False)

    log.info("\n" + "=" * 60)
    log.info("DONE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
