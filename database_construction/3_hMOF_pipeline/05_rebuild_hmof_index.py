#!/usr/bin/env python3
"""
rebuild_hmof_index.py — Build hMOF index from enriched v2 JSONs with full metal_node fields.

Extended version of build_hmof_index.py that includes metal_node enrichment fields
(geometry, nuclearity, sbu_type, oxidation_states, coordinating_groups, ligand_chemistry)
added by enrich_hmof_metal_nodes.py (2026-05-15).

Outputs:
  1. hmof_index.json           (compact, all records)
  2. hmof_index_pretty_sample.json (first 10, indented)

Archives the old index before overwriting.

Usage:
    python rebuild_hmof_index.py                          # build in database_construction
    python rebuild_hmof_index.py --output-dir /path/to    # custom output
    python rebuild_hmof_index.py --dry-run                # report only, don't write
"""

import json
import os
import re
import shutil
import sys
import time
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_PATH = Path(__file__).parent / "rebuild_hmof_index.log"
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
# Constants
# ---------------------------------------------------------------------------
GAS_KEYS = [
    "h2_uptake_2bar_77K",
    "h2_uptake_100bar_77K",
    "ch4_uptake_35bar_298K",
    "co2_uptake_2_5bar_298K",
    "xe_loading_1bar_273K",
    "kr_loading_1bar_273K",
    "xekr_selectivity_1bar",
]

TOPOLOGY_INVALID = {"ERROR", "NA.NA", "NA.NAno_mof"}
HMOF_NUM_RE = re.compile(r"^hMOF-(\d+)\.json$")


def safe_get(d, *keys, default=None):
    current = d
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k, default)
    return current


def clean_topology(val):
    if val is None or (isinstance(val, str) and val in TOPOLOGY_INVALID):
        return None
    return val


def extract_record(data):
    """Extract compact index record with full metal_node fields."""
    l1 = data.get("layer1_facts") or {}
    l2 = data.get("layer2_semantics") or {}
    gas = l1.get("gas_adsorption") or {}
    metal_node = l1.get("metal_node") or {}
    fg = l2.get("functional_groups") or {}
    vr = data.get("validation_report") or {}

    record = {
        "hmof_id": data.get("hmof_id"),
        "metals": metal_node.get("metals") or [],
        "functional_groups": fg.get("rule_based") or [],
        "topology": clean_topology(l1.get("topology")),
        "surface_area_m2g": l1.get("surface_area_m2g"),
        "void_fraction": l1.get("void_fraction"),
        "density": l1.get("density"),
        "pld": l1.get("pld"),
        "lcd": l1.get("lcd"),
    }

    # Gas adsorption
    for gk in GAS_KEYS:
        record[gk] = gas.get(gk)

    record["synthesized"] = l1.get("synthesized")
    record["is_valid"] = vr.get("status") != "fail"
    record["readable_name"] = l2.get("readable_name")
    record["has_open_metal_sites"] = metal_node.get("has_open_metal_sites")

    # --- NEW: metal_node enrichment fields ---
    record["nuclearity"] = metal_node.get("nuclearity")
    record["geometry"] = metal_node.get("geometry")
    record["sbu_type"] = metal_node.get("sbu_type")
    record["oxidation_states"] = metal_node.get("oxidation_states")
    record["coordinating_groups"] = metal_node.get("coordinating_groups") or []
    record["ligand_chemistry"] = metal_node.get("ligand_chemistry") or []

    # Functional groups categorized
    record["functional_groups_categorized"] = {
        "backbone": fg.get("backbone") or [],
        "substituents": fg.get("substituents") or [],
        "rule_based": fg.get("rule_based") or [],
        "rule_based_counts": fg.get("rule_based_counts") or {},
    }

    return record


def main():
    parser = argparse.ArgumentParser(description="Rebuild hMOF index with metal_node fields")
    parser.add_argument("--source-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    source_dir = Path(args.source_dir) if args.source_dir else base / "hMOF" / "hmof_enriched_v2"
    output_dir = Path(args.output_dir) if args.output_dir else base / "hMOF"
    output_index = output_dir / "hmof_index_v3.json"
    output_sample = output_dir / "hmof_index_v3_pretty_sample.json"

    log.info("=" * 60)
    log.info("hMOF Index Rebuild (with metal_node enrichment)")
    log.info("=" * 60)
    log.info(f"  Source:    {source_dir}")
    log.info(f"  Output:    {output_index}")
    log.info(f"  Mode:      {'DRY RUN' if args.dry_run else 'WRITE'}")
    log.info(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    log.info("")

    # Scan files
    t0 = time.perf_counter()
    json_files = []
    for entry in os.scandir(source_dir):
        if not entry.is_file():
            continue
        m = HMOF_NUM_RE.match(entry.name)
        if m:
            json_files.append((int(m.group(1)), entry.path))

    json_files.sort(key=lambda x: x[0])
    total = len(json_files)
    log.info(f"Found {total:,} hMOF JSON files ({time.perf_counter() - t0:.2f}s)")

    # Process
    records = []
    stats = Counter()
    errors = []

    for i, (num, fpath) in enumerate(json_files):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            rec = extract_record(data)
            records.append(rec)

            # Track stats
            if rec.get("nuclearity") is not None:
                stats["has_nuclearity"] += 1
            if rec.get("geometry") is not None:
                stats["has_geometry"] += 1
            if rec.get("oxidation_states") is not None:
                stats["has_oxidation_states"] += 1
            stats["total"] += 1

        except Exception as e:
            errors.append(f"{os.path.basename(fpath)}: {e}")

        if (i + 1) % 10000 == 0 or (i + 1) == total:
            elapsed = time.perf_counter() - t0
            log.info(f"  [{i+1:>6}/{total}] {elapsed:.1f}s, {(i+1)/elapsed:.0f} rec/s")

    log.info("")
    log.info(f"  Records extracted: {len(records)}")
    log.info(f"  Has nuclearity:    {stats['has_nuclearity']} ({stats['has_nuclearity']/total*100:.1f}%)")
    log.info(f"  Has geometry:      {stats['has_geometry']} ({stats['has_geometry']/total*100:.1f}%)")
    log.info(f"  Has ox. states:    {stats['has_oxidation_states']} ({stats['has_oxidation_states']/total*100:.1f}%)")
    log.info(f"  Errors:            {len(errors)}")

    if errors:
        for e in errors[:10]:
            log.error(f"    {e}")

    if args.dry_run:
        log.info("\n  DRY RUN -- no files written.")
        return

    # Write index
    log.info(f"\nWriting {output_index}...")
    with open(output_index, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, separators=(",", ":"))
    size_mb = output_index.stat().st_size / (1024 * 1024)
    log.info(f"  Written: {output_index} ({size_mb:.1f} MB, {len(records):,} records)")

    # Write pretty sample
    log.info(f"Writing {output_sample}...")
    with open(output_sample, "w", encoding="utf-8") as f:
        json.dump(records[:10], f, indent=2, ensure_ascii=False)
    log.info(f"  Written: {output_sample} (10 records)")

    # Validation: compare old vs new record count
    old_index = output_dir / "hmof_index.json"
    if old_index.exists():
        old_data = json.load(open(old_index, "r", encoding="utf-8"))
        old_count = len(old_data)
        new_count = len(records)
        log.info(f"\n  Validation: old index has {old_count} records, new has {new_count}")
        if old_count != new_count:
            log.warning(f"  RECORD COUNT MISMATCH: {old_count} vs {new_count}")
        else:
            log.info(f"  Record count MATCH: {new_count}")

        # Compare a sample record
        old_by_id = {r["hmof_id"]: r for r in old_data[:100]}
        new_by_id = {r["hmof_id"]: r for r in records[:100]}
        common = set(old_by_id) & set(new_by_id)
        if common:
            sample_id = sorted(common)[0]
            old_r = old_by_id[sample_id]
            new_r = new_by_id[sample_id]
            # Check that old fields are preserved
            preserved = all(
                old_r.get(k) == new_r.get(k)
                for k in ["hmof_id", "metals", "functional_groups", "topology",
                          "surface_area_m2g", "density", "pld", "lcd", "synthesized"]
            )
            log.info(f"  Sample {sample_id}: old fields preserved = {preserved}")
            # Show new fields
            log.info(f"  Sample {sample_id}: NEW nuclearity={new_r.get('nuclearity')}, geometry={new_r.get('geometry')}")

    log.info("\n" + "=" * 60)
    log.info("DONE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
