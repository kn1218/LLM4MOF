#!/usr/bin/env python3
"""
build_hmof_index.py — Build compact hMOF index from 51K enriched JSON files.

Reads individual enriched JSON files from data/hMOF/hmof_enriched_v2/
and produces:
  1. data/hMOF/hmof_index.json          (compact, all records)
  2. data/hMOF/hmof_index_pretty_sample.json  (first 10, indented)

Standalone script — no external dependencies beyond stdlib.
"""

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "hMOF" / "hmof_enriched_v2"
OUTPUT_INDEX = PROJECT_ROOT / "data" / "hMOF" / "hmof_index.json"
OUTPUT_SAMPLE = PROJECT_ROOT / "data" / "hMOF" / "hmof_index_pretty_sample.json"

# Gas adsorption keys to extract
GAS_KEYS = [
    "h2_uptake_2bar_77K",
    "h2_uptake_100bar_77K",
    "ch4_uptake_35bar_298K",
    "co2_uptake_2_5bar_298K",
    "xe_loading_1bar_273K",
    "kr_loading_1bar_273K",
    "xekr_selectivity_1bar",
]

# Topology values to treat as null
TOPOLOGY_INVALID = {"ERROR", "NA.NA"}

# Regex to extract numeric ID from hMOF-<number>.json
HMOF_NUM_RE = re.compile(r"^hMOF-(\d+)\.json$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_get(d: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    """Nested dict lookup with graceful None handling."""
    current = d
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k, default)
    return current


def clean_topology(val):
    """Return None for invalid topology sentinels, else the value."""
    if val is None or (isinstance(val, str) and val in TOPOLOGY_INVALID):
        return None
    return val


def extract_record(data: dict[str, Any]) -> dict[str, Any]:
    """Extract compact index record from a full enriched JSON object."""
    l1 = data.get("layer1_facts") or {}
    l2 = data.get("layer2_semantics") or {}
    gas = l1.get("gas_adsorption") or {}
    metal_node = l1.get("metal_node") or {}
    fg = l2.get("functional_groups") or {}
    vr = data.get("validation_report") or {}

    record: dict[str, Any] = {
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

    # Gas adsorption — flat keys
    for gk in GAS_KEYS:
        record[gk] = gas.get(gk)

    record["synthesized"] = l1.get("synthesized")
    record["is_valid"] = vr.get("status") != "fail"
    record["readable_name"] = l2.get("readable_name")
    record["has_open_metal_sites"] = metal_node.get("has_open_metal_sites")

    # Full categorized functional groups (backbone, substituents, rule_based, rule_based_counts)
    record["functional_groups_categorized"] = {
        "backbone": fg.get("backbone") or [],
        "substituents": fg.get("substituents") or [],
        "rule_based": fg.get("rule_based") or [],
        "rule_based_counts": fg.get("rule_based_counts") or {},
    }

    return record


def numeric_hmof_key(record: dict[str, Any]) -> int:
    """Sort key: extract numeric part from hmof_id for natural ordering."""
    hmof_id = record.get("hmof_id", "")
    m = re.search(r"(\d+)$", str(hmof_id))
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("hMOF Index Builder")
    print("=" * 60)

    if not SOURCE_DIR.is_dir():
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        sys.exit(1)

    print(f"Source : {SOURCE_DIR}")
    print(f"Output : {OUTPUT_INDEX}")
    print(f"Sample : {OUTPUT_SAMPLE}")
    print()

    # ------------------------------------------------------------------
    # Phase 1: Scan directory for hMOF-*.json files using os.scandir
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    json_files: list[tuple[int, str]] = []

    with os.scandir(SOURCE_DIR) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            m = HMOF_NUM_RE.match(entry.name)
            if m:
                json_files.append((int(m.group(1)), entry.path))

    # Sort by numeric ID for deterministic processing & output order
    json_files.sort(key=lambda x: x[0])
    total = len(json_files)
    t_scan = time.perf_counter() - t0
    print(f"Found {total:,} hMOF JSON files (scan: {t_scan:.2f}s)")

    if total == 0:
        print("ERROR: No hMOF-*.json files found. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 2: Read & extract
    # ------------------------------------------------------------------
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    t1 = time.perf_counter()

    for i, (num_id, fpath) in enumerate(json_files):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            records.append(extract_record(data))
        except Exception as e:
            errors.append(f"{os.path.basename(fpath)}: {e}")

        # Progress every 10,000
        count = i + 1
        if count % 10000 == 0 or count == total:
            elapsed = time.perf_counter() - t1
            rate = count / elapsed if elapsed > 0 else 0
            print(f"  Processed {count:>6,} / {total:,}  "
                  f"({count * 100 / total:.1f}%)  "
                  f"[{rate:.0f} files/s]")

    t_read = time.perf_counter() - t1
    print(f"\nExtraction complete: {len(records):,} records in {t_read:.2f}s")
    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors[:10]:
            print(f"    {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")

    # ------------------------------------------------------------------
    # Phase 3: Self-test
    # ------------------------------------------------------------------
    print("\n--- Self-Test ---")
    test_pass = True
    missing_id = sum(1 for r in records if not r.get("hmof_id"))
    missing_metals = sum(1 for r in records if not r.get("metals"))
    missing_fg = sum(1 for r in records if not r.get("functional_groups"))

    # At least one gas adsorption value present
    missing_gas = 0
    for r in records:
        has_any_gas = any(r.get(gk) is not None for gk in GAS_KEYS)
        if not has_any_gas:
            missing_gas += 1

    for label, count in [
        ("hmof_id", missing_id),
        ("metals", missing_metals),
        ("functional_groups", missing_fg),
        ("gas_adsorption (>=1)", missing_gas),
    ]:
        status = "PASS" if count == 0 else "WARN"
        if count > 0:
            test_pass = False
        print(f"  {label:30s}: {status} (missing: {count})")

    if test_pass:
        print("  Self-test: ALL PASS")
    else:
        print("  Self-test: WARNINGS detected (non-fatal, continuing)")

    # ------------------------------------------------------------------
    # Phase 4: Summary statistics
    # ------------------------------------------------------------------
    print("\n--- Summary ---")
    print(f"Total MOFs: {len(records):,}")

    # Metals distribution
    metal_counter: Counter[str] = Counter()
    for r in records:
        for m in r.get("metals") or []:
            metal_counter[m] += 1
    print(f"\nMetals distribution (top 5):")
    for metal, cnt in metal_counter.most_common(5):
        print(f"  {metal:6s}: {cnt:>6,}")

    # Topology distribution
    topo_counter: Counter[str] = Counter()
    for r in records:
        t = r.get("topology") or "(null)"
        topo_counter[t] += 1
    print(f"\nTopology distribution (top 10):")
    for topo, cnt in topo_counter.most_common(10):
        print(f"  {topo:20s}: {cnt:>6,}")

    # Gas adsorption stats — H2 at 100bar
    h2_vals = [r["h2_uptake_100bar_77K"] for r in records
               if r.get("h2_uptake_100bar_77K") is not None]
    if h2_vals:
        print(f"\nH2 uptake (100bar, 77K) stats ({len(h2_vals):,} values):")
        print(f"  min:  {min(h2_vals):.4f}")
        print(f"  max:  {max(h2_vals):.4f}")
        print(f"  mean: {sum(h2_vals) / len(h2_vals):.4f}")

    # Unique functional groups
    all_fgs: set[str] = set()
    for r in records:
        all_fgs.update(r.get("functional_groups") or [])
    print(f"\nUnique functional groups: {len(all_fgs)}")

    # ------------------------------------------------------------------
    # Phase 5: Write compact index (streaming)
    # ------------------------------------------------------------------
    print(f"\n--- Writing Output ---")
    t_write = time.perf_counter()

    # Streaming write: open file and write records one by one
    # This avoids building the full JSON string in memory
    with open(OUTPUT_INDEX, "w", encoding="utf-8") as f:
        f.write("[")
        for i, rec in enumerate(records):
            if i > 0:
                f.write(",")
            json.dump(rec, f, separators=(",", ":"), ensure_ascii=False)
        f.write("]")

    size_mb = os.path.getsize(OUTPUT_INDEX) / (1024 * 1024)
    t_w = time.perf_counter() - t_write
    print(f"  {OUTPUT_INDEX.name}: {size_mb:.1f} MB ({t_w:.2f}s)")

    # Pretty sample (first 10)
    sample = records[:10]
    with open(OUTPUT_SAMPLE, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    sample_kb = os.path.getsize(OUTPUT_SAMPLE) / 1024
    print(f"  {OUTPUT_SAMPLE.name}: {sample_kb:.1f} KB")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    total_time = time.perf_counter() - t0
    print(f"\nDone in {total_time:.1f}s")


if __name__ == "__main__":
    main()
