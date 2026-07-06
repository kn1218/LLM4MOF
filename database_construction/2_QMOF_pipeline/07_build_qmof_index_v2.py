#!/usr/bin/env python3
"""
build_qmof_index_v2.py — QMOF Index Adapter Script

Reads 20,373 individual enriched JSON files from data/qmof/qmof_enriched_v2/
and generates a single monolithic data/qmof_index_v2.json in the OLD format
that the existing system expects, plus new enrichment fields.

Usage:
    python scripts/migration/build_qmof_index_v2.py

Standalone — no imports from core/ or config.py. stdlib only.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "qmof" / "qmof_enriched_v2"
OUTPUT_PATH = PROJECT_ROOT / "data" / "qmof_index_v2.json"
OLD_INDEX_PATH = PROJECT_ROOT / "data" / "qmof_index.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    """Nested dict access without KeyError."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def convert_record(src: dict[str, Any]) -> dict[str, Any]:
    """Map one enriched-v2 JSON → old-format dict with additive fields."""
    l1 = src.get("layer1_facts", {}) or {}
    l2 = src.get("layer2_semantics", {}) or {}
    mn = l1.get("metal_node", {}) or {}
    vr = src.get("validation_report", {}) or {}

    # topology: empty string → null
    topology_raw = l1.get("topology", "")
    topology = topology_raw if topology_raw else None

    # is_valid: True if status is "pass" or "warning"
    status = vr.get("status", "pass")
    is_valid = status in ("pass", "warning")

    # functional_groups: from layer2_semantics.functional_groups.rule_based
    fg_dict = l2.get("functional_groups", {}) or {}
    func_groups = fg_dict.get("rule_based", []) or []

    return {
        # --- backward-compatible fields ---
        "qmof_id": src.get("qmof_id", ""),
        "metals": mn.get("metals", []) or [],
        "functional_groups": list(func_groups),
        "topology": topology,
        "connectivity_points": mn.get("connectivity"),
        "geometry": mn.get("geometry"),
        "has_open_metal_sites": mn.get("has_open_metal_sites", False),
        "bandgap": l1.get("bandgap_pbe"),
        "is_valid": is_valid,
        # --- NEW additive fields ---
        "bandgap_hle17": l1.get("bandgap_hle17"),
        "bandgap_hse06_10hf": l1.get("bandgap_hse06_10hf"),
        "bandgap_hse06": l1.get("bandgap_hse06"),
        "oxidation_states": mn.get("oxidation_states"),
        "spin_state": mn.get("spin_state"),
        "coordinating_groups": mn.get("coordinating_groups"),
        "readable_name": l2.get("readable_name"),
        "synthesized": l1.get("synthesized"),
        "functional_groups_categorized": fg_dict if fg_dict else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()

    # 1. Discover source files
    pattern = str(SOURCE_DIR / "qmof-*.json")
    files = sorted(glob.glob(pattern))
    total = len(files)
    print(f"[build_qmof_index_v2] Found {total} enriched JSON files in {SOURCE_DIR}")
    if total == 0:
        print("ERROR: No files found. Exiting.")
        sys.exit(1)

    # 2. Read & convert
    results = []
    errors = []
    metal_counter = Counter()
    topo_counter = Counter()
    fg_counter = Counter()

    for i, fpath in enumerate(files):
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  ... processed {i + 1}/{total} files ({elapsed:.1f}s)")

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                src = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            errors.append((os.path.basename(fpath), str(e)))
            continue

        rec = convert_record(src)
        results.append(rec)

        # Accumulate stats
        for m in rec.get("metals") or []:
            metal_counter[m] += 1
        t = rec.get("topology")
        topo_counter[t if t else "(none)"] += 1
        for fg in rec.get("functional_groups") or []:
            fg_counter[fg] += 1

    # 3. Sort by qmof_id
    results.sort(key=lambda r: r.get("qmof_id", ""))

    elapsed = time.time() - t0
    print(f"\n[build_qmof_index_v2] Conversion complete in {elapsed:.1f}s")
    print(f"  Total converted:  {len(results)}")
    print(f"  Errors/skipped:   {len(errors)}")

    if errors:
        print(f"\n  First 10 errors:")
        for fname, msg in errors[:10]:
            print(f"    {fname}: {msg}")

    # 4. Summary stats
    print(f"\n--- SUMMARY ---")
    print(f"Total MOFs: {len(results)}")

    print(f"\nTop 10 metals:")
    for metal, count in metal_counter.most_common(10):
        print(f"  {metal:10s}  {count:>6d}")

    print(f"\nTop 10 topologies:")
    for topo, count in topo_counter.most_common(10):
        print(f"  {str(topo):20s}  {count:>6d}")

    print(f"\nTop 10 functional groups:")
    for fg, count in fg_counter.most_common(10):
        print(f"  {fg:25s}  {count:>6d}")

    # 5. Self-test: verify required fields
    print(f"\n--- SELF-TEST ---")
    missing_qmof_id = sum(1 for r in results if not r.get("qmof_id"))
    missing_metals = sum(1 for r in results if not r.get("metals"))
    missing_fg = sum(1 for r in results if not r.get("functional_groups"))
    missing_bg = sum(1 for r in results if r.get("bandgap") is None)

    print(f"  Missing qmof_id:          {missing_qmof_id}")
    print(f"  Missing metals:           {missing_metals}")
    print(f"  Missing functional_groups: {missing_fg}")
    print(f"  Missing bandgap:          {missing_bg}")

    all_ok = (missing_qmof_id == 0)
    if all_ok:
        print("  PASS: All records have qmof_id")
    else:
        print("  FAIL: Some records missing qmof_id")

    # 6. Compare with existing qmof_index.json
    print(f"\n--- COMPARISON WITH OLD INDEX ---")
    if OLD_INDEX_PATH.exists():
        try:
            with open(OLD_INDEX_PATH, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            old_ids = {r["qmof_id"] for r in old_data if "qmof_id" in r}
            new_ids = {r["qmof_id"] for r in results}

            shared = old_ids & new_ids
            only_old = old_ids - new_ids
            only_new = new_ids - old_ids

            print(f"  Old index:      {len(old_data)} entries")
            print(f"  New index:      {len(results)} entries")
            print(f"  Shared IDs:     {len(shared)}")
            print(f"  Only in old:    {len(only_old)}")
            print(f"  Only in new:    {len(only_new)}")

            # Field diff on first shared entry
            if shared:
                old_map = {r["qmof_id"]: r for r in old_data if "qmof_id" in r}
                new_map = {r["qmof_id"]: r for r in results}

                # Pick a known ID for comparison
                sample_id = sorted(shared)[0]
                old_rec = old_map[sample_id]
                new_rec = new_map[sample_id]

                old_keys = set(old_rec.keys())
                new_keys = set(new_rec.keys())
                added_keys = new_keys - old_keys
                removed_keys = old_keys - new_keys

                print(f"\n  Sample comparison ({sample_id}):")
                print(f"    Old fields: {sorted(old_keys)}")
                print(f"    New fields: {sorted(new_keys)}")
                if added_keys:
                    print(f"    NEW fields added: {sorted(added_keys)}")
                if removed_keys:
                    print(f"    Fields dropped:   {sorted(removed_keys)}")

                # Value diffs on shared keys
                diffs = []
                for k in sorted(old_keys & new_keys):
                    ov = old_rec.get(k)
                    nv = new_rec.get(k)
                    if ov != nv:
                        diffs.append(k)
                if diffs:
                    print(f"    Fields with value changes: {diffs}")
                    for k in diffs[:5]:
                        print(f"      {k}: old={old_rec.get(k)!r}  →  new={new_rec.get(k)!r}")

        except Exception as e:
            print(f"  Error loading old index: {e}")
    else:
        print(f"  Old index not found at {OLD_INDEX_PATH}")

    # 7. Write output
    print(f"\n--- WRITING OUTPUT ---")
    print(f"  Output: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    file_size = OUTPUT_PATH.stat().st_size
    print(f"  Size: {file_size / 1024 / 1024:.1f} MB")

    total_time = time.time() - t0
    print(f"\n[build_qmof_index_v2] Done in {total_time:.1f}s")


if __name__ == "__main__":
    main()
