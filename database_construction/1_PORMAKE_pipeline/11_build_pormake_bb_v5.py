#!/usr/bin/env python3
"""
build_pormake_bb_v5.py - Convert enriched BB JSONs to monolithic v5 dictionary.

Reads individual enriched JSON files from data/pormake/pormake_buildingblock_jsons/
and generates a single data/pormake_bb_dictionary_v5.json in the OLD format that the
existing system expects, plus NEW additive fields (sbu_type, design_hints,
abstract_features, functional_groups_categorized).

Usage:
    python scripts/migration/build_pormake_bb_v5.py

Must be run from the project root directory.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SOURCE_DIR = BASE_DIR / "data" / "pormake" / "pormake_buildingblock_jsons"
OUTPUT_PATH = BASE_DIR / "data" / "pormake_bb_dictionary_v5.json"
V4_PATH = BASE_DIR / "data" / "pormake_bb_dictionary_v4.json"

# Files to skip
SKIP_FILES = {"_crossval_report.json", "_pipeline_report.json"}

# Element symbol → full name mapping for connection_chemistry
ELEMENT_NAME_MAP = {
    "C": "Carbon",
    "N": "Nitrogen",
    "O": "Oxygen",
    "S": "Sulfur",
    "P": "Phosphorus",
    "H": "Hydrogen",
    "B": "Boron",
    "Si": "Silicon",
    "F": "Fluorine",
    "Cl": "Chlorine",
    "Br": "Bromine",
    "I": "Iodine",
}


def parse_bb_id(bb_id: str) -> tuple:
    """Parse BB ID into (prefix, number) for sorting. E.g. 'N109' -> ('N', 109)."""
    match = re.match(r"([A-Za-z]+)(\d+)", bb_id)
    if match:
        return (match.group(1), int(match.group(2)))
    return (bb_id, 0)


def resolve_connection_chemistry(chem_list: list) -> str:
    """
    Map element symbols to names. If all same, return single string.
    If mixed, return the most common element as a string.
    """
    if not chem_list:
        return "Carbon"  # safe default

    names = [ELEMENT_NAME_MAP.get(sym, sym) for sym in chem_list]
    counts = Counter(names)

    if len(counts) == 1:
        return names[0]
    else:
        # Return most common
        return counts.most_common(1)[0][0]


def safe_get(data: dict, *keys, default=None):
    """Safely traverse nested dict keys."""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
        if current is None and key != keys[-1]:
            return default
    return current


def convert_bb(src: dict) -> dict:
    """Convert a single enriched BB JSON to old-format dict."""
    l1 = src.get("layer1_facts", {})
    l2 = src.get("layer2_semantics", {})
    cp = l1.get("connection_points", {})
    fg = l2.get("functional_groups", {})

    # ligand_chemistry: null -> [], element symbols -> full names
    raw_lig_chem = l2.get("ligand_chemistry")
    if raw_lig_chem is None:
        lig_chem = []
    else:
        # Map element symbols to full names (v4 used "Oxygen", new data has "O")
        lig_chem = [ELEMENT_NAME_MAP.get(e, e) for e in raw_lig_chem]

    result = {
        # --- Core fields (v4-compatible) ---
        "ID": src.get("bb_id", ""),
        "Type": (src.get("bb_type", "")).capitalize(),
        "readable_name": l2.get("readable_name", ""),
        "smiles": l1.get("smiles", ""),
        "formula": l1.get("formula", ""),
        "connectivity": cp.get("count", 0),
        "is_rigid": l1.get("is_rigid", False),
        "functional_groups": fg.get("rule_based", []),
        "connection_chemistry": resolve_connection_chemistry(
            cp.get("connection_chemistry", [])
        ),
        "molecular_weight": l1.get("molecular_weight", 0.0),
        "length": l1.get("length_angstroms"),
        "metals": l1.get("metals", []),
        "nuclearity": l1.get("nuclearity"),
        "ligand_chemistry": lig_chem,
        # --- NEW additive fields ---
        "sbu_type": l2.get("sbu_type"),
        "design_hints": l2.get("design_hints", ""),
        "abstract_features": l2.get("abstract_features", {}),
        "functional_groups_categorized": {
            "backbone": fg.get("backbone", []),
            "substituents": fg.get("substituents", []),
            "rule_based": fg.get("rule_based", []),
            "rule_based_counts": fg.get("rule_based_counts", {}),
        },
    }

    return result


def sort_key(item: dict) -> tuple:
    """Sort: Nodes first (N), then Edges (E), numerically within each group."""
    prefix, num = parse_bb_id(item["ID"])
    # N sorts before E alphabetically — but we want N first, E second
    order = 0 if prefix.upper() == "N" else 1
    return (order, num)


def load_v4(path: Path) -> list:
    """Load v4 dictionary for comparison."""
    if not path.exists():
        print(f"  [WARN] v4 file not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def diff_v4_v5(v4_list: list, v5_list: list):
    """Compare v4 and v5 by ID and print diff summary."""
    v4_map = {item["ID"]: item for item in v4_list}
    v5_map = {item["ID"]: item for item in v5_list}

    v4_ids = set(v4_map.keys())
    v5_ids = set(v5_map.keys())

    common = v4_ids & v5_ids
    only_v4 = v4_ids - v5_ids
    only_v5 = v5_ids - v4_ids

    print(f"\n{'='*60}")
    print("  v4 vs v5 COMPARISON")
    print(f"{'='*60}")
    print(f"  v4 total entries:  {len(v4_list)}")
    print(f"  v5 total entries:  {len(v5_list)}")
    print(f"  Common IDs:        {len(common)}")
    print(f"  Only in v4:        {len(only_v4)}")
    print(f"  Only in v5:        {len(only_v5)}")

    if only_v4:
        sorted_v4 = sorted(only_v4, key=lambda x: parse_bb_id(x))
        print(f"\n  IDs only in v4 ({len(only_v4)}):")
        # Show first 20
        for bb_id in sorted_v4[:20]:
            print(f"    - {bb_id}")
        if len(sorted_v4) > 20:
            print(f"    ... and {len(sorted_v4) - 20} more")

    if only_v5:
        sorted_v5 = sorted(only_v5, key=lambda x: parse_bb_id(x))
        print(f"\n  IDs only in v5 ({len(only_v5)}):")
        for bb_id in sorted_v5[:20]:
            print(f"    - {bb_id}")
        if len(sorted_v5) > 20:
            print(f"    ... and {len(sorted_v5) - 20} more")

    # Field-level diff on common IDs
    if common:
        # Check which v4 fields are shared
        v4_fields_common = set()
        for bb_id in common:
            v4_fields_common.update(v4_map[bb_id].keys())

        field_diffs = Counter()
        field_matches = Counter()
        for bb_id in common:
            v4_item = v4_map[bb_id]
            v5_item = v5_map[bb_id]
            # Only compare fields present in v4
            for key in v4_item:
                if key in v5_item:
                    if v4_item[key] == v5_item[key]:
                        field_matches[key] += 1
                    else:
                        field_diffs[key] += 1
                else:
                    field_diffs[key] += 1

        print(f"\n  Field-level comparison on {len(common)} common IDs:")
        all_fields = sorted(set(field_diffs.keys()) | set(field_matches.keys()))
        for field in all_fields:
            m = field_matches.get(field, 0)
            d = field_diffs.get(field, 0)
            total = m + d
            if d > 0:
                print(f"    {field:30s}  match={m:4d}  diff={d:4d}  (of {total})")
            else:
                print(f"    {field:30s}  match={m:4d}  (all match)")


def self_test(path: Path):
    """Load v5 and verify required keys on every entry."""
    print(f"\n{'='*60}")
    print("  SELF-TEST")
    print(f"{'='*60}")
    required_keys = ["ID", "Type", "connectivity", "functional_groups", "readable_name"]

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    errors = []
    for item in data:
        for key in required_keys:
            if key not in item:
                errors.append(f"  {item.get('ID', '???')}: missing '{key}'")

    if errors:
        print(f"  FAIL - {len(errors)} issues:")
        for e in errors[:20]:
            print(f"    {e}")
    else:
        print(f"  PASS - All {len(data)} items have required keys: {required_keys}")


def main():
    print(f"{'='*60}")
    print("  build_pormake_bb_v5.py")
    print(f"{'='*60}")
    print(f"  BASE_DIR:    {BASE_DIR}")
    print(f"  SOURCE_DIR:  {SOURCE_DIR}")
    print(f"  OUTPUT_PATH: {OUTPUT_PATH}")

    # Validate source directory
    if not SOURCE_DIR.is_dir():
        print(f"\n  [ERROR] Source directory not found: {SOURCE_DIR}")
        sys.exit(1)

    # Collect JSON files
    json_files = sorted(SOURCE_DIR.glob("*.json"))
    print(f"\n  Found {len(json_files)} JSON files in source directory")

    # Filter out report files
    json_files = [
        f for f in json_files if f.name not in SKIP_FILES
    ]
    print(f"  After filtering reports: {len(json_files)} BB files")

    # Convert each file
    results = []
    warnings = []
    all_functional_groups = set()

    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                src = json.load(f)
        except json.JSONDecodeError as e:
            warnings.append(f"  [WARN] JSON parse error in {fpath.name}: {e}")
            continue
        except Exception as e:
            warnings.append(f"  [WARN] Read error for {fpath.name}: {e}")
            continue

        try:
            converted = convert_bb(src)
            results.append(converted)
            # Track functional groups
            for fg in converted.get("functional_groups", []):
                all_functional_groups.add(fg)
        except Exception as e:
            warnings.append(f"  [WARN] Conversion error for {fpath.name}: {e}")

    # Sort: Nodes first then Edges, numerically
    results.sort(key=sort_key)

    # Count nodes and edges
    node_count = sum(1 for r in results if r["Type"] == "Node")
    edge_count = sum(1 for r in results if r["Type"] == "Edge")

    # Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*60}")
    print("  BUILD SUMMARY")
    print(f"{'='*60}")
    print(f"  Total BBs written: {len(results)}")
    print(f"    Nodes:           {node_count}")
    print(f"    Edges:           {edge_count}")
    print(f"  Unique functional groups: {len(all_functional_groups)}")
    print(f"  Output file: {OUTPUT_PATH}")

    output_size = OUTPUT_PATH.stat().st_size
    print(f"  Output size: {output_size:,} bytes ({output_size / 1024:.1f} KB)")

    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    {w}")
    else:
        print(f"\n  No warnings.")

    # Show sample of functional groups
    fg_sorted = sorted(all_functional_groups)
    print(f"\n  Functional groups found ({len(fg_sorted)}):")
    for fg in fg_sorted[:30]:
        print(f"    - {fg}")
    if len(fg_sorted) > 30:
        print(f"    ... and {len(fg_sorted) - 30} more")

    # v4 comparison
    v4_data = load_v4(V4_PATH)
    if v4_data:
        diff_v4_v5(v4_data, results)

    # Self-test
    self_test(OUTPUT_PATH)

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
