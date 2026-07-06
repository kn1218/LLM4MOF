"""
Enrich QMOF index v2 with generic/abstract tags.

Same logic as enrich_generic_tags.py for PORMAKE, but applied to the QMOF index.
Also compares v1 vs enriched v2 tag matching.
"""

import json
from pathlib import Path
from collections import Counter

# Reuse the same tag hierarchy
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from enrich_generic_tags import enrich_tags, TAG_HIERARCHY

BASE_DIR = Path(__file__).resolve().parent.parent.parent
V2_PATH = BASE_DIR / "data" / "qmof_index_v2.json"
V1_PATH = BASE_DIR / "data" / "qmof_index.json"


def canon(s: str) -> str:
    if not s:
        return ""
    return s.lower().strip().replace("-", "_").replace(" ", "_")


def main():
    print("=" * 60)
    print("QMOF INDEX v2 - GENERIC TAG ENRICHMENT")
    print("=" * 60)

    # Load v2
    with open(V2_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} QMOF entries")

    # Enrich
    tags_before = Counter()
    tags_after = Counter()
    items_enriched = 0

    for item in data:
        fg = item.get("functional_groups", [])
        for t in fg:
            tags_before[t] += 1

        enriched = enrich_tags(fg)
        new_tags = set(enriched) - set(fg)
        if new_tags:
            items_enriched += 1
        item["functional_groups"] = enriched

        for t in enriched:
            tags_after[t] += 1

    # Save
    with open(V2_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    unique_before = len(tags_before)
    unique_after = len(tags_after)
    new_generic = set(tags_after.keys()) - set(tags_before.keys())

    print(f"\nItems enriched: {items_enriched} / {len(data)}")
    print(f"Unique tags before: {unique_before}")
    print(f"Unique tags after:  {unique_after}")
    print(f"New generic tags: {len(new_generic)}")

    if new_generic:
        print("\nNew generic tags added:")
        for tag in sorted(new_generic):
            print(f"  {tag:30s} {tags_after[tag]:>6} MOFs")

    # Key coverage
    print("\nKey generic tag coverage:")
    for kt in ["aromatic", "ring", "heterocycle", "nitrogen", "oxygen",
                "sulfur", "halogen", "carbonyl", "carbon_framework"]:
        count = tags_after.get(kt, 0)
        pct = (count / len(data)) * 100
        status = "OK" if count > 0 else "MISSING"
        print(f"  [{status}] {kt:25s} {count:>6} MOFs ({pct:.1f}%)")

    # Compare with v1
    if V1_PATH.exists():
        print("\n--- COMPARISON WITH v1 ---")
        with open(V1_PATH, "r", encoding="utf-8") as f:
            v1_data = json.load(f)

        v1_lookup = {x["qmof_id"]: x for x in v1_data}
        v2_lookup = {x["qmof_id"]: x for x in data}

        common_ids = set(v1_lookup.keys()) & set(v2_lookup.keys())
        print(f"Common IDs: {len(common_ids)}")

        # Test: search for "Aromatic" in both
        v1_aromatic = sum(1 for x in v1_data if "Aromatic" in x.get("functional_groups", []))
        v2_aromatic = sum(1 for x in data if "aromatic" in [canon(t) for t in x.get("functional_groups", [])])
        print(f"MOFs with 'Aromatic': v1={v1_aromatic}  v2={v2_aromatic}")

        v1_nitrogen = sum(1 for x in v1_data if "Nitrogen" in x.get("functional_groups", []))
        v2_nitrogen = sum(1 for x in data if "nitrogen" in [canon(t) for t in x.get("functional_groups", [])])
        print(f"MOFs with 'Nitrogen': v1={v1_nitrogen}  v2={v2_nitrogen}")

        v1_carboxyl = sum(1 for x in v1_data if "Carboxyl" in x.get("functional_groups", []))
        v2_carboxyl = sum(1 for x in data if "carboxyl" in [canon(t) for t in x.get("functional_groups", [])])
        print(f"MOFs with 'Carboxyl': v1={v1_carboxyl}  v2={v2_carboxyl}")

        v1_oxygen = sum(1 for x in v1_data if "Oxygen" in x.get("functional_groups", []))
        v2_oxygen = sum(1 for x in data if "oxygen" in [canon(t) for t in x.get("functional_groups", [])])
        print(f"MOFs with 'Oxygen':   v1={v1_oxygen}  v2={v2_oxygen}")

    print(f"\nSaved enriched v2 to: {V2_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
