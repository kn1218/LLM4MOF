"""
enrich_generic_tags.py - Post-process v5 BB dictionary to add generic/abstract tags.

PROBLEM:
  Old v4 BB dictionary had BOTH specific AND generic tags on each item:
    v4: ["Aromatic", "Aryl", "Benzene", "Ring", "Nitrogen", "Heterocycle", "Pyridine"]
  New v5 only has SMARTS-detected SPECIFIC tags:
    v5: ["benzene_ring", "pyridine"]

  When Agent 2 outputs "Aromatic" and the matchmaker searches for it,
  v4 BBs match (they have "Aromatic") but v5 BBs don't (they only have "benzene_ring").

SOLUTION:
  Derive generic tags from specific ones using a hierarchy mapping.
  If a BB has "benzene_ring" -> also add "aromatic", "aryl", "ring".
  If a BB has "pyridine" -> also add "heterocycle", "nitrogen", "ring".

  This is purely additive - no existing tags are removed.

USAGE:
  python scripts/migration/enrich_generic_tags.py

  Input:  data/pormake_bb_dictionary_v5.json
  Output: data/pormake_bb_dictionary_v5.json (in-place enrichment)
          + prints stats
"""

import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent.parent
V5_PATH = BASE_DIR / "data" / "pormake_bb_dictionary_v5.json"

# ── HIERARCHY: specific_tag -> set of generic tags to also add ──────────
# These reflect the abstract groupings the old BB dictionary used.
# All tags here are in LOWERED snake_case (the v5 canonical form).
TAG_HIERARCHY = {
    # Aromatic rings -> add "aromatic", "aryl", "ring"
    "benzene_ring":     {"aromatic", "aryl", "ring"},
    "naphthalene":      {"aromatic", "aryl", "ring"},
    "biphenyl":         {"aromatic", "aryl", "ring"},
    "anthracene":       {"aromatic", "aryl", "ring"},
    "phenanthrene":     {"aromatic", "aryl", "ring"},
    "fluorene":         {"aromatic", "aryl", "ring"},
    "terphenyl":        {"aromatic", "aryl", "ring"},
    "pyrene":           {"aromatic", "aryl", "ring"},
    "triptycene":       {"aromatic", "aryl", "ring"},

    # Heterocycles -> add "heterocycle", "ring", and sometimes element tags
    "pyridine":         {"heterocycle", "ring", "nitrogen", "aromatic"},
    "imidazole":        {"heterocycle", "ring", "nitrogen", "aromatic", "azole", "azolate"},
    "thiophene":        {"heterocycle", "ring", "sulfur", "aromatic"},
    "furan":            {"heterocycle", "ring", "oxygen", "aromatic"},
    "pyrazole":         {"heterocycle", "ring", "nitrogen", "aromatic", "azole"},
    "triazole":         {"heterocycle", "ring", "nitrogen", "aromatic", "azole", "azolate"},
    "triazole_any":     {"heterocycle", "ring", "nitrogen", "aromatic", "azole", "azolate"},
    "triazole_1_2_4":   {"heterocycle", "ring", "nitrogen", "aromatic", "azole", "azolate"},
    "tetrazole":        {"heterocycle", "ring", "nitrogen", "aromatic", "azole", "azolate"},
    "tetrazine":        {"heterocycle", "ring", "nitrogen", "aromatic"},
    "pyrimidine":       {"heterocycle", "ring", "nitrogen", "aromatic"},
    "pyrazine":         {"heterocycle", "ring", "nitrogen", "aromatic"},
    "pyridazine":       {"heterocycle", "ring", "nitrogen", "aromatic"},
    "triazine":         {"heterocycle", "ring", "nitrogen", "aromatic"},
    "oxadiazole":       {"heterocycle", "ring", "nitrogen", "oxygen", "aromatic"},
    "thiadiazole":      {"heterocycle", "ring", "nitrogen", "sulfur", "aromatic"},
    "thiazole":         {"heterocycle", "ring", "nitrogen", "sulfur", "aromatic"},
    "quinoline":        {"heterocycle", "ring", "nitrogen", "aromatic"},
    "isoquinoline":     {"heterocycle", "ring", "nitrogen", "aromatic"},
    "benzofuran":       {"heterocycle", "ring", "oxygen", "aromatic"},
    "benzothiophene":   {"heterocycle", "ring", "sulfur", "aromatic"},
    "benzodioxole":     {"heterocycle", "ring", "oxygen", "aromatic"},
    "piperazine":       {"heterocycle", "ring", "nitrogen"},
    "piperidine":       {"heterocycle", "ring", "nitrogen"},
    "pyridinium":       {"heterocycle", "ring", "nitrogen", "aromatic"},
    "benzimidazole":    {"heterocycle", "ring", "nitrogen", "aromatic", "azole"},
    "benzothiazole":    {"heterocycle", "ring", "nitrogen", "sulfur", "aromatic"},
    "isothiazole":      {"heterocycle", "ring", "nitrogen", "sulfur", "aromatic"},
    "phenanthroline":   {"heterocycle", "ring", "nitrogen", "aromatic"},
    "bipyridine":       {"heterocycle", "ring", "nitrogen", "aromatic"},
    "pyrrole":          {"heterocycle", "ring", "nitrogen", "aromatic"},

    # Non-aromatic rings
    "cyclohexane":      {"ring", "aliphatic_ring"},
    "cyclopentane":     {"ring", "aliphatic_ring"},
    "adamantane":       {"ring", "aliphatic_ring"},
    "cubane":           {"ring", "aliphatic_ring"},

    # Nitrogen-containing groups -> add "nitrogen"
    "amine_any":        {"nitrogen"},
    "primary_amine":    {"nitrogen", "amine"},
    "secondary_amine":  {"nitrogen", "amine"},
    "tertiary_amine":   {"nitrogen", "amine"},
    "amide":            {"nitrogen", "oxygen", "carbonyl"},
    "imine":            {"nitrogen"},
    "imine_any":        {"nitrogen"},
    "nitro":            {"nitrogen", "oxygen"},
    "nitrile":          {"nitrogen"},
    "nitrile_sub":      {"nitrogen"},
    "azo":              {"nitrogen"},
    "hydrazide":        {"nitrogen", "oxygen"},
    "urea":             {"nitrogen", "oxygen", "carbonyl"},
    "primary_amide":    {"nitrogen", "oxygen", "carbonyl"},
    "isothiocyanate":   {"nitrogen", "sulfur"},
    "thiourea":         {"nitrogen", "sulfur"},
    "sulfonamide":      {"nitrogen", "sulfur", "oxygen"},

    # Oxygen-containing groups -> add "oxygen"
    "carboxyl_any":     {"oxygen", "carbonyl"},
    "carboxylate":      {"oxygen", "carbonyl"},
    "carboxylic_acid":  {"oxygen", "carbonyl"},
    "hydroxyl":         {"oxygen"},
    "phenol":           {"oxygen", "aromatic"},
    "ether":            {"oxygen"},
    "aryl_ether":       {"oxygen", "aromatic"},
    "methoxy":          {"oxygen"},
    "ester":            {"oxygen", "carbonyl"},
    "aldehyde":         {"oxygen", "carbonyl"},
    "ketone":           {"oxygen", "carbonyl"},
    "phosphonate":      {"oxygen", "phosphorus"},
    "phosphonic_acid":  {"oxygen", "phosphorus"},
    "sulfonate":        {"oxygen", "sulfur"},
    "sulfonic_acid":    {"oxygen", "sulfur"},
    "sulfonyl":         {"oxygen", "sulfur"},
    "sulfone":          {"oxygen", "sulfur"},

    # Sulfur-containing -> add "sulfur"
    "thiol":            {"sulfur"},
    "thioether":        {"sulfur"},

    # Halogens -> add "halogen"
    "fluorine":         {"halogen"},
    "chlorine":         {"halogen"},
    "bromine":          {"halogen"},
    "iodine":           {"halogen"},
    "trifluoromethyl":  {"halogen"},

    # Carbon framework tags
    "vinyl":            {"carbon_framework"},
    "acetylene":        {"carbon_framework"},
    "alkene":           {"carbon_framework"},
    "alkyne":           {"carbon_framework"},
    "butadiyne":        {"carbon_framework"},
    "methyl":           set(),  # no generic tag needed
    "ethyl":            set(),
    "isopropyl":        set(),
    "tert_butyl":       set(),

    # Boron
    "boron_any":        set(),
    "boronic_acid":     {"oxygen"},
}


def enrich_tags(functional_groups: list) -> list:
    """
    Given a list of specific functional group tags,
    derive and add generic/abstract tags.
    Returns a NEW list with both specific and generic tags (no duplicates).
    """
    enriched = set(functional_groups)

    for tag in functional_groups:
        tag_lower = tag.lower().strip()
        generic = TAG_HIERARCHY.get(tag_lower, set())
        enriched.update(generic)

    return sorted(enriched)


def main():
    print("=" * 60)
    print("GENERIC TAG ENRICHMENT FOR v5 BB DICTIONARY")
    print("=" * 60)

    # Load v5
    with open(V5_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} building blocks from v5")

    # Track stats
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

    # Save back
    with open(V5_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Stats
    unique_before = len(tags_before)
    unique_after = len(tags_after)
    new_generic_tags = set(tags_after.keys()) - set(tags_before.keys())

    print(f"\nItems enriched: {items_enriched} / {len(data)}")
    print(f"Unique tags before: {unique_before}")
    print(f"Unique tags after:  {unique_after}")
    print(f"New generic tags added: {len(new_generic_tags)}")

    if new_generic_tags:
        print("\nNew generic tags (with occurrence counts):")
        for tag in sorted(new_generic_tags):
            print(f"  {tag:30s} {tags_after[tag]:>5} BBs")

    # Verify key generic tags exist
    print("\nKey generic tag coverage check:")
    key_tags = ["aromatic", "ring", "heterocycle", "nitrogen", "oxygen",
                "sulfur", "halogen", "carbonyl", "carbon_framework"]
    for kt in key_tags:
        count = tags_after.get(kt, 0)
        pct = (count / len(data)) * 100
        status = "OK" if count > 0 else "MISSING"
        print(f"  [{status}] {kt:25s} {count:>4} BBs ({pct:.1f}%)")

    print(f"\nSaved enriched v5 to: {V5_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
