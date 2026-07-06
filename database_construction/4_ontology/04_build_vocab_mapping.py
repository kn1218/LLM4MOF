"""
Phase 0D: Build vocabulary mapping between new SMARTS-detected tags and old ontology.

Reads a sample of new enriched JSONs from all three databases (PORMAKE, QMOF, hMOF),
collects all unique functional group tags, then maps them against the existing
unified_ontology.json to find:
  1. Tags that already have mappings (covered)
  2. Tags that are new and need aliases added to the ontology
  3. Old ontology tags that don't appear in new data (potentially obsolete)

Output: A proposed ontology patch file at data/ontology_v2_patch.json
"""

import json
import os
import glob
from collections import Counter

# ── paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

ONTOLOGY_PATH = os.path.join(DATA_DIR, "unified_ontology.json")
PORMAKE_BB_DIR = os.path.join(DATA_DIR, "pormake", "pormake_buildingblock_jsons")
QMOF_ENRICHED_DIR = os.path.join(DATA_DIR, "qmof", "qmof_enriched_v2")
HMOF_ENRICHED_DIR = os.path.join(DATA_DIR, "hMOF", "hmof_enriched_v2")
OLD_BB_DICT_PATH = os.path.join(DATA_DIR, "pormake_bb_dictionary_v4.json")
OLD_QMOF_INDEX_PATH = os.path.join(DATA_DIR, "qmof_index.json")

OUTPUT_PATH = os.path.join(DATA_DIR, "ontology_v2_patch.json")


def load_ontology(path: str) -> dict:
    """Load ontology and build alias→canonical mapping."""
    with open(path, "r", encoding="utf-8") as f:
        onto = json.load(f)

    # Build reverse map: normalized_alias → canonical_tag
    alias_to_canonical = {}
    canonical_tags = set()

    for canonical, info in onto.get("canonical_tags", {}).items():
        norm = canonical.lower().strip().replace("-", "_").replace(" ", "_")
        canonical_tags.add(norm)
        alias_to_canonical[norm] = norm  # self-map

        for alias in info.get("aliases", []):
            a_norm = alias.lower().strip().replace("-", "_").replace(" ", "_")
            alias_to_canonical[a_norm] = norm

    return onto, alias_to_canonical, canonical_tags


def collect_new_tags_from_pormake(directory: str, sample_limit: int = 0) -> Counter:
    """Collect all rule_based functional group tags from PORMAKE enriched JSONs."""
    tag_counter = Counter()
    files = [f for f in os.listdir(directory)
             if f.endswith(".json") and not f.startswith("_")]

    if sample_limit > 0:
        files = files[:sample_limit]

    for fname in files:
        fpath = os.path.join(directory, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        fg = data.get("layer2_semantics", {}).get("functional_groups", {})
        for tag in fg.get("rule_based", []):
            tag_counter[tag] += 1

    return tag_counter


def collect_new_tags_from_mof_db(directory: str, sample_limit: int = 2000) -> Counter:
    """Collect rule_based tags from enriched MOF JSONs (QMOF or hMOF). Sample for speed."""
    tag_counter = Counter()
    count = 0

    for entry in os.scandir(directory):
        if not entry.name.endswith(".json"):
            continue
        count += 1
        if sample_limit > 0 and count > sample_limit:
            break

        with open(entry.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fg = data.get("layer2_semantics", {}).get("functional_groups", {})
        for tag in fg.get("rule_based", []):
            tag_counter[tag] += 1

    return tag_counter


def collect_old_tags_from_bb_dict(path: str) -> Counter:
    """Collect functional group tags from the old BB dictionary."""
    tag_counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        for tag in item.get("functional_groups", []):
            tag_counter[tag] += 1
    return tag_counter


def collect_old_tags_from_qmof_index(path: str) -> Counter:
    """Collect functional group tags from the old QMOF index."""
    tag_counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        for tag in item.get("functional_groups", []):
            tag_counter[tag] += 1
    return tag_counter


def propose_mapping(new_tag: str) -> str | None:
    """
    Propose a canonical ontology tag for a new SMARTS-detected tag.
    Returns the proposed canonical form or None if no obvious mapping.
    """
    # Known direct mappings between SMARTS snake_case and ontology
    KNOWN_MAPPINGS = {
        # Scaffolds / Rings
        "benzene_ring": "benzene",
        "naphthalene": "naphthalene",
        "biphenyl": "biphenyl",
        "anthracene": "anthracene",
        "phenanthrene": "phenanthrene",
        "fluorene": "fluorene",
        "terphenyl": "terphenyl",
        "pyrene": "pyrene",
        "triptycene": "triptycene",
        "cyclohexane": "cyclohexane",
        "cyclopentane": "cyclopentane",
        "adamantane": "adamantane",
        "cubane": "cubane",
        "indane": "indane",

        # Functional groups
        "carboxyl_any": "carboxyl",
        "carboxylate": "carboxyl",
        "carboxylic_acid": "carboxyl",
        "amine_any": "amine",
        "primary_amine": "primary_amine",
        "secondary_amine": "secondary_amine",
        "tertiary_amine": "tertiary_amine",
        "imine_any": "imine",
        "amide": "amide",
        "hydroxyl": "hydroxyl",
        "carbonyl": "carbonyl",
        "ketone": "ketone",
        "aldehyde": "aldehyde",
        "ether": "ether",
        "thioether": "thioether",
        "methoxy": "methoxy",
        "nitro": "nitro",
        "nitrile": "nitrile",
        "azo": "azo",
        "thiol": "thiol",
        "sulfonyl": "sulfonyl",
        "sulfonate": "sulfonate",
        "phosphonate": "phosphonate",
        "phenoxide": "phenoxide",
        "trifluoromethyl": "trifluoromethyl",

        # Heterocycles
        "pyridine": "pyridine",
        "imidazole": "imidazole",
        "thiophene": "thiophene",
        "furan": "furan",
        "pyrazole": "pyrazole",
        "triazole": "triazole",
        "triazole_124": "triazole",
        "triazole_123": "triazole",
        "tetrazole": "tetrazole",
        "tetrazine": "tetrazine",
        "pyrimidine": "pyrimidine",
        "pyrazine": "pyrazine",
        "pyridazine": "pyridazine",
        "triazine": "triazine",
        "oxadiazole": "oxadiazole",
        "thiadiazole": "thiadiazole",
        "thiazole": "thiazole",
        "quinoline": "quinoline",
        "isoquinoline": "isoquinoline",
        "benzofuran": "benzofuran",
        "benzothiophene": "benzothiophene",
        "benzodioxole": "benzodioxole",
        "piperazine": "piperazine",
        "piperidine": "piperidine",
        "pyridinium": "pyridinium",

        # Halogens
        "fluorine": "fluoro",
        "chlorine": "chloro",
        "bromine": "bromo",
        "iodine": "iodo",

        # Substituents
        "methyl": "methyl",
        "ethyl": "ethyl",
        "isopropyl": "isopropyl",
        "tert_butyl": "tert_butyl",

        # Linker types
        "vinyl": "alkene",
        "acetylene": "alkyne",
        "alkene": "alkene",
        "alkyne": "alkyne",
    }

    tag_lower = new_tag.lower().strip().replace("-", "_").replace(" ", "_")
    return KNOWN_MAPPINGS.get(tag_lower)


def main():
    print("=" * 70)
    print("PHASE 0D: VOCABULARY MAPPING ANALYSIS")
    print("=" * 70)

    # ── Load existing ontology ───────────────────────────────────────────
    print("\n1. Loading existing ontology...")
    ontology, alias_map, canonical_tags = load_ontology(ONTOLOGY_PATH)
    print(f"   Canonical tags: {len(canonical_tags)}")
    print(f"   Alias mappings: {len(alias_map)}")

    # ── Collect tags from ALL new data sources ───────────────────────────
    print("\n2. Collecting tags from new enriched data...")

    print("   Reading PORMAKE building blocks (all 869)...")
    pormake_tags = collect_new_tags_from_pormake(PORMAKE_BB_DIR)
    print(f"   → {len(pormake_tags)} unique tags, {sum(pormake_tags.values())} total occurrences")

    print("   Sampling QMOF enriched (2000 files)...")
    qmof_tags = collect_new_tags_from_mof_db(QMOF_ENRICHED_DIR, sample_limit=2000)
    print(f"   → {len(qmof_tags)} unique tags, {sum(qmof_tags.values())} total occurrences")

    print("   Sampling hMOF enriched (2000 files)...")
    hmof_tags = collect_new_tags_from_mof_db(HMOF_ENRICHED_DIR, sample_limit=2000)
    print(f"   → {len(hmof_tags)} unique tags, {sum(hmof_tags.values())} total occurrences")

    # Merge all new tags
    all_new_tags = Counter()
    all_new_tags.update(pormake_tags)
    all_new_tags.update(qmof_tags)
    all_new_tags.update(hmof_tags)
    print(f"\n   ALL NEW TAGS: {len(all_new_tags)} unique")

    # ── Collect tags from OLD data ───────────────────────────────────────
    print("\n3. Collecting tags from OLD data...")

    old_bb_tags = collect_old_tags_from_bb_dict(OLD_BB_DICT_PATH)
    print(f"   Old BB dictionary: {len(old_bb_tags)} unique tags")

    old_qmof_tags = collect_old_tags_from_qmof_index(OLD_QMOF_INDEX_PATH)
    print(f"   Old QMOF index: {len(old_qmof_tags)} unique tags")

    all_old_tags = Counter()
    all_old_tags.update(old_bb_tags)
    all_old_tags.update(old_qmof_tags)
    print(f"   ALL OLD TAGS: {len(all_old_tags)} unique")

    # ── Classify new tags ────────────────────────────────────────────────
    print("\n4. Classifying new tags against ontology...")

    covered = {}      # tag → canonical (already has alias in ontology)
    needs_alias = {}   # tag → proposed_canonical (mapping known, alias missing)
    unmapped = []      # tag with no known mapping

    for tag in sorted(all_new_tags.keys()):
        tag_norm = tag.lower().strip().replace("-", "_").replace(" ", "_")

        if tag_norm in alias_map:
            covered[tag] = alias_map[tag_norm]
        else:
            proposed = propose_mapping(tag)
            if proposed:
                needs_alias[tag] = proposed
            else:
                unmapped.append(tag)

    print(f"\n   [OK] COVERED (already in ontology): {len(covered)} tags")
    for tag, canon in sorted(covered.items()):
        count = all_new_tags[tag]
        print(f"      {tag:30s} → {canon:25s} ({count:,} occurrences)")

    print(f"\n   [WARN] NEEDS ALIAS (mapping known, not in ontology): {len(needs_alias)} tags")
    for tag, canon in sorted(needs_alias.items()):
        count = all_new_tags[tag]
        print(f"      {tag:30s} → {canon:25s} ({count:,} occurrences)")

    print(f"\n   [NEW] UNMAPPED (no known mapping): {len(unmapped)} tags")
    for tag in sorted(unmapped):
        count = all_new_tags[tag]
        print(f"      {tag:30s} ({count:,} occurrences)")

    # ── Check for old tags missing from new data ─────────────────────────
    print("\n5. Old tags NOT found in new data...")
    old_only = set()
    for tag in all_old_tags:
        tag_norm = tag.lower().strip().replace("-", "_").replace(" ", "_")
        # Check if this old tag (or its canonical form) appears in new data
        found = False
        for new_tag in all_new_tags:
            new_norm = new_tag.lower().strip().replace("-", "_").replace(" ", "_")
            if new_norm == tag_norm:
                found = True
                break
            # Check if they map to the same canonical
            old_canon = alias_map.get(tag_norm, tag_norm)
            new_proposed = propose_mapping(new_tag)
            if new_proposed and new_proposed == old_canon:
                found = True
                break
        if not found:
            old_only.add(tag)

    if old_only:
        print(f"   {len(old_only)} old tags have no equivalent in new data:")
        for tag in sorted(old_only):
            count = all_old_tags[tag]
            print(f"      {tag:30s} ({count:,} occurrences in old data)")
    else:
        print("   All old tags have equivalents in new data.")

    # ── Generate ontology patch ──────────────────────────────────────────
    print("\n6. Generating ontology patch...")

    patch = {
        "description": "Ontology v2 patch: adds SMARTS-detected tag aliases from enriched v2 data",
        "generated_by": "scripts/migration/build_vocab_mapping.py",
        "new_aliases": {},
        "new_canonical_tags": {},
        "unmapped_tags": unmapped,
        "stats": {
            "covered": len(covered),
            "needs_alias": len(needs_alias),
            "unmapped": len(unmapped),
            "old_only": len(old_only),
        }
    }

    # For tags that need aliases: group by canonical target
    from collections import defaultdict
    alias_groups = defaultdict(list)
    for tag, canon in needs_alias.items():
        alias_groups[canon].append(tag)

    for canon, aliases in sorted(alias_groups.items()):
        patch["new_aliases"][canon] = sorted(aliases)

    # For unmapped tags: propose as new canonical entries
    for tag in unmapped:
        patch["new_canonical_tags"][tag] = {
            "proposed_canonical": tag,
            "category": "smarts_detected",
            "occurrences": all_new_tags[tag],
            "agent2_approved": False,
            "note": "New tag from SMARTS enrichment pipeline, needs manual review"
        }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(patch, f, indent=2, ensure_ascii=False)

    print(f"   Saved to: {OUTPUT_PATH}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  New SMARTS tags found:        {len(all_new_tags)}")
    print(f"  Already covered by ontology:  {len(covered)}")
    print(f"  Need alias added:             {len(needs_alias)}")
    print(f"  Completely new (unmapped):     {len(unmapped)}")
    print(f"  Old tags missing from new:    {len(old_only)}")
    print(f"\n  Patch file: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
