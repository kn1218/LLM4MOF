# =============================================================================
# LLM4MOF Autonomous System - Shared Constraint Utilities
# =============================================================================
# Function & Purpose:
# This module acts as the central hub for parsing and enforcing Agent 2's chemical constraints.
# It ensures that both the Matchmaker (which picks the MOFs) and the SensitivityAnalyzer 
# (which creates the statistical reports) evaluate the LLM tags exactly the same way.
# 
# The functions here (`parse_functional_groups`, `check_global_requirements`, etc.) actively
# read lists like ["Aromatic", "Nitrogen"] and intersect them against the tags of the actual
# MOF building blocks in the dataset. This successfully filters down the 12,000+ theoretical 
# combinations to a much smaller, chemically valid subset (often reducing the space by 95%+).
# Without this file, Agent 2's functional group and ligand definitions would be ignored.
# =============================================================================

import re
import json
import os
from typing import Set, Dict, List, Tuple, Optional, Any


# =============================================================================
# PORMAKE COORDINATION TAGS
# =============================================================================
# In PorMake's building-block grammar, coordination groups (the atoms that bond
# the organic linker to the metal node) live on the **Node SBU**, not on the
# Edge (linker backbone). When Agent 2 describes a "biphenyl dicarboxylate"
# linker, the "dicarboxylate" part is captured by the node; the edge only
# carries the "biphenyl" backbone.
#
# These tags should be stripped from linker-branch matching in PorMake mode
# because edges will almost never carry them (only 10/219 anomalous edges do).
# For hMOF/QMOF (whole-MOF filtering), these tags ARE valid and must NOT be
# stripped — whole MOFs legitimately carry carboxyl, azolate, etc.
# =============================================================================

PORMAKE_COORDINATION_TAGS: frozenset = frozenset({
    "carboxyl",         # covers carboxyl_any, carboxylate, carboxylic_acid via canon()
    "carbonyl",         # C=O often part of carboxylate coordination shell
    "phosphonate",      # P-O donor coordination
    "sulfonate",        # S-O donor coordination (rare but node-dominant)
})


# =============================================================================
# ONTOLOGY LOADING (Singleton)
# =============================================================================

_alias_map: Optional[Dict[str, str]] = None
_approved_vocab: Optional[Set[str]] = None


def _load_ontology() -> None:
    """Load unified ontology and build alias→canonical mapping (once)."""
    global _alias_map, _approved_vocab

    # Import here to avoid circular dependency at module level
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import UNIFIED_ONTOLOGY_PATH

    _alias_map = {}
    _approved_vocab = set()

    if not os.path.exists(UNIFIED_ONTOLOGY_PATH):
        print(f"[constraint_utils] WARNING: Ontology not found at {UNIFIED_ONTOLOGY_PATH}")
        return

    with open(UNIFIED_ONTOLOGY_PATH, 'r', encoding='utf-8') as f:
        ontology = json.load(f)

    for canonical_tag, info in ontology.get('canonical_tags', {}).items():
        canonical_lower = canonical_tag.lower().strip().replace('-', '_').replace(' ', '_')
        _approved_vocab.add(canonical_lower)

        # Map each alias to the canonical form
        for alias in info.get('aliases', []):
            alias_lower = alias.lower().strip().replace('-', '_').replace(' ', '_')
            _alias_map[alias_lower] = canonical_lower

        # Also map the canonical tag itself (in case of case differences)
        _alias_map[canonical_lower] = canonical_lower

    print(f"[constraint_utils] Ontology loaded: {len(_approved_vocab)} canonical tags, {len(_alias_map)} alias mappings")


def get_approved_vocab() -> Set[str]:
    """Return the set of approved canonical vocabulary tags."""
    if _approved_vocab is None:
        _load_ontology()
    return _approved_vocab


# =============================================================================
# CORE NORMALIZATION
# =============================================================================

def canon(s: str) -> str:
    """
    Canonicalize a tag/feature string for consistent matching.

    Transformations:
    1. lowercase, strip whitespace, hyphens/spaces → underscores
    2. Resolve synonyms via unified_ontology.json alias mapping

    Examples:
        'Primary Amine' → 'primary_amine'
        'Aromatic_Ring' → 'aromatic'  (alias resolved)
        'Aromatic' → 'aromatic'
    """
    global _alias_map
    if not s:
        return ''

    normalized = s.lower().strip().replace('-', '_').replace(' ', '_')

    # Lazy-load ontology on first use
    if _alias_map is None:
        _load_ontology()

    # Resolve alias if known, otherwise return normalized form
    return _alias_map.get(normalized, normalized)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def get_item_features(item: dict, include_elements: bool = True) -> Set[str]:
    """
    Extract chemical features from a BB item.
    
    Args:
        item: BB dictionary item (Node or Edge)
        include_elements: If True, include elements from formula (for matching).
                         If False, only return allowed tags (for validation).
    
    Returns:
        Set of canonicalized feature strings
    """
    features = set()
    
    # 1. Functional Groups (allowed tags)
    for g in item.get('functional_groups', []):
        features.add(canon(g))
    
    # 2. Ligand/Connection Chemistry (allowed tags)
    chem = item.get('ligand_chemistry', item.get('connection_chemistry', []))
    if chem is None:
        chem = []
    elif isinstance(chem, str):
        chem = [chem]
    for c in chem:
        features.add(canon(c))
    
    # 3. Elements from Formula (internal features only)
    if include_elements:
        formula = item.get('formula', '')
        if formula:
            elements = re.findall(r'([A-Z][a-z]?)', formula)
            for e in elements:
                features.add(e.lower())
    
    return features


# =============================================================================
# TAG PARSING
# =============================================================================

def parse_functional_groups(specs: dict, approved_vocab: Optional[Set[str]] = None,
                            tracker: Any = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Parse constraints for positive and negative functional group tags.

    Separates tags into three groups with different matching semantics:
    1. global_and_tags  (from global_requirements.include_tags) → AND logic
       Every (node, edge) pair must satisfy ALL of these.
    2. linker_or_tags   (from linker_query.functional_groups)   → OR logic
       At least ONE of these must be present in the (node, edge) pair.
    3. negative_tags    (from global_requirements.exclude_tags)  → Bouncer
       Any item containing these is excluded.

    Args:
        specs: Agent2 specifications dict
        approved_vocab: Optional set of approved tags for validation
        tracker: Optional ProvenanceTracker to record unrecognized tags

    Returns:
        (global_and_tags, linker_or_tags, negative_tags) tuple - all CANONICALIZED

    Backward-compat: callers that unpack only 2 values will get
        (global_and_tags, negative_tags) — see _parse_negative_only() helper.
    """
    global_and_tags = []
    linker_or_tags = []
    negative_tags = []

    seen_global = set()
    seen_linker = set()
    seen_negative = set()

    # --- 1. global_requirements ---
    global_reqs = specs.get('global_requirements', {})
    for raw_tag in global_reqs.get('include_tags', []):
        clean = raw_tag.strip()
        if not clean:
            continue
        canonicalized = canon(clean)
        if canonicalized in seen_global:
            continue
        seen_global.add(canonicalized)
        if approved_vocab is not None and canonicalized not in approved_vocab:
            if tracker:
                tracker.unrecognized_tags.append({"tag": raw_tag, "type": "positive"})
            print(f"  ⚠️  UNRECOGNIZED GLOBAL TAG: '{raw_tag}'")
        global_and_tags.append(canonicalized)

    for raw_tag in global_reqs.get('exclude_tags', []):
        clean = raw_tag.strip()
        if not clean:
            continue
        canonicalized = canon(clean)
        if canonicalized in seen_negative:
            continue
        seen_negative.add(canonicalized)
        if approved_vocab is not None and canonicalized not in approved_vocab:
            if tracker:
                tracker.unrecognized_tags.append({"tag": raw_tag, "type": "negative"})
            print(f"  ⚠️  UNRECOGNIZED NEGATIVE TAG: '{raw_tag}'")
            print(f"      This constraint will be IGNORED (no matching possible).")
            continue
        negative_tags.append(canonicalized)

    # --- 2. linker_query.functional_groups (OR semantics) ---
    linker_tags = specs.get('linker_query', {}).get('functional_groups', [])
    for tag in linker_tags:
        t = tag.strip()
        if not t:
            continue
        # Defensive: intercept accidental "avoid_" or "avoid " prefixes from Agent 2
        t_lower = t.lower()
        if t_lower.startswith('avoid_') or t_lower.startswith('avoid '):
            clean_tag = t[6:].strip()
            if clean_tag:
                print(f"  [DEFENSIVE] Found 'avoid' prefix in functional_groups: '{t}'")
                print(f"  [DEFENSIVE] Redirecting to negative constraint: '{clean_tag}'")
                canonicalized = canon(clean_tag)
                if canonicalized not in seen_negative:
                    seen_negative.add(canonicalized)
                    if approved_vocab is not None and canonicalized not in approved_vocab:
                        continue
                    negative_tags.append(canonicalized)
            continue
        canonicalized = canon(t)
        if canonicalized in seen_linker:
            continue
        seen_linker.add(canonicalized)
        if approved_vocab is not None and canonicalized not in approved_vocab:
            if tracker:
                tracker.unrecognized_tags.append({"tag": t, "type": "positive"})
            print(f"  ⚠️  UNRECOGNIZED LINKER TAG: '{t}'")
        linker_or_tags.append(canonicalized)

    # Store requested tags in tracker for zero-hit diagnostics
    if tracker:
        tracker.requested_tags = global_and_tags + linker_or_tags

    return global_and_tags, linker_or_tags, negative_tags


# =============================================================================
# UNION LOGIC CHECK
# =============================================================================

def check_global_requirements(node_id: str, edge_id: str,
                               global_and_tags: List[str],
                               bb_lookup: Dict[str, dict],
                               linker_or_tags: Optional[List[str]] = None,
                               tracker: Any = None) -> bool:
    """
    UNION LOGIC CHECK: Validates that a (node, edge) pair satisfies tag constraints.

    Two-tier match logic:
      1. global_and_tags (AND): ALL must be present in (node ∪ edge) features.
      2. linker_or_tags  (OR):  At least ONE must be present (if list is non-empty).

    Feature match: tag ∈ (node.features ∪ edge.features)  (exact, canonicalized)

    Args:
        node_id: Node BB ID
        edge_id: Edge BB ID
        global_and_tags: Tags that ALL must match (from global_requirements.include_tags)
        bb_lookup: Dictionary mapping ID → BB item
        linker_or_tags: Tags where ANY one match suffices (from linker_query.functional_groups)
        tracker: Optional ProvenanceTracker for recording satisfaction

    Returns:
        True if constraints are satisfied, False otherwise
    """
    if not global_and_tags and not linker_or_tags:
        return True

    node = bb_lookup.get(node_id, {})
    edge = bb_lookup.get(edge_id, {})

    # Get features
    node_feats = get_item_features(node, include_elements=True)
    edge_feats = get_item_features(edge, include_elements=True)
    combined_feats = node_feats | edge_feats

    # Track provenance for each tag (only recorded if pair passes ALL checks)
    tag_provenance = []

    # --- AND check: every global_and_tag must be present ---
    for tag in (global_and_tags or []):
        in_node = tag in node_feats
        in_edge = tag in edge_feats

        if not (in_node or in_edge):
            if tracker:
                tracker.record_first_fail(tag)
            return False

        match_type = "exact_feature"
        if in_node and in_edge:
            satisfied_by = "both"
        elif in_node:
            satisfied_by = "node_only"
        else:
            satisfied_by = "edge_only"
        tag_provenance.append((tag, satisfied_by, match_type, None))

    # --- OR check: at least one linker_or_tag must be present ---
    if linker_or_tags:
        or_hit = False
        or_provenance = None
        for tag in linker_or_tags:
            in_node = tag in node_feats
            in_edge = tag in edge_feats
            if in_node or in_edge:
                or_hit = True
                match_type = "exact_feature"
                if in_node and in_edge:
                    satisfied_by = "both"
                elif in_node:
                    satisfied_by = "node_only"
                else:
                    satisfied_by = "edge_only"
                or_provenance = (tag, satisfied_by, match_type, None)
                break  # one hit is enough

        if not or_hit:
            if tracker:
                tracker.record_first_fail(f"OR({','.join(linker_or_tags)})")
            return False

        if or_provenance:
            tag_provenance.append(or_provenance)

    # All checks passed - record provenance if tracker provided
    if tracker:
        for (t, satisfied_by, match_type, _) in tag_provenance:
            tracker.record(t, satisfied_by, match_type, node_id, edge_id, None)

    return True


# =============================================================================
# NEGATIVE TAG CHECK
# =============================================================================

def check_negative_tags(item: dict, negative_tags: List[str]) -> bool:
    """
    Check if an item contains any forbidden (negative) tags.
    Used as a "bouncer" to filter out items before Union Logic.
    
    Args:
        item: BB item (Node or Edge)
        negative_tags: List of canonicalized forbidden tags
    
    Returns:
        True if item is CLEAN (no forbidden tags), False if BANNED
    """
    if not negative_tags:
        return True
    
    # Get item features (without elements - just functional groups and chemistry)
    item_features = get_item_features(item, include_elements=False)
    
    # Also check readable_name
    name_canon = canon(item.get('readable_name', ''))
    
    for neg in negative_tags:
        # Check in features
        if neg in item_features:
            return False
        
        # Check in name (handle underscore vs space)
        if neg in name_canon or neg.replace('_', ' ') in name_canon:
            return False
    
    return True


# =============================================================================
# TEST FUNCTION
# =============================================================================

def _test_constraint_utils():
    """Run basic tests for constraint utilities."""
    print("\n" + "=" * 60)
    print("CONSTRAINT UTILS MODULE TEST")
    print("=" * 60 + "\n")
    
    # Test canon() — now includes alias resolution via unified_ontology.json
    print("Testing canon()...")
    assert canon("Primary Amine") == "primary_amine"
    assert canon("N-rich") == "n_rich"  # no alias in ontology, stays normalized
    assert canon("  Aromatic  ") == "aromatic"
    assert canon("carboxylic-acid") == "carboxyl"  # alias: carboxylic_acid → carboxyl
    assert canon("Aromatic_Ring") == "aromatic"  # alias resolved
    assert canon("Pyridine_Ring") == "pyridine"  # alias resolved
    assert canon("triazole_124") == "triazole"   # variant alias resolved
    print("OK canon() tests passed")
    
    # Test get_item_features()
    print("\nTesting get_item_features()...")
    test_item = {
        "functional_groups": ["Aromatic", "Ring"],
        "ligand_chemistry": ["Oxygen"],
        "formula": "C6H4O2"
    }
    feats = get_item_features(test_item, include_elements=True)
    assert "aromatic" in feats
    assert "ring" in feats
    assert "oxygen" in feats
    assert "c" in feats
    assert "h" in feats
    assert "o" in feats
    print(f"OK Features extracted: {feats}")
    
    feats_no_elem = get_item_features(test_item, include_elements=False)
    assert "c" not in feats_no_elem
    print(f"OK Features (no elements): {feats_no_elem}")
    
    # Test parse_functional_groups()
    print("\nTesting parse_functional_groups()...")
    test_specs = {
        "linker_query": {
            "functional_groups": ["Aromatic", "avoid Halogen", "Nitrogen"]
        },
        "global_requirements": {
            "include_tags": ["Ring"],
            "exclude_tags": ["Fluoro"]
        }
    }
    global_and, linker_or, neg = parse_functional_groups(test_specs)
    assert "ring" in global_and, f"Expected 'ring' in global_and_tags, got {global_and}"
    assert "aromatic" in linker_or, f"Expected 'aromatic' in linker_or_tags, got {linker_or}"
    assert "nitrogen" in linker_or, f"Expected 'nitrogen' in linker_or_tags, got {linker_or}"
    assert "fluoro" in neg
    assert "halogen" in neg  # redirected from 'avoid Halogen'

    print(f"OK Global AND tags: {global_and}")
    print(f"OK Linker OR tags: {linker_or}")
    print(f"OK Negatives: {neg}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60 + "\n")


# =============================================================================
# CATEGORIZED FUNCTIONAL GROUP CHECK
# =============================================================================

def check_categorized_groups(
    item: dict,
    backbone_reqs: Optional[List[str]] = None,
    substituent_reqs: Optional[List[str]] = None,
    min_counts: Optional[Dict[str, int]] = None
) -> bool:
    """
    Check if item satisfies categorized functional group requirements.

    Uses item['functional_groups_categorized'] which has:
      backbone:           list of backbone scaffold tags
      substituents:       list of substituent tags
      rule_based_counts:  dict of {tag: count}

    All comparisons are canonicalized via canon().

    Args:
        item: BB dict or MOF index dict with optional
              'functional_groups_categorized' key.
        backbone_reqs: Tags that MUST appear in item's backbone (AND logic).
        substituent_reqs: Tags that MUST appear in item's substituents (AND logic).
        min_counts: Dict of {tag: minimum_count} checked against
                    rule_based_counts.

    Returns:
        True  if ALL specified requirements are met.
        True  if item has no categorized data (benefit of doubt).
        False if any requirement is violated.
    """
    cat = item.get('functional_groups_categorized')
    if not cat:
        return True  # No categorized data → don't filter out

    # 1. Backbone requirements (AND logic — ALL must be present)
    if backbone_reqs:
        item_backbone = {canon(t) for t in cat.get('backbone', [])}
        for req in backbone_reqs:
            if canon(req) not in item_backbone:
                return False

    # 2. Substituent requirements (AND logic — ALL must be present)
    if substituent_reqs:
        item_subs = {canon(t) for t in cat.get('substituents', [])}
        for req in substituent_reqs:
            if canon(req) not in item_subs:
                return False

    # 3. Minimum group counts
    if min_counts:
        item_counts = cat.get('rule_based_counts', {})
        # Canonicalize keys for comparison
        canon_counts = {canon(k): v for k, v in item_counts.items()}
        for tag, min_val in min_counts.items():
            if canon_counts.get(canon(tag), 0) < min_val:
                return False

    return True


def check_linker_branches(item: dict, branches: list) -> bool:
    """
    OR-of-ANDs branch matching for linker functional groups.
    
    Each branch has 'required_tags' (AND within branch).
    Candidate passes if ANY branch is fully satisfied.
    
    Args:
        item: BB or MOF dict with 'functional_groups' key
        branches: List of dicts, each with 'required_tags' (list of strings)
        
    Returns:
        True if no branches (passthrough), or if any branch matches.
    """
    if not branches:
        return True  # No branches = no filter (backward compat)
    
    item_tags = {canon(t) for t in item.get('functional_groups', [])}
    
    for branch in branches:
        required = branch.get('required_tags', [])
        if not required:
            continue  # Empty branch = skip
        required_canon = {canon(t) for t in required}
        if required_canon.issubset(item_tags):
            return True  # ALL tags in this branch are present -> match
    
    return False  # No branch matched


def strip_coordination_tags_from_branches(
    branches: list,
    coordination_tags: frozenset = PORMAKE_COORDINATION_TAGS
) -> list:
    """
    Strip coordination-group tags from linker branch required_tags for PorMake mode.

    In PorMake's BB grammar, coordination groups (carboxylate, phosphonate, etc.)
    live on the Node SBU, not on the Edge. Agent 2 may include them in branch
    required_tags (e.g., ["Biphenyl", "Carboxyl"]) because that's natural chemistry
    language. This function removes them so branch matching works against edges.

    Args:
        branches: List of branch dicts with 'required_tags' and 'description'.
        coordination_tags: Set of canonical coordination tags to strip.

    Returns:
        New list of branches with coordination tags removed.
        Empty branches (all tags stripped) are preserved with a warning flag.
    """
    cleaned = []
    for branch in branches:
        required = branch.get('required_tags', [])
        original_count = len(required)

        kept = [t for t in required if canon(t) not in coordination_tags]
        stripped = [t for t in required if canon(t) in coordination_tags]

        if stripped:
            print(f"   [PorMake] Branch '{branch.get('description', '?')}': "
                  f"stripped coordination tags {stripped} (node-side in PorMake)")

        if kept:
            cleaned.append({
                "description": branch.get("description", ""),
                "required_tags": kept
            })
        else:
            # All tags were coordination-only → branch becomes empty → skip it
            print(f"   [PorMake] Branch '{branch.get('description', '?')}': "
                  f"all tags were coordination-only, branch removed")

    return cleaned


def _test_strip_coordination_tags():
    """Unit tests for strip_coordination_tags_from_branches."""
    # Basic stripping
    branches = [
        {"description": "biphenyl dicarboxylate", "required_tags": ["Biphenyl", "Carboxyl"]},
        {"description": "pyridine linker", "required_tags": ["Pyridine"]},
    ]
    result = strip_coordination_tags_from_branches(branches)
    assert len(result) == 2
    assert result[0]["required_tags"] == ["Biphenyl"]
    assert result[1]["required_tags"] == ["Pyridine"]

    # Branch with only coordination tags → removed
    branches2 = [
        {"description": "carboxylate only", "required_tags": ["Carboxyl", "Carbonyl"]},
        {"description": "amine backbone", "required_tags": ["Amine"]},
    ]
    result2 = strip_coordination_tags_from_branches(branches2)
    assert len(result2) == 1
    assert result2[0]["required_tags"] == ["Amine"]

    # Empty branches input
    assert strip_coordination_tags_from_branches([]) == []

    # No coordination tags → unchanged
    branches3 = [{"description": "test", "required_tags": ["Benzene", "Amine"]}]
    result3 = strip_coordination_tags_from_branches(branches3)
    assert result3[0]["required_tags"] == ["Benzene", "Amine"]

    print("[OK] strip_coordination_tags_from_branches all tests passed")


def _test_linker_branches():
    """Unit tests for check_linker_branches."""
    # No branches = pass
    assert check_linker_branches({}, []) == True
    assert check_linker_branches({'functional_groups': []}, []) == True
    
    # Single branch matches
    assert check_linker_branches(
        {'functional_groups': ['pyridine', 'carboxyl', 'aromatic']},
        [{'required_tags': ['Pyridine', 'Carboxyl']}]
    ) == True
    
    # Second branch matches (first doesn't)
    assert check_linker_branches(
        {'functional_groups': ['ether', 'aromatic', 'benzene']},
        [{'required_tags': ['Pyridine', 'Carboxyl']}, {'required_tags': ['Ether', 'Aromatic']}]
    ) == True
    
    # Neither branch matches
    assert check_linker_branches(
        {'functional_groups': ['benzene', 'aromatic']},
        [{'required_tags': ['Pyridine', 'Carboxyl']}, {'required_tags': ['Ether', 'Aromatic']}]
    ) == False
    
    # Empty branch skipped, other branch matches
    assert check_linker_branches(
        {'functional_groups': ['azolate']},
        [{'required_tags': []}, {'required_tags': ['Azolate']}]
    ) == True
    
    # Item has no functional_groups
    assert check_linker_branches(
        {},
        [{'required_tags': ['Benzene']}]
    ) == False
    
    print("[OK] check_linker_branches all tests passed")


if __name__ == "__main__":
    _test_constraint_utils()
    _test_linker_branches()
    _test_strip_coordination_tags()
