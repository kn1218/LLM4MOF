# Phase X: Robust Hypothesis-to-Matchmaker Bridge (Branched Matching)

> **Plan version**: v2 (2026-03-19) -- Updated with verified line numbers post-Phase 1/2/3/7, evidence from 178 experiments, and fixed architectural gaps.

## Executive Summary

The Agent 1 -> Agent 2 -> Matchmaker pipeline has a fundamental expressiveness gap. Agent 1 proposes **branched hypotheses** ("use pyridine dicarboxylate OR ether aromatics OR azolate linkers"), but Agent 2's output schema can only express flat AND or flat OR -- not OR-of-ANDs. This forces Agent 2 to extract the lowest common denominator ("Aromatic"), which matches 52-72% of every database and produces essentially unfiltered results.

**Evidence (from 178 real agent2_output.json files across 65 experiments):**
- `Aromatic` appears in **73% (130/178)** of all Agent 2 extractions -- the #1 tag by 3x
- **39% (69/178)** of extractions use ONLY generic tags (Aromatic, Ring, Heterocycle, etc.)
- **52% (93/178)** of Agent 1 outputs contain explicit OR/alternative linker strategies
- **0/178** outputs have ever used `linker_branches` (feature doesn't exist yet)
- Specific tags like `Pyridine` (2%), `Benzene` (8%), `Azole` (3%) are drastically underused

This is NOT lazy extraction -- it's the CORRECT behavior given the current prompt guidance:
> "If Agent 1 lists alternatives, do NOT add both to this list... pick the one feature that is common to both (e.g., 'Aromatic')."

**Solution**: Add `linker_branches` to Agent 2's output schema -- an array of alternative search branches, each with AND-within-branch semantics, combined with OR-between-branches. This directly maps to the disjunctive normal form (DNF) that Agent 1's hypotheses naturally produce.

---

## Design Principles

1. **Backward-compatible**: Empty `linker_branches` = existing behavior unchanged
2. **Shared utility**: One `check_linker_branches()` function in constraint_utils.py, used by ALL 3 matchmakers + SA
3. **Linker-first**: Node branching deferred -- PORMAKE already handles per-CN separation, and node alternatives are less common
4. **Global exclusions stay global**: `exclude_tags` remains at top level (not per-branch)
5. **Phase 2 features stay global**: `backbone_requirements`, `substituent_requirements`, `min_group_counts` apply post-branch as precision filters
6. **Consistent semantics**: All 3 matchmakers treat branches identically (OR-logic bypass when branches present)

---

## Current Codebase State (Post-Phase 1/2/3/7)

| File | Total Lines | Key Methods |
|---|---|---|
| `core/constraint_utils.py` | 515 | `canon()` L75, `parse_functional_groups()` L148, `check_categorized_groups()` L455 |
| `core/matchmaker.py` | 538 | `_search_linkers()` L243, Phase D Union Logic L390, Phase E Categorized L425 |
| `core/qmof_matchmaker.py` | 213 | `_check_or_tags()` L59, `_check_and_tags()` L52, `match()` L123 |
| `core/hmof_matchmaker.py` | 314 | `_check_or_tags()` L91, `_check_and_tags()` L84, `match()` L113 |
| `core/sensitivity_analyzer.py` | 934 | `_get_linker_list()` L195 |
| `core/agent2_handler.py` | 254 | `_validate_constraints()` L70, `_print_constraints()` L132 |

### Tag Aliasing Verification (unified_ontology.json)

Verified: `canon()` normalizes to lowercase + underscore, then resolves via alias map. Key tags confirmed:
- `Carboxyl` -> `carboxyl` (canonical, no further alias)
- `Carboxylate` -> `carboxylate` (DIFFERENT from `carboxyl` -- NOT aliased)
- `Pyridine` -> `pyridine`, `Benzene` -> `benzene`, `Aromatic` -> `aromatic`
- `Azole` -> `azole`, `Azolate` -> `azolate` (separate canonical tags)

**Implication**: Agent 2 MUST use exact approved vocabulary tags. `check_linker_branches` uses `canon()` for comparison, which handles case normalization but NOT cross-tag aliasing. This is consistent with all existing tag matching.

---

## Schema Change

### Current (broken for alternatives)

```json
"linker_query": {
    "functional_groups": ["Aromatic", "Ring"],
    ...
}
```

### New (expressive)

```json
"linker_query": {
    "functional_groups": [],
    "linker_branches": [
        {
            "description": "N-heteroaromatic dicarboxylate",
            "required_tags": ["Pyridine", "Carboxyl"]
        },
        {
            "description": "Ether-containing aromatic",
            "required_tags": ["Ether", "Aromatic", "Carboxyl"]
        },
        {
            "description": "Azolate/imidazolate",
            "required_tags": ["Azolate"]
        }
    ],
    ...
}
```

### Matching semantics

```
A candidate passes if:
  1. It passes ALL universal filters (connectivity, length, rigidity, negative_tags, abstract_features, Phase 2 filters)
  2. AND it matches AT LEAST ONE branch:
     - Branch match = ALL required_tags are present in the candidate's functional_groups
  3. If linker_branches is empty/absent: fall back to existing functional_groups OR logic
```

### How this fixes real experiments

| Experiment | Agent 1 Said | Current Agent 2 | With Branches |
|---|---|---|---|
| 20260319_1120 iter4 | "naphthalene-based, pyrazine-based, thiophene-containing, alkynyl spacers" | `["Aromatic", "Alkyne"]` | branches: `[["Naphthalene","Carboxyl"], ["Pyrazine"], ["Thiophene","Aromatic"], ["Alkyne","Aromatic"]]` |
| 20260319_1247 iter6 | "azolate N-donor OR sulfonate O-donor OR tfz-d-type" | `[]` (empty!) | branches: `[["Azolate"], ["Sulfonate"], ["Triazole"]]` |
| 20260319_1314 iter5 | "azines/azoles/pyridyl-type backbones" | `["Aromatic", "Heterocycle", "Nitrogen"]` | branches: `[["Azine"], ["Azole"], ["Pyridine"]]` |
| 20260319_1512 iter3 | "Benzene+Cyclohexane" (specific mix) | `["Carboxyl", "Benzene", "Cyclohexane"]` (AND = 0 matches) | branches: `[["Benzene","Carboxyl"], ["Cyclohexane","Carboxyl"]]` |
| 20260318_1455 iter2 | "pyridine, fluoro, nitrile" (alternative decorations) | `["Aromatic","Benzene","Pyridine","Fluoro","Nitrile"]` (AND = 0) | branches: `[["Pyridine","Aromatic"], ["Fluoro","Aromatic"], ["Nitrile","Aromatic"]]` |

---

## Implementation Steps

Each step is atomic and independently testable. **Order matters** -- utility function first, matchmakers second, handler third, prompts last.

---

### STEP X.1: Add `check_linker_branches()` utility

**File**: `core/constraint_utils.py` (515 lines)
**Insert after**: `check_categorized_groups()` which ends at L511
**New function at**: L513+

```python
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
```

**Validation**:
```python
# Test cases (add to bottom of file)
assert check_linker_branches({}, []) == True                              # no branches = pass
assert check_linker_branches({'functional_groups': []}, []) == True       # empty both = pass
assert check_linker_branches(
    {'functional_groups': ['pyridine', 'carboxyl', 'aromatic']},
    [{'required_tags': ['Pyridine', 'Carboxyl']}]
) == True                                                                  # branch 1 matches
assert check_linker_branches(
    {'functional_groups': ['ether', 'aromatic', 'benzene']},
    [{'required_tags': ['Pyridine', 'Carboxyl']}, {'required_tags': ['Ether', 'Aromatic']}]
) == True                                                                  # branch 2 matches
assert check_linker_branches(
    {'functional_groups': ['benzene', 'aromatic']},
    [{'required_tags': ['Pyridine', 'Carboxyl']}, {'required_tags': ['Ether', 'Aromatic']}]
) == False                                                                 # neither branch matches
print("[OK] check_linker_branches all tests passed")
```

---

### STEP X.2: Update PORMake Matchmaker

**File**: `core/matchmaker.py` (538 lines)

#### Change A: Import (L18)

Add `check_linker_branches` to the existing import:
```python
from core.constraint_utils import canon, get_item_features, parse_functional_groups, check_global_requirements, check_negative_tags, get_approved_vocab, check_categorized_groups, check_linker_branches
```

#### Change B: Extract branches before linker loop

In `_search_linkers()` (L243), after `linker_af` extraction (around L265-266), add:
```python
        # Branch matching prep
        linker_branches = specs.get('linker_query', {}).get('linker_branches', [])
```

#### Change C: Add branch check in linker loop

After the abstract features check (L300-302) and BEFORE `linker_ads.append(item['ID'])` (L305), insert:
```python
            # --- BRANCH MATCHING (OR-of-ANDs) ---
            if linker_branches:
                if not check_linker_branches(item, linker_branches):
                    continue
```

#### Change D: Adjust Phase D Union Logic when branches present

At the top of Phase D (after L397 `vocab_set` line), extract branches and set flag:
```python
        linker_branches = specs.get('linker_query', {}).get('linker_branches', [])
        use_branch_mode = bool(linker_branches)
```

In the `check_global_requirements` call (L412-413), when `use_branch_mode` is True, pass empty `linker_or_tags` to skip redundant OR check (branches already filtered linkers in Stage 1):
```python
            for n_id in all_nodes:
                for l_id in linker_ids:
                    # When branches are present, skip OR-tag check (branches already handled it)
                    effective_or_tags = [] if use_branch_mode else linker_or_tags
                    if check_global_requirements(n_id, l_id, global_and_tags, self.bb_lookup,
                                                 linker_or_tags=effective_or_tags):
                        valid_nodes.add(n_id)
                        valid_edges.add(l_id)
```

**Keep** `global_and_tags` check unchanged -- universally required AND tags apply regardless.

**Validation**:
1. `python core/matchmaker.py` -- existing Test 1 (empty AF) and Test 2 (active AF) must pass identically
2. Add Test 3 with branches -- verify branch filtering reduces linker count
3. Add Test 4 with branches + empty functional_groups -- verify Phase D doesn't double-filter

---

### STEP X.3: Update QMOF Matchmaker

**File**: `core/qmof_matchmaker.py` (213 lines)

#### Change A: Import (L7)

Add `check_linker_branches` to the existing import.

#### Change B: Add branch check

**Insert after L177** (end of step 4b OR-tags check) and **before L179** (step 5 categorized check):

```python
                # 4c. Branch matching (OR-of-ANDs)
                linker_branches = specs.get('linker_query', {}).get('linker_branches', [])
                if linker_branches:
                    if not check_linker_branches(qmof, linker_branches):
                        continue
```

**SEMANTICS NOTE (CONSISTENT WITH MATCHMAKER):** When `linker_branches` is present AND `functional_groups` is empty (which is the expected pattern), step 4b `_check_or_tags()` is a no-op (empty OR = pass). When `functional_groups` has universal minimums (e.g., `["Carboxyl"]`), step 4b enforces the floor and step 4c adds branch precision. This is consistent with PORMake's Phase D behavior.

**Optimization**: Hoist `linker_branches` extraction outside the inner loop. Place it after L130 (`global_and_tags, linker_or_tags, neg_tags = ...`):
```python
        linker_branches = specs.get('linker_query', {}).get('linker_branches', [])
```
Then the inner check becomes just:
```python
                if linker_branches:
                    if not check_linker_branches(qmof, linker_branches):
                        continue
```

**Validation**: `python core/qmof_matchmaker.py` -- baseline test must pass. Test with branches -> reduced matches.

---

### STEP X.4: Update hMOF Matchmaker

**File**: `core/hmof_matchmaker.py` (314 lines)

#### Change A: Import (L17-20 area)

Add `check_linker_branches` to the existing import.

#### Change B: Add branch check

**Insert after L179** (end of step 4b OR-tags check) and **before L181** (step 4c categorized check):

```python
                # 4b.5: Branch matching (OR-of-ANDs)
                linker_branches = specs.get('linker_query', {}).get('linker_branches', [])
                if linker_branches:
                    if not check_linker_branches(hmof, linker_branches):
                        continue
```

**Same hoisting optimization as QMOF**: extract `linker_branches` before the loop (after L135-137 where `global_and_tags` is parsed).

**Validation**: `python core/hmof_matchmaker.py` -- baseline test must pass. Test with branches -> reduced matches.

---

### STEP X.5: Update Sensitivity Analyzer

**File**: `core/sensitivity_analyzer.py` (934 lines)

#### Change A: Import

Add `check_linker_branches` to existing constraint_utils import.

#### Change B: Add branch check in `_get_linker_list()` (L195)

After the negative tag check (L243 `if is_banned: continue`) and before `valid_ids.add(item['ID'])` (L247), insert:

```python
            # Branch matching (must be consistent with matchmaker)
            linker_branches = query.get('linker_branches', [])
            if linker_branches:
                if not check_linker_branches(item, linker_branches):
                    continue
```

**Optimization**: Hoist `linker_branches = query.get(...)` before the loop (after L221).

**For QMOF/hMOF modes**: SA delegates to respective matchmaker's `.match()` with `search_mode="linker_only"`. Since we updated those matchmakers in Steps X.3 and X.4, Set G in QMOF/hMOF modes is automatically consistent. No additional SA changes needed for those modes.

**Validation**: Run a PORMake test experiment and verify Set G linker count is consistent with matchmaker filtering.

---

### STEP X.6: Update Agent 2 Handler (Validation + Display)

**File**: `core/agent2_handler.py` (254 lines)

#### Change A: Soft validation

In `_validate_constraints()` (L70), after the Phase 3 electronic metadata validation (around L120), add:

```python
        # Check linker_branches (soft validation -- optional field)
        branches = l_q.get('linker_branches')
        if branches is not None:
            if not isinstance(branches, list):
                print("   [Validation] WARNING: linker_query.linker_branches should be a list, got:", type(branches))
            else:
                for i, branch in enumerate(branches):
                    if not isinstance(branch, dict):
                        print(f"   [Validation] WARNING: linker_branches[{i}] should be a dict, got:", type(branch))
                    elif 'required_tags' not in branch:
                        print(f"   [Validation] WARNING: linker_branches[{i}] missing 'required_tags'")
                    elif not isinstance(branch['required_tags'], list):
                        print(f"   [Validation] WARNING: linker_branches[{i}].required_tags should be a list")
                if len(branches) > 5:
                    print(f"   [Validation] NOTE: {len(branches)} branches detected (>5 may indicate over-decomposition)")
```

#### Change B: Display

In `_print_constraints()` (L132), after the categorized FG display (around L175-180), add:

```python
        # Linker branches (if present)
        branches = linker_q.get('linker_branches', [])
        if branches:
            print(f"Linker Branches ({len(branches)} alternatives):")
            for i, branch in enumerate(branches):
                desc = branch.get('description', '')
                tags = branch.get('required_tags', [])
                print(f"  Branch {i+1}: {desc} -> required: {tags}")
```

**Validation**: `python core/agent2_handler.py` -- existing test passes. Branches display correctly when present.

---

### STEP X.7: Update Agent 2 Prompt (Schema + Extraction Rules)

**File**: `prompts/agent2_v4.0.md` (218 lines)

This is the **highest-risk step** -- the LLM must reliably decompose branched hypotheses.

#### Change A: Add Step 2.7 (Branch Extraction Rules)

Insert AFTER Step 2.6 (categorized FG extraction, ends around L133) and BEFORE Step 3 (geometry extraction):

```markdown
### Step 2.7: Decompose Alternative Linker Strategies into Branches (CRITICAL)

> When Agent 1 proposes **multiple alternative linker strategies** connected by "or", "alternatively", "as a second branch", numbered options, or comma-separated families, you MUST decompose them into `linker_branches`.
>
> **Rules:**
> - Each branch represents ONE alternative linker strategy.
> - `required_tags` within a branch use **AND logic**: ALL tags must be present for a candidate to match that branch.
> - Branches use **OR logic**: a candidate passes if it matches ANY branch.
> - **CRITICAL: Do NOT compute the common denominator.** If Agent 1 says "pyridine dicarboxylate OR azolate linkers", do NOT extract `["Aromatic"]`. Extract two branches: `[["Pyridine", "Carboxyl"], ["Azolate"]]`.
> - `functional_groups` field should contain ONLY tags that are **universally required across ALL branches** (the common minimum). If branches share nothing, set `functional_groups: []`.
> - If Agent 1 describes only ONE linker type (no alternatives), use a SINGLE branch. Still use `linker_branches`.
> - **ALWAYS populate linker_branches** for every extraction. This is now the primary mechanism for linker chemistry matching.
>
> **Examples:**
>
> Agent 1: "Use pyridine dicarboxylate or ether-containing aromatics or azolate linkers"
> ```json
> "functional_groups": [],
> "linker_branches": [
>     {"description": "pyridine dicarboxylate", "required_tags": ["Pyridine", "Carboxyl"]},
>     {"description": "ether aromatic", "required_tags": ["Ether", "Aromatic", "Carboxyl"]},
>     {"description": "azolate", "required_tags": ["Azolate"]}
> ]
> ```
>
> Agent 1: "Rigid aromatic dicarboxylate backbone (BDC or NDC type)"
> ```json
> "functional_groups": ["Carboxyl"],
> "linker_branches": [
>     {"description": "BDC-type", "required_tags": ["Benzene", "Carboxyl"]},
>     {"description": "NDC-type", "required_tags": ["Naphthalene", "Carboxyl"]}
> ]
> ```
>
> Agent 1: "Terphenyl dicarboxylate linker" (single strategy, no alternatives)
> ```json
> "functional_groups": [],
> "linker_branches": [
>     {"description": "terphenyl dicarboxylate", "required_tags": ["Terphenyl", "Carboxyl"]}
> ]
> ```
>
> Agent 1: "naphthalene-based, pyrazine-based, thiophene-containing, and alkynyl spacers"
> ```json
> "functional_groups": [],
> "linker_branches": [
>     {"description": "naphthalene dicarboxylate", "required_tags": ["Naphthalene", "Carboxyl"]},
>     {"description": "pyrazine-based", "required_tags": ["Pyrazine"]},
>     {"description": "thiophene aromatic", "required_tags": ["Thiophene", "Aromatic"]},
>     {"description": "alkynyl spacer", "required_tags": ["Alkyne"]}
> ]
> ```
>
> **ANTI-PATTERN (FORBIDDEN):**
> Agent 1: "Use pyridine OR ether OR azolate"
> WRONG: `"functional_groups": ["Aromatic", "Ring"]` -- This is the common denominator. NEVER DO THIS.
> CORRECT: `"linker_branches": [{"required_tags": ["Pyridine"]}, {"required_tags": ["Ether"]}, {"required_tags": ["Azolate"]}]`
>
> **`functional_groups` semantics reminder:** When `linker_branches` is populated, `functional_groups` typically should be empty `[]` or contain only a truly universal requirement like `["Carboxyl"]` (if ALL branches involve carboxylates). Do NOT put branch-specific tags in `functional_groups`.
```

#### Change B: Update the output JSON schema

In the output format section (around L178-189), update `linker_query`:

```json
  "linker_query": {
      "reasoning": "Explain derivation of length/rigidity here...",
      "connectivity": Integer_or_Null,
      "length_min": Float_or_Null,
      "length_max": Float_or_Null,
      "is_rigid": Boolean_or_Null,
      "functional_groups": ["Universal_Minimum_Tags_or_Empty"],
      "abstract_features": {},
      "backbone_requirements": ["Tag1_or_Null"],
      "substituent_requirements": ["Tag1_or_Null"],
      "min_group_counts": {"tag": Integer_or_Null},
      "linker_branches": [
          {"description": "Branch_Name", "required_tags": ["Tag1", "Tag2"]}
      ]
  },
```

Add note after the schema:
```markdown
> **`linker_branches` format**: Each branch has `description` (short text) and `required_tags` (list of approved vocabulary tags).
> Required tags within a branch use AND logic. Branches use OR logic.
> ALWAYS populate linker_branches -- even for single strategies. Use functional_groups only for universal minimums shared by ALL branches.
```

#### Change C: Update the Union Logic warning (L36-40)

Replace the existing "WARNING (Mutually Exclusive Options)" section with:

```markdown
*   **Global Functional Groups (Union Logic):** This field is now the **universal minimum floor**.
    *   Only list groups that MUST be present in ALL candidates across ALL branches.
    *   If `linker_branches` is populated (which it should always be), `functional_groups` should typically be `[]` or contain only truly universal tags like `["Carboxyl"]`.
    *   **DO NOT** use `functional_groups` to express alternatives. Use `linker_branches` instead.
```

**Validation**:
1. Read the modified prompt -- verify JSON syntax is valid
2. Test Agent 2 with a branched hypothesis -- verify it outputs branches instead of `["Aromatic"]`
3. Test Agent 2 with a single-strategy hypothesis -- verify it outputs a single branch
4. Test Agent 2 with no specific linker chemistry -- verify it outputs empty branches

---

### STEP X.8: Update Agent 1 Prompt (Recommended)

**File**: `prompts/agent1_v2.2.9.md`

In the Components section (after `functional_groups` bullet, around L36), add:

```markdown
    * **Alternative Strategies:** You may propose multiple linker strategies (e.g., "Use pyridine dicarboxylate OR ether-containing aromatics"). Each alternative will be searched independently -- be specific with functional group names rather than generic categories like "aromatic". Using "benzene dicarboxylate OR naphthalene dicarboxylate" is far more effective than "aromatic linker".
```

**Validation**: Read the modified prompt -- verify formatting is clean.

---

### STEP X.9: (SEPARATE PHASE -- DO NOT BUNDLE)

> **IMPORTANT**: Step X.9 (removing Basic Elements from FG vocabulary) should be done as a SEPARATE change AFTER Phase X core is validated. Reason: 7% of current outputs use `Nitrogen` as a FG tag. Removing it changes behavior for those experiments. This needs its own validation cycle.

**Deferred to Phase X.9-standalone.**

---

## Files Modified (Summary)

| File | Change | Risk | Lines Affected |
|---|---|---|---|
| `core/constraint_utils.py` | New `check_linker_branches()` function | Low (additive) | After L511 |
| `core/matchmaker.py` | Branch check in `_search_linkers()` L243 + Phase D L390 adjustment | Medium | L18, L265, L302, L397-415 |
| `core/qmof_matchmaker.py` | Branch check after OR tags L177 | Low | L7, L130, L177 |
| `core/hmof_matchmaker.py` | Branch check after OR tags L179 | Low | L17, L135, L179 |
| `core/sensitivity_analyzer.py` | Branch check in `_get_linker_list()` L195 | Low | Import, L221, L243 |
| `core/agent2_handler.py` | Soft validation L120 + display L175 | Low | L120, L175 |
| `prompts/agent2_v4.0.md` | Step 2.7 + schema + Union Logic update | High (LLM behavior) | After L133, L178-189, L36-40 |
| `prompts/agent1_v2.2.9.md` | Alternative strategies note | Zero | L36 |

## Files NOT Modified

| File | Reason |
|---|---|
| `core/feedback_generator.py` | Per-branch feedback deferred to Phase X+ |
| `run_experiment.py` | Generic orchestration -- branches flow through automatically |
| `config.py` | No new configuration needed |
| `data/*.json` | Read-only data files -- never modified |

---

## End-to-End Validation

### Test A: Backward Compatibility (no branches)
```
python core/matchmaker.py     # existing Test 1+2 -- must pass identically
python core/qmof_matchmaker.py # existing test -- must pass identically
python core/hmof_matchmaker.py # existing test -- must pass identically
```

### Test B: Branch Matching Unit Tests
Run the assertion-based tests from Step X.1 in `constraint_utils.py`.

### Test C: PORMake with branches
Run matchmaker with branch specs and verify:
- Result count < unfiltered baseline (branches filter)
- Result count > strict AND of all branch tags combined (OR-of-ANDs is broader than AND-of-all)
- Individual branches produce disjoint or overlapping candidate sets (both valid)

### Test D: Full Experiment -- Bandgap 3-4 eV
Re-run with electronic metadata enabled.
**Expected**: Agent 2 outputs branches, matchmaker returns targeted results (not 0, not 70% of DB).

### Test E: Full Experiment -- H2 Storage (PORMake)
**Expected**: Agent 2 outputs specific branches instead of `["Aromatic"]`. Matchmaker returns focused candidates.

### Test F: Full Experiment -- CH4 Storage (hMOF)
**Expected**: Same improvement in filtering specificity.

### Test G: Full Experiment -- Xe/Kr Selectivity (hMOF)
**Expected**: Agent 2 decomposes pore-lining group alternatives into branches.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Agent 2 LLM still outputs `["Aromatic"]` instead of branches | **HIGH** (73% of 178 outputs did this) | High | Multiple anti-pattern examples + "ALWAYS populate linker_branches" rule + 4 worked examples. If persistent after deployment, add few-shot examples in prompt. |
| Agent 2 creates too many branches (>5) | Low | Low | Matchmaker handles any number. Soft warning in handler if >5. |
| Branch required_tags use wrong vocabulary | Medium | Medium | Approved vocabulary enforced by prompt. Soft validation in handler catches malformed tags. |
| Empty branches (required_tags: []) | Low | Zero | `check_linker_branches` skips empty branches. |
| Phase D Union Logic interaction | Medium | Medium | `effective_or_tags = []` when branches present -- skip redundant OR check. Thoroughly tested. |
| SA inconsistency with matchmaker | Low | Medium | Shared `check_linker_branches` function ensures identical logic. QMOF/hMOF SA modes delegate to matchmaker automatically. |
| Tag aliasing (Carboxyl vs Carboxylate) | Low | Low | Verified: canon() does NOT alias these. Agent 2 MUST use approved vocab. Prompt already enforces this. |

---

## Execution Order

```
WAVE 1 (Code -- can be tested with manual specs):
  X.1 (constraint_utils) -> X.2 (matchmaker) -> X.3 (QMOF) -> X.4 (hMOF) -> X.5 (SA) -> X.6 (handler)
  Validate: Test A + B + C

WAVE 2 (Prompts -- requires live LLM testing):
  X.7 (Agent 2 prompt) -> X.8 (Agent 1 prompt)
  Validate: Test D + E + F + G

WAVE 3 (Separate -- after Phase X core validated):
  X.9-standalone (vocabulary cleanup)
  Validate: Regression on all modes
```

Total estimated effort: 2-3 sessions (Wave 1: 1 session, Wave 2: 1 session, Wave 3: 0.5 session).
