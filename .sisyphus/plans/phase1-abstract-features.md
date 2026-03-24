# Phase 1: `abstract_features` Integration Plan

## Executive Summary

The PORMake building block dictionary (`data/pormake_bb_dictionary_v5.json`) contains 11 boolean chemical properties per building block under the `abstract_features` key. These properties (e.g., `is_conjugated`, `has_open_metal_site`, `has_hydrogen_bond_donor`) map directly to the qualitative chemical reasoning Agent 1 already performs — but the pipeline currently ignores them entirely.

This plan adds abstract_features support across the full pipeline: Agent 1 prompt → Agent 2 extraction → Matchmaker filtering → Feedback display. Every step is backward-compatible and independently testable.

**Scope**: PORMake mode only. QMOF and hMOF matchmakers are unaffected (their data doesn't have abstract_features).

---

## Data Context (READ THIS FIRST)

### Building Block Inventory
- **867 total BBs**: 648 Nodes + 219 Edges
- **100% have abstract_features** — zero missing data
- **Only `has_open_metal_site` has nulls**: 114 Node nulls (17.6%), 219 Edge nulls (100% — edges never have open metal sites)

### Distribution Table (Critical for understanding filter effectiveness)

```
Feature                        | Node T Node F Node N | Edge T Edge F Edge N
----------------------------------------------------------------------------------------------------
is_fluorinated                 |      1    647      0 |      6    213      0
is_electron_deficient          |     14    634      0 |      6    213      0
is_electron_rich               |     26    622      0 |     26    193      0
is_symmetric                   |     95    553      0 |    215      4      0
is_conjugated                  |    605     43      0 |    177     42      0
is_metalated                   |    534    114      0 |     19    200      0
has_hydrogen_bond_donor        |    221    427      0 |     23    196      0
has_hydrogen_bond_acceptor     |    578     70      0 |    124     95      0
is_charged                     |      0    648      0 |      0    219      0
is_photoswitchable             |      2    646      0 |      4    215      0
has_open_metal_site            |    233    301    114 |      0      0    219
```

T=True, F=False, N=Null.

### What This Means for Filtering
- **High discriminative power** (20-80% split): `has_hydrogen_bond_donor` (34%/11%), `has_open_metal_site` (36%/N/A), `has_hydrogen_bond_acceptor` (57% edge split)
- **Moderate**: `is_metalated` (82%/9%), `is_conjugated` (93%/81%), `is_symmetric` (15%/98%)
- **Poor** (<5% true): `is_charged` (0%), `is_fluorinated` (<3%), `is_photoswitchable` (<2%), `is_electron_deficient` (<3%)
- All 11 features should be exposed (for completeness and future-proofing), but the prompt should warn about scarce features.

---

## Architecture: Data Flow

```
Agent 1 (prompt mentions abstract features in node/linker composition text)
    |
    v
Agent 2 (extracts abstract_features into node_query + linker_query)
    |
    v  constraints dict:
    |  {
    |    "node_query": { ..., "abstract_features": {"has_open_metal_site": true} },
    |    "linker_query": { ..., "abstract_features": {"is_conjugated": true} },
    |    "geometry_filter": { ... },
    |    "global_requirements": { ... }
    |  }
    |
    v
Matchmaker._search_nodes()  --> checks node_query.abstract_features
Matchmaker._search_linkers() --> checks linker_query.abstract_features
    |
    v
Results flow to SensitivityAnalyzer and FeedbackGenerator (no changes needed there for Phase 1)
```

**Key design decisions:**
1. `abstract_features` lives INSIDE `node_query` and `linker_query` (not as a new top-level key) because nodes and linkers have different relevant features
2. Only non-null features filter. Missing key = null = no filter.
3. `null` in the BB data (e.g., `has_open_metal_site: null` on edges) means "unknown" and should NOT be excluded by a `true` filter — only explicit `false` excludes.

---

## Steps

Each step is atomic and independently testable. Complete one, validate, then proceed to the next.

---

### STEP 1.1: Update Agent 1 Prompt

**File**: `prompts/agent1_v2.2.9.md`

**Goal**: Tell Agent 1 that building block properties exist so it can mention them in its hypothesis text.

**What to change**: Add a new bullet point under the existing `Components` section.

**CURRENT** (lines 29-33):
```markdown
* **Components (The Cause - Your Final Choice):**
    * `node_metal` (e.g., Symbol_A, Symbol_B)
    * `node_connectivity` (integer values)
    * `linker_length` (Å; provide min/max bounds)
    * `functional_groups` (names of functional groups)
```

**REPLACE WITH**:
```markdown
* **Components (The Cause - Your Final Choice):**
    * `node_metal` (e.g., Symbol_A, Symbol_B)
    * `node_connectivity` (integer values)
    * `linker_length` (Å; provide min/max bounds)
    * `functional_groups` (names of functional groups)
    * `building_block_properties` (optional boolean filters for PORMAKE building blocks):
        * **Node-relevant**: `has_open_metal_site` (coordinatively unsaturated metal; critical for strong gas binding and catalysis), `is_metalated` (contains metal center), `is_conjugated` (extended pi-system), `has_hydrogen_bond_donor` (N-H/O-H groups), `has_hydrogen_bond_acceptor` (lone-pair N/O atoms), `is_symmetric`, `is_electron_rich`, `is_electron_deficient`
        * **Linker-relevant**: `is_conjugated` (for electronic delocalization and bandgap tuning), `has_hydrogen_bond_donor`/`has_hydrogen_bond_acceptor` (for selective guest binding, CO2 capture), `is_symmetric` (for regular pore geometry), `is_electron_rich`/`is_electron_deficient` (for electronic modulation), `is_fluorinated` (for hydrophobicity and stability)
        * **SCARCE FEATURES WARNING**: `is_fluorinated` (<3%), `is_electron_deficient` (<3%), `is_charged` (0%), `is_photoswitchable` (<2%) have very low availability in the database. Requiring them as `true` may yield zero candidates. Prefer using them as `false` (avoidance) filters.
        * Specify ONLY properties critical to your mechanism in the `node_composition` or `linker_composition` text. Unmentioned properties will not be filtered.
```

**Agent 1 OUTPUT format**: NO CHANGE. Agent 1's JSON schema stays the same. It mentions building block properties naturally in `node_composition` and `linker_composition` text fields. Examples:
- `"node_composition": "12-connected Zr6 cluster with open metal sites for strong H2 binding..."`
- `"linker_composition": "Rigid, conjugated terphenyl dicarboxylate with hydrogen bond acceptor groups..."`

**Validation**:
1. Read the modified prompt file — verify the new section is syntactically clean and doesn't break surrounding markdown
2. Verify the `{SCIENTIFIC_JOURNAL}` placeholder (line 74) is still present and unmodified
3. Verify the JSON output format section (lines 82-96) is completely unchanged

---

### STEP 1.2: Update Agent 2 Prompt

**File**: `prompts/agent2_v4.0.md`

**Goal**: Tell Agent 2 how to extract building_block_properties from Agent 1's text into structured JSON.

**Two changes needed:**

#### Change A: Add extraction step (after existing Step 2, before Step 3)

**INSERT** the following new section between the end of Step 2 (line 50, after the "BINDING TERM HANDOVER" box) and Step 3 (line 87, "Extract Pore Geometry Constraints"):

```markdown
### Step 2.5: Extract Building Block Properties (Optional)

> If Agent 1 describes **chemical properties** of the nodes or linkers using terms like "conjugated", "electron-rich", "open metal sites", "hydrogen bond donor/acceptor", "fluorinated", etc., extract them as boolean filters into `abstract_features`.
>
> **Rules:**
> - Only extract features **explicitly stated** by Agent 1. Do NOT infer or assume.
> - Set to `true` if Agent 1 **requires** the property (e.g., "must have open metal sites").
> - Set to `false` if Agent 1 explicitly **rejects** the property (e.g., "avoid metalated linkers").
> - **Omit** the feature entirely if Agent 1 does not mention it. Do NOT fill with `null`.
> - Place **node-relevant** features in `node_query.abstract_features`.
> - Place **linker-relevant** features in `linker_query.abstract_features`.
> - If Agent 1 does not mention ANY building block properties, set `abstract_features` to `{}` (empty dict).
>
> **Available Features:**
>
> | Feature | Node? | Linker? | Extract when Agent 1 says... |
> |---|---|---|---|
> | `has_open_metal_site` | Yes | No | "open metal site", "unsaturated metal", "exposed metal", "OMS" |
> | `is_conjugated` | Yes | Yes | "conjugated", "pi-conjugation", "extended aromatic system", "delocalized" |
> | `has_hydrogen_bond_donor` | Yes | Yes | "H-bond donor", "hydrogen bond donor", "NH groups", "OH groups" |
> | `has_hydrogen_bond_acceptor` | Yes | Yes | "H-bond acceptor", "hydrogen bond acceptor", "lone pairs", "Lewis base" |
> | `is_electron_rich` | Yes | Yes | "electron-rich", "electron-donating", "EDG", "donor groups" |
> | `is_electron_deficient` | Yes | Yes | "electron-deficient", "electron-withdrawing", "EWG", "acceptor groups" |
> | `is_metalated` | Yes | Yes | "metalated", "metal-containing", "metallolinker" |
> | `is_symmetric` | Yes | Yes | "symmetric", "high-symmetry", "symmetrical" |
> | `is_fluorinated` | Yes | Yes | "fluorinated", "perfluoro", "-CF3", "fluoro" |
> | `is_charged` | Yes | Yes | "charged", "ionic", "cationic", "anionic" |
> | `is_photoswitchable` | Yes | Yes | "photoswitchable", "azobenzene", "diarylethene", "light-responsive" |
```

#### Change B: Update output JSON schema

**CURRENT** output schema (lines 103-143):
```json
{
  "node_query": {
      "reasoning": "Explain derivation of constraints here...",
      "metals_include": ["List", "Symbols"],
      "connectivity": [Integer_List_or_Null],
      "nuclearity": Integer_or_Null,
      "ligand_chemistry": ["List", "Element name of the Ligand atom"]
  },
  "linker_query": {
      "reasoning": "Explain derivation of length/rigidity here...",
      "connectivity": Integer_or_Null,
      "length_min": Float_or_Null,
      "length_max": Float_or_Null,
      "is_rigid": Boolean_or_Null,
      "functional_groups": ["List", "Specific", "Tags"]
  },
```

**REPLACE** `node_query` and `linker_query` blocks with:
```json
{
  "node_query": {
      "reasoning": "Explain derivation of constraints here...",
      "metals_include": ["List", "Symbols"],
      "connectivity": [Integer_List_or_Null],
      "nuclearity": Integer_or_Null,
      "ligand_chemistry": ["List", "Element name of the Ligand atom"],
      "abstract_features": {}
  },
  "linker_query": {
      "reasoning": "Explain derivation of length/rigidity here...",
      "connectivity": Integer_or_Null,
      "length_min": Float_or_Null,
      "length_max": Float_or_Null,
      "is_rigid": Boolean_or_Null,
      "functional_groups": ["List", "Specific", "Tags"],
      "abstract_features": {}
  },
```

Add this comment/note right after the JSON block:
```markdown
> **`abstract_features` format**: Dict of boolean properties. Include ONLY features Agent 1 explicitly mentions. Omit all others.
> Example: `"abstract_features": {"is_conjugated": true, "has_open_metal_site": true}`
> Empty dict `{}` if Agent 1 mentions no building block properties.
```

**IMPORTANT**: Leave `global_requirements` and `geometry_filter` blocks COMPLETELY UNCHANGED.

**Validation**:
1. Read the modified prompt — verify no existing content was accidentally deleted
2. Verify the APPROVED VOCABULARY section (lines 53-84) is completely unchanged
3. Verify the geometry_filter section (lines 87-143) is completely unchanged
4. Verify the output JSON is valid JSON syntax (balanced braces, proper commas)

---

### STEP 1.3: Update Agent 2 Handler (Validation + Display)

**File**: `core/agent2_handler.py`

**Goal**: Add soft validation and display for the new abstract_features field.

#### Change A: Add soft validation

In `_validate_constraints()` method, **AFTER** the existing linker_query check (after line 94) and **BEFORE** the geometry_filter check (line 97), **INSERT**:

```python
        # Check abstract_features (soft validation — optional field)
        node_af = node_q.get('abstract_features', {})
        if node_af and not isinstance(node_af, dict):
            print("   [Validation] WARNING: node_query.abstract_features should be a dict, got:", type(node_af))
        linker_af = l_q.get('abstract_features', {})
        if linker_af and not isinstance(linker_af, dict):
            print("   [Validation] WARNING: linker_query.abstract_features should be a dict, got:", type(linker_af))
```

**DO NOT** add `abstract_features` to the `required_keys` list on line 72. It must remain optional.

#### Change B: Add display

In `_print_constraints()` method, **AFTER** the existing `linker_funcs` display (after line 134), **INSERT**:

```python
        # Abstract Features (if present)
        node_af = node_q.get('abstract_features', {})
        if node_af:
            active = {k: v for k, v in node_af.items() if v is not None}
            if active:
                print(f"Node Abstract Features: {active}")
        linker_af = linker_q.get('abstract_features', {})
        if linker_af:
            active = {k: v for k, v in linker_af.items() if v is not None}
            if active:
                print(f"Linker Abstract Features: {active}")
```

**Validation**:
1. Run `python core/agent2_handler.py` — existing test must still pass
2. Verify no import changes were needed
3. Verify existing required_keys list on line 72 is unchanged: `['node_query', 'linker_query', 'geometry_filter']`

---

### STEP 1.4: Update PORMake Matchmaker (Core Filtering Logic)

**File**: `core/matchmaker.py`

**Goal**: Add abstract_features filtering to node and linker search. Only active when abstract_features is non-empty. Transparent when absent.

#### Change A: Add helper method

**INSERT** the following method to the `Matchmaker` class, before the `_search_nodes` method (before line 128):

```python
    @staticmethod
    def _check_abstract_features(item: dict, required_features: dict) -> bool:
        """
        Check if a building block's abstract_features match the required filter.
        
        Rules:
        - Only non-null required features are checked.
        - If the item's feature is None (unknown), it is NOT excluded (benefit of doubt).
        - If the item's feature is explicitly False and required is True (or vice versa), exclude.
        
        Args:
            item: BB dictionary item (Node or Edge)
            required_features: Dict of {feature_name: True/False/None}
            
        Returns:
            True if item passes all checks, False if any required feature mismatches.
        """
        if not required_features:
            return True
        
        item_af = item.get('abstract_features', {})
        if not item_af:
            return True  # No abstract_features data on item = don't exclude
        
        for feat_key, feat_val in required_features.items():
            if feat_val is None:
                continue  # null requirement = don't filter
            item_val = item_af.get(feat_key)
            if item_val is None:
                continue  # unknown in data = benefit of doubt, don't exclude
            if item_val != feat_val:
                return False  # explicit mismatch
        
        return True
```

#### Change B: Filter nodes

In `_search_nodes()`, **AFTER** the negative filter check (after line 192, the `if not check_negative_tags(item, negative_tags): continue` block) and **BEFORE** `node_candidates.append(item['ID'])` (line 197), **INSERT**:

```python
            # --- ABSTRACT FEATURES FILTER (Optional Boolean Filters) ---
            node_af = specs.get('node_query', {}).get('abstract_features', {})
            if not self._check_abstract_features(item, node_af):
                continue
```

**OPTIMIZATION NOTE**: The `node_af` extraction happens inside the loop but its value is constant. For better performance, move `node_af = specs.get(...)` to BEFORE the loop (e.g., right after `target_ligand_chem` on line 154). Then the inner loop just does:

```python
            # --- ABSTRACT FEATURES FILTER ---
            if not self._check_abstract_features(item, node_af):
                continue
```

**Recommended placement for the extraction** — insert after line 154 (`target_ligand_chem = set(...)`):

```python
        # Abstract Features Prep
        node_af = specs.get('node_query', {}).get('abstract_features', {})
```

Then add the filter check after the negative tags check (after line 192):

```python
            # --- ABSTRACT FEATURES FILTER ---
            if not self._check_abstract_features(item, node_af):
                continue
```

#### Change C: Filter linkers

Similarly in `_search_linkers()`, **extract before the loop** — insert after line 220 (`_, _, negative_tags = parse_functional_groups(specs, vocab_set)`):

```python
        # Abstract Features Prep
        linker_af = specs.get('linker_query', {}).get('abstract_features', {})
```

Then **add the filter check** after the negative tag check (after line 248, `if not check_negative_tags(item, negative_tags): continue`) and **BEFORE** `diag["chem_match"] += 1` (line 252):

```python
            # --- ABSTRACT FEATURES FILTER ---
            if not self._check_abstract_features(item, linker_af):
                continue
```

#### Change D: Update test function

In `test_matchmaker()` (line 381), update `test_specs` to include abstract_features. **REPLACE** the test_specs dict (lines 389-407) with:

```python
    test_specs = {
        "node_query": {
            "metals_include": ["Zr"],
            "connectivity": [12],
            "nuclearity": 6,
            "abstract_features": {}
        },
        "linker_query": {
            "connectivity": 2,
            "length_min": 6.0,
            "length_max": 12.0,
            "is_rigid": True,
            "functional_groups": ["Oxygen"],
            "abstract_features": {}
        },
        "geometry_filter": {
            "target_Di_min": 12.0, "target_Di_max": 20.0,
            "target_Df_min": 7.0, "target_Df_max": 10.0
        }
    }
```

Then **AFTER** the existing test run completes (after line 424), **ADD** a second test with abstract_features active:

```python
    # --- TEST 2: With Abstract Features ---
    print("\n--- TEST 2: Abstract Features Filter ---")
    test_specs_af = {
        "node_query": {
            "metals_include": ["Zr"],
            "connectivity": [12],
            "nuclearity": 6,
            "abstract_features": {"has_open_metal_site": True}
        },
        "linker_query": {
            "connectivity": 2,
            "length_min": 6.0,
            "length_max": 12.0,
            "is_rigid": True,
            "functional_groups": ["Oxygen"],
            "abstract_features": {"is_conjugated": True}
        },
        "geometry_filter": {
            "target_Di_min": 12.0, "target_Di_max": 20.0,
            "target_Df_min": 7.0, "target_Df_max": 10.0
        }
    }
    results_af = matcher.smart_matchmaker_single_node(test_specs_af)
    
    if results_af.get('status') == 'error':
        print(results_af['message'])
    else:
        print(f"Nodes (with abstract_features): {len(results_af['node'])}")
        print(f"Linkers (with abstract_features): {len(results_af['edge'])}")
        
        # Verify: abstract_features should REDUCE candidates vs Test 1
        if len(results_af['node']) <= len(results['node']) and len(results_af['edge']) <= len(results['edge']):
            print("PASS: Abstract features reduced candidate set (as expected)")
        else:
            print("WARNING: Abstract features did NOT reduce candidates (unexpected)")
```

**Validation**:
1. `python core/matchmaker.py` — Both Test 1 (empty AF) and Test 2 (active AF) must run without error
2. Test 1 results must be IDENTICAL to pre-change results (backward compatibility)
3. Test 2 must produce FEWER or EQUAL candidates vs Test 1 (abstract features can only restrict, never expand)
4. If Test 2 produces zero candidates, that's OK — it means the Zr12 + OMS + conjugated filter is very selective

---

### STEP 1.5: NO CHANGES to QMOF/hMOF Matchmakers

**Files**: `core/qmof_matchmaker.py`, `core/hmof_matchmaker.py`

**Reason**: These matchmakers filter whole MOFs (not building blocks). Their index entries (`qmof_index_v2.json`, `hmof_index.json`) do NOT have `abstract_features`. If Agent 2 outputs `abstract_features` while in QMOF/hMOF mode, the matchmakers simply ignore it because they never access that key.

**Validation**: No changes needed. Verify by reading `qmof_matchmaker.py` and `hmof_matchmaker.py` — confirm they never access `abstract_features`.

---

### STEP 1.6: NO CHANGES to Sensitivity Analyzer

**File**: `core/sensitivity_analyzer.py`

**Reason**: The SA receives `agent2_output` and passes parts of it to internal helpers (`_get_node_list`, `_get_linker_list`). These helpers search the BB dict using their own logic (metals, connectivity, length, etc.) — they don't go through the Matchmaker's `_search_nodes`/`_search_linkers`.

For Phase 1, the SA's internal search functions are NOT updated. This means:
- The Matchmaker (used for Set A/D/Z) WILL apply abstract_features filters
- The SA's internal Sets F/G (node-only, linker-only controls) will NOT apply abstract_features
- This asymmetry is ACCEPTABLE for Phase 1 — it actually creates a useful diagnostic: if Set A (with AF) << Set F (without AF), it shows abstract_features are actively constraining

**Validation**: No changes needed.

---

### STEP 1.7: NO CHANGES to run_experiment.py

**File**: `run_experiment.py`

**Reason**: The orchestration layer passes the constraints dict generically. The new `abstract_features` field flows through automatically:
- Line 332: `constraints = agent2.extract_constraints(current_hypothesis)` — Agent 2 now includes abstract_features
- Line 377-387: `matchmaker.match(constraints)` or `matchmaker.smart_matchmaker_single_node(constraints)` — matchmaker now reads abstract_features
- Lines 344-368: Post-extraction validation checks include/exclude tag contradictions only — abstract_features are independent booleans with no contradiction risk

**Validation**: No changes needed.

---

### STEP 1.8: NO CHANGES to Feedback Generator (Phase 1)

**File**: `core/feedback_generator.py`

**Reason**: The feedback tables display rows from the master database (df_master), not from the BB dictionary directly. Adding abstract_features to feedback tables requires parsing `filename` → extracting BB IDs → looking up abstract_features from bb_lookup. This is valuable but adds complexity.

**Deferred to Phase 7** (Enriched Feedback Loop). For Phase 1, the feedback tables remain unchanged. Agent 1 will see the EFFECTS of abstract_features filtering (fewer, more targeted candidates) even without explicit AF columns.

**Validation**: No changes needed.

---

## Final End-to-End Validation

After all steps (1.1 through 1.4) are complete, run the full validation sequence:

### Test A: Unit Test — Matchmaker
```powershell
python core/matchmaker.py
```
**Expected**: Both Test 1 (no AF) and Test 2 (with AF) pass. Test 2 shows reduced candidate counts.

### Test B: Unit Test — Agent 2 Handler
```powershell
python core/agent2_handler.py
```
**Expected**: Existing test passes. If the LLM happens to output abstract_features, they're displayed.

### Test C: Integration Test — Full Experiment (PORMAKE H2 mode)
```powershell
python run_experiment.py
```
Select: `[2] Direct Inquiry` → `[1] H2 Storage`

**Check**:
1. Agent 1 output should mention building block properties (if the LLM picks up on the new prompt)
2. Agent 2 output should contain `abstract_features` dicts (possibly empty if Agent 1 didn't mention any)
3. Matchmaker should run without errors
4. Sensitivity analysis should produce results
5. Check saved `agent2_output.json` in the experiment directory — verify `abstract_features` key exists in both `node_query` and `linker_query`

### Test D: Backward Compatibility — QMOF Mode
```powershell
python run_experiment.py
```
Select: `[2] Direct Inquiry` → `[2] Band Gap`

**Check**: Entire pipeline runs without errors. QMOF matchmaker ignores abstract_features silently.

### Test E: Backward Compatibility — hMOF Mode
```powershell
python run_experiment.py
```
Select: `[2] Direct Inquiry` → `[7] CH4 Storage`

**Check**: Entire pipeline runs without errors. hMOF matchmaker ignores abstract_features silently.

---

## Files Modified (Summary)

| File | Change Type | Risk |
|---|---|---|
| `prompts/agent1_v2.2.9.md` | Prompt text addition | Zero (no code) |
| `prompts/agent2_v4.0.md` | Prompt text + schema addition | Low (LLM output format) |
| `core/agent2_handler.py` | Soft validation + display | Low (additive only) |
| `core/matchmaker.py` | New method + filter logic + test | Medium (core filtering) |

## Files NOT Modified

| File | Reason |
|---|---|
| `core/qmof_matchmaker.py` | No abstract_features in QMOF data |
| `core/hmof_matchmaker.py` | No abstract_features in hMOF data |
| `core/sensitivity_analyzer.py` | Deferred to Phase 7 |
| `core/feedback_generator.py` | Deferred to Phase 7 |
| `core/constraint_utils.py` | abstract_features are independent of tag logic |
| `core/name_resolver.py` | No changes needed |
| `config.py` | No new configuration needed |
| `run_experiment.py` | Generic orchestration — no changes needed |
| `data/*.json` | Read-only data files — never modified |

---

## What Could Go Wrong & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Agent 2 LLM outputs malformed abstract_features (list instead of dict) | Medium | Low | Soft validation in agent2_handler.py catches this; matchmaker's `.get()` returns {} safely |
| Agent 2 LLM invents feature names not in the list | Low | Zero | Matchmaker checks `item_af.get(feat_key)` — unknown keys return None, which passes |
| Agent 2 LLM always outputs empty abstract_features | Medium | Zero | Equivalent to pre-change behavior. No harm. |
| Matchmaker returns 0 candidates due to overly strict AF filter | Low | Low | Same as existing behavior for any overconstrained query — Agent 1 learns from zero-hit feedback |
| Test 2 produces 0 nodes (Zr12 + OMS is very specific) | Medium | Zero | This is valid — not all Zr12 nodes have OMS. The test validates that filtering works. |

---

## Order of Implementation

**MUST be sequential** (each step depends on the previous):

1. **Step 1.1** (Agent 1 prompt) — can be done independently
2. **Step 1.2** (Agent 2 prompt) — can be done independently (but logically follows 1.1)
3. **Step 1.3** (agent2_handler.py) — depends on understanding the schema from 1.2
4. **Step 1.4** (matchmaker.py) — depends on understanding the schema from 1.2
5. **Final validation** — depends on all 4 steps

Steps 1.1 and 1.2 can be done in parallel. Steps 1.3 and 1.4 can be done in parallel. But validate after each pair.
