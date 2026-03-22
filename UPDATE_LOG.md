# Update Log: Phase X + Agent 1 v2.3.0

**Date**: 2026-03-22
**Branch**: `feat/phase-x-branched-matching-and-v230-prompts`

---

## Overview

Two major changes in this branch:

1. **Phase X: Branched Hypothesis Matching** -- Adds OR-of-ANDs (`linker_branches`) to the Agent 2 output schema, enabling Agent 1's alternative linker strategies to be preserved through the pipeline instead of being collapsed into a lowest-common-denominator tag like "Aromatic".

2. **Agent 1 Prompt v2.3.0** -- Redesigned reasoning strategy ("Beat Bayesian Optimization") with database-type-specific variants for PorMake, QMOF, and hMOF. Includes structured exploration budget, hypothesis falsification, pattern-based beam analysis, and over-constraint avoidance.

---

## Part 1: Phase X -- Branched Hypothesis Matching

### Problem

Agent 1 proposes alternatives: "use pyridine dicarboxylate OR ether aromatics OR azolate linkers". But Agent 2's output schema could only express flat AND or flat OR -- not OR-of-ANDs. This forced Agent 2 to extract the lowest common denominator ("Aromatic"), which matched 52-72% of every database and produced essentially unfiltered results.

**Evidence (from 178 real agent2_output.json files across 65 experiments):**
- `Aromatic` appeared in 73% (130/178) of all Agent 2 extractions -- the #1 tag by 3x
- 39% (69/178) of extractions used ONLY generic tags (Aromatic, Ring, Heterocycle, etc.)
- 52% (93/178) of Agent 1 outputs contained explicit OR/alternative linker strategies
- 0/178 outputs ever used `linker_branches` (feature didn't exist)

### Solution

Added `linker_branches` to Agent 2's output schema -- an array of alternative search branches, each with AND-within-branch semantics, combined with OR-between-branches.

**Before:**
```json
"functional_groups": ["Aromatic"]
```

**After:**
```json
"functional_groups": [],
"linker_branches": [
    {"description": "pyridine dicarboxylate", "required_tags": ["Pyridine", "Carboxyl"]},
    {"description": "ether aromatic", "required_tags": ["Ether", "Aromatic"]},
    {"description": "azolate", "required_tags": ["Azolate"]}
]
```

### Files Changed (Phase X)

| File | Change | Risk |
|------|--------|------|
| `core/constraint_utils.py` | New `check_linker_branches()` utility + tests | Low (additive) |
| `core/matchmaker.py` | Branch check in `_search_linkers()` + Phase D union logic bypass | Medium |
| `core/qmof_matchmaker.py` | Branch check after OR-tags filter | Low |
| `core/hmof_matchmaker.py` | Branch check after OR-tags filter | Low |
| `core/sensitivity_analyzer.py` | Branch check in `_get_linker_list()` + null connectivity fix | Low |
| `core/agent2_handler.py` | Soft validation + display + cp949 encoding fixes | Low |
| `prompts/agent2_v4.0.md` | Step 2.7 branch extraction rules + schema + union logic rewrite | High (LLM behavior) |
| `prompts/agent1_v2.2.9.md` | Added "Alternative Strategies" guidance (1 line) | Zero |

### Backward Compatibility

Empty `linker_branches` = existing behavior unchanged. All existing experiments continue to work identically.

### Validation

- `check_linker_branches()`: All assertion-based unit tests pass
- PorMake matchmaker: Test 1 (baseline) passes, Test 3 (branches) reduces 54->12 linkers
- QMOF matchmaker: Import + integration OK
- hMOF matchmaker: Tests 1-3 pass (22,422 / 5,063 / 51,163 matches)
- Agent 2 handler: Branches display correctly, validation works

---

## Part 2: Agent 1 Prompt v2.3.0

### Motivation

Analysis of the v2.2.9 H2 Storage experiment (batch_20260320_2258) revealed:

1. **Anecdote-chasing**: Agent 1 copied individual high-performing structures from Beam 3 feedback instead of extracting distributional patterns (e.g., "Dy+thiophene scored 572" led to iter 4 targeting Dy+thiophene specifically)
2. **No exploration budget**: All 5 iterations were reactive, with no planned exploration-exploitation allocation
3. **Over-constraining**: Iter 1 used SA>=4000 + 10 geometry filters, producing 0 matches
4. **No beam comparison**: Agent 1 never compared Beam 1 vs 2 vs 3 to diagnose whether geometry or chemistry was the bottleneck
5. **Match instability**: 0, 1, 6, 1, 0 matches across iterations

### Design Philosophy: Strategy Injection, Not Knowledge Injection

The v2.3.0 changes inject **reasoning strategies** (how to search), not **domain knowledge** (what to search for). No chemistry, metal, linker, or property value is mentioned in any rule. The same prompt (per database type) handles all applications equally.

This is analogous to teaching a scientist experimental design principles, not giving them the answer. It is comparable to how Bayesian Optimization requires choosing an acquisition function (EI/UCB) -- a "strategy injection" -- without encoding domain-specific knowledge.

### New Reasoning Rules

| Rule | Name | What it does |
|------|------|-------------|
| A | Extract Patterns, Not Individuals | Read Beam 3 Pattern Summary percentages, not top-1 structure |
| B | Hypothesis Falsification | Each iteration states a testable mechanism + prediction |
| C | Exploration Budget Management | Iter 1-2 explore broadly, 3-4 exploit, 5 refine/pivot |
| D | Diversify per Iteration | Use 3-4 branches to test multiple chemistries simultaneously |
| E | Beam Comparisons | Compare Beam 1 vs 2 vs 3 to diagnose geometry vs chemistry bottleneck |
| F | Avoid Over-Constraining | 0 matches -> fewer constraints next time, not different ones |

### New JSON Output Fields

Added to `meta_cognition`:
- `hypothesis_to_test`: The specific mechanism being tested this iteration
- `prediction`: Expected performance range and falsification criteria
- `beam_analysis`: Which beam patterns informed this iteration's strategy

### Database-Type Variants

| File | Database | Additional Rules |
|------|----------|-----------------|
| `agent1_v2.3.0.md` | PorMake (12K, assembly) | Base rules A-F |
| `agent1_v2.3.0_qmof.md` | QMOF (20K, electronic) | Rule G: Electronic Property Awareness (chemistry-first, geometry deprioritized) |
| `agent1_v2.3.0_hmof.md` | hMOF (51K, gas adsorption) | Rule G: Whole-MOF Awareness, Rule H: Gas-agnostic geometry |

### Auto-Routing

`config.get_agent1_prompt_path()` returns the correct variant based on `ACTIVE_METRIC_COLUMN`:
- `target` -> PorMake prompt
- `outputs.pbe.bandgap` -> QMOF prompt
- hMOF metrics -> hMOF prompt

### Files Changed (v2.3.0)

| File | Change |
|------|--------|
| `prompts/agent1_v2.3.0.md` | New: PorMake variant with "Beat BO" reasoning strategy |
| `prompts/agent1_v2.3.0_qmof.md` | New: QMOF electronic property variant |
| `prompts/agent1_v2.3.0_hmof.md` | New: hMOF gas adsorption variant |
| `config.py` | Added prompt routing: `_AGENT1_PROMPT_*` + `get_agent1_prompt_path()` |
| `core/agent1_handler.py` | Uses `get_agent1_prompt_path()` instead of static `AGENT1_PROMPT_PATH` |

---

## Part 3: Bug Fixes

| Bug | File | Fix |
|-----|------|-----|
| cp949 encoding crash on Windows | `core/agent2_handler.py` | Replaced all non-ASCII chars (A, m2/g, A^3, ->, [OK], [FAIL]) |
| `connectivity: null` crash in SA | `core/sensitivity_analyzer.py` | Added null guard defaulting to ditopic (2) |
| PorMake mode rules for branches | `core/agent2_handler.py` | Added `_PORMAKE_MODE_RULES` warning: do NOT put coordination tags (Carboxyl, Azolate) in linker_branches for PorMake mode |

---

## Part 4: Experimental Results

### A/B Test: H2 Storage PorMake (v2.2.9 vs v2.3.0)

| Metric | v2.2.9 | v2.3.0 | Improvement |
|--------|--------|--------|-------------|
| Best H2 (full hypothesis) | 572.1 | 609.8 | +37.7 (+6.6%) |
| Gap to DB best (616.9) | 7.3% | 1.2% | 6.1pp closer |
| Zero-match iterations | 2/5 | 0/5 | Eliminated |
| Match stability | 0,1,6,1,0 | 65,56,61,56,23 | Stable |
| linker_branches adoption | N/A | 100% (5/5) | Full adoption |

### Full 10-Experiment Batch (v2.3.0)

**Gas Adsorption (EF-based):**

| Experiment | Best EF@1% | Best EF@5% | Median Uplift | Gap to DB Best |
|---|---|---|---|---|
| XeKr Selectivity (hMOF) | 15.5x | 11.7x | +178% | 84.8% |
| CH4 Storage (hMOF) | 7.1x | 3.8x | +23.4% | 8.6% |
| H2 Storage (PorMake) | 1.5x | 9.3x | +16.0% | 1.2% |
| H2 Storage (hMOF) | 2.8x | 2.9x | +8.4% | 0.1% |
| CO2 Capture (hMOF) | 1.5x | 5.2x | +83.7% | 68.0% |

**QMOF Bandgap (distribution shift):**

| Experiment | Target | Baseline% | Best Filtered% | Enrichment | Shift Correct? |
|---|---|---|---|---|---|
| UV Activity | 3.1-4.0 eV | 12.8% | 32.4% | 2.5x | YES |
| Above 4 eV | >4.0 eV | 5.4% | 15.0% | 2.8x | YES |
| Vis Water Split | 1.6-3.1 eV | 46.2% | 62.3% | 1.3x | YES (iter 3) |
| 3-4 eV | 3.0-4.0 eV | N/A | N/A | N/A | FAIL (0 matches) |
| Below 0.1 eV | <0.1 eV | N/A | N/A | N/A | FAIL (0 matches) |

**Key finding**: When the system produces matches, the distribution shift is in the correct direction 100% of the time. The 2 QMOF failures are constraint-to-database intersection problems (the chemistry tags proposed don't exist in QMOF), not scientific reasoning failures.

---

## Additional Files

| File | Purpose |
|------|---------|
| `run_batch_experiments.py` | Non-interactive batch runner for all 10 experiments |
| `.sisyphus/plans/phase-X-branched-hypothesis-bridge.md` | Detailed implementation plan for Phase X |
| `.sisyphus/plans/phase1-abstract-features.md` | Prior plan (reference) |
