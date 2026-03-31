# LLM2POR: Autonomous MOF Designer v3 (v2.4)

An autonomous agent system that designs Metal-Organic Frameworks (MOFs) through iterative hypothesis generation, constraint extraction, and database-driven feedback. The system uses LLMs (GPT / Gemini) to propose MOF designs and evaluates them against real computational databases (QMOF, hMOF, PORMAKE).

**v2.5 highlights:** hMOF metal constraint (Cu/V/Zn/Zr) in user queries, Scientific Journal removed (redundant with multi-turn context), unified 4-beam feedback for all databases, hMOF sensitivity crash fix. See [Branch Changes](#branch-changes) for details.

## How It Works

```
User Inquiry ("Design a MOF with band gap 3-4 eV")
        |
        v
  [Agent 0] Problem Consultant (optional)
    Clarifies requirements via interview
        |
        v
  [Agent 1] Hypothesis Generator
    Proposes MOF design (metals, linkers, geometry)
        |
        v
  [Agent 2] Constraint Extractor
    Converts hypothesis into searchable database filters
        |
        v
  [Matchmaker] Component Discovery
    Finds matching MOFs in the database
        |
        v
  [Sensitivity Analyzer] Evaluation
    Measures how well the hypothesis performs
        |
        v
  [Feedback Generator] Learning Signal
    Creates structured feedback for Agent 1
        |
        v
  Agent 1 refines hypothesis (loop back)
```

## Setup

### 1. Python Environment

Requires **Python 3.10+**. Create a virtual environment and install dependencies:

```bash
# Create and activate virtual environment
python -m venv llm2auto
# Windows:
llm2auto\Scripts\activate
# macOS/Linux:
source llm2auto/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Key Configuration

Create a `.env` file in the project root (this file is git-ignored and never committed):

```bash
# .env
LLM_PROVIDER=openai          # "openai" or "gemini"
OPENAI_API_KEY=...           # Your OpenAI API key
GEMINI_API_KEY=...           # Your Gemini API key (if using Gemini)
```

- **OpenAI**: Get your key at https://platform.openai.com/api-keys
- **Gemini**: Get your key at https://aistudio.google.com/apikey

The active model is set in `config.py`:
- OpenAI: `OPENAI_MODEL = "gpt-5.2"` (line 29)
- Gemini: `GEMINI_MODEL = "gemini-3-flash-preview"` (line 33)

Switch providers by changing `LLM_PROVIDER` in your `.env` file.

### 3. Data Files

The repository includes all required data files via Git LFS. After cloning, run:

```bash
git lfs pull
```

This downloads the large database files (~98 MB total):
- `data/qmof_index_v2.json` (25 MB) - 20,373 QMOF MOFs for band gap mode
- `data/hMOF/hmof_index.json` (50 MB) - 51,163 hypothetical MOFs for gas adsorption mode
- `data/qmof.csv` (21 MB) - QMOF property database
- `data/total_characteristics&name_singleonly_20251203.csv` (1 MB) - PORMAKE master database

## Usage

```bash
python run_experiment.py
```

### Experiment Modes

You will be prompted to choose:

**[1] With Agent 0** - An AI consultant interviews you to clarify your design requirements before passing them to the hypothesis generator.

**[2] Direct Inquiry** (recommended) - Skip the interview and go straight to hypothesis generation with a preset or custom inquiry.

### Available Inquiry Types

| # | Category | Inquiry | Database |
|---|----------|---------|----------|
| 1 | H2 Storage | High capacity hydrogen storage at 77K | PORMAKE |
| 2 | Band Gap | Optimal band gap for visible-light water splitting | QMOF |
| 3 | Band Gap | Band gap between 3-4 eV | QMOF |
| 4 | Band Gap | Band gap for UV activity | QMOF |
| 5 | Band Gap | Band gap below 0.1 eV | QMOF |
| 6 | Band Gap | Band gap above 4 eV | QMOF |
| 7 | CH4 Storage | High methane storage at 298K. **Metals: Cu/V/Zn/Zr only.** | hMOF |
| 8 | CO2 Capture | CO2 capture at low pressure. **Metals: Cu/V/Zn/Zr only.** | hMOF |
| 9 | Xe/Kr Selectivity | High Xe/Kr selectivity. **Metals: Cu/V/Zn/Zr only.** | hMOF |
| 10 | H2 Storage | High H2 uptake at 100 bar 77K. **Metals: Cu/V/Zn/Zr only.** | hMOF |
| 11 | H2 Storage | High H2 uptake at 5 bar 77K | PORMAKE (5bar) |
| 12 | Custom | Type your own design inquiry | Auto-detected |

**hMOF metal constraint:** The hMOF database (Snurr group, 51K hypothetical MOFs) contains only Cu (25.6%), V (7.2%), Zn (62.1%), and Zr (4.7%) metal nodes. All other metals have <0.4% representation. The metal constraint is included in the user query (not the system prompt) so the LLM knows what materials are available without revealing database identity. Experimental validation showed this eliminated zero-match failures (H2_Storage: 80% zero-match rate reduced to 0%) and improved Top-1 performance across all hMOF targets.

### Feedback Types

After each iteration, choose a feedback strategy:

| # | Type | Description |
|---|------|-------------|
| 1 | 4-Beam Diagnostic | Unified chemistry-first diagnostic for all databases. Beams: Full Hypothesis (Z), Chemistry Only (A), Metal Only (F), Global Baseline. **Default.** |
| 2 | Universe Baseline | Samples across all DB; useful when 0 candidates found |
| 3 | Geometric Optimizer | Tests random vs constrained geometry |
| 4 | Chemical Pivot | Tests random metals vs your geometry |
| 5 | Best vs Worst | Stratified sampling to find patterns |
| 6 | Hypothesis Validation | Tests only the complete hypothesis block |
| 7 | Virtual Synthesis | Lab synthesis simulation |

Agent 1 feedback is **blinded** — anonymous MOF labels, no database names, generic beam headers. Agent 1 cannot infer which database is active.

Type `quit` at any feedback prompt to end the experiment.

### Output

Results are saved to `experiments/exp_YYYYMMDD_HHMM_{mode}/`:
- `raw_user_input.txt` - Your original inquiry
- `experiment_log.txt` - Full run log
- `iteration_N/` - Per-iteration outputs (hypothesis, constraints, sensitivity reports)

---

## Branch Changes: `feat/phase-x-branched-matching-and-v230-prompts`

This branch introduces three major changes to the system, each addressing a specific weakness discovered during experimental runs. Together, they form a controlled ablation study of the LLM's reasoning capabilities.

### 1. Phase X: Branched Hypothesis Matching

#### The Problem: Information Loss in the Constraint Pipeline

Agent 1 frequently proposes alternative linker strategies in its hypotheses. For example:

> "Use pyridine dicarboxylate OR ether-containing aromatics OR azolate linkers"

But Agent 2's output schema could only express flat AND or flat OR logic for functional groups -- not OR-of-ANDs (where each alternative branch requires multiple tags simultaneously). This forced Agent 2 to extract the lowest common denominator tag that all alternatives shared.

**Evidence from 178 real agent2_output.json files across 65 experiments:**
- `Aromatic` appeared in 73% (130/178) of all Agent 2 extractions -- the #1 tag by 3x
- 39% (69/178) of extractions used ONLY generic tags (Aromatic, Ring, Heterocycle)
- 52% (93/178) of Agent 1 outputs contained explicit OR/alternative strategies
- 0/178 outputs ever preserved the branch structure

The result: Agent 1's carefully reasoned chemistry was collapsed into "Aromatic", which matched 52-72% of every database, producing essentially unfiltered results. The pipeline was discarding exactly the information that made the LLM valuable.

#### The Solution: `linker_branches` Schema Extension

Added `linker_branches` to Agent 2's output schema -- an array of alternative search branches with AND-within-branch semantics combined with OR-between-branches.

**Before (flat OR, information lost):**
```json
{
  "functional_groups": ["Aromatic"]
}
```

**After (OR-of-ANDs, alternatives preserved):**
```json
{
  "functional_groups": [],
  "linker_branches": [
    {"description": "pyridine dicarboxylate", "required_tags": ["Pyridine", "Carboxyl"]},
    {"description": "ether aromatic", "required_tags": ["Ether", "Aromatic"]},
    {"description": "azolate", "required_tags": ["Azolate"]}
  ]
}
```

Each branch is searched independently. A linker matches if it satisfies ALL tags within ANY single branch. Empty `linker_branches` preserves backward compatibility -- all existing experiments continue to work identically.

#### Files Changed

| File | Change |
|------|--------|
| `core/constraint_utils.py` | New `check_linker_branches()` utility with assertion-based unit tests |
| `core/matchmaker.py` | Branch check in `_search_linkers()` + Phase D union logic bypass |
| `core/qmof_matchmaker.py` | Branch check after OR-tags filter |
| `core/hmof_matchmaker.py` | Branch check after OR-tags filter |
| `core/sensitivity_analyzer.py` | Branch check in `_get_linker_list()` + null connectivity guard |
| `core/agent2_handler.py` | Soft validation, display formatting, cp949 encoding fixes |
| `prompts/agent2_v4.0.md` | Step 2.7 branch extraction rules + schema update + union logic rewrite |

---

### 2. Agent 1 Prompt Versions: Three-Way Ablation Study

The branch introduces three prompt versions for Agent 1, designed as a controlled ablation to isolate which factors drive performance improvements.

#### v2.2.9 -- Baseline LLM (`prompts/agent1_v2.2.9.md`)

The original prompt. Provides the LLM with:

- **Core Philosophy (4 principles):**
  1. *Mechanism-Grounded Reasoning:* Justify choices with chemical rationale, not popularity ("Use Zr because it's popular" is forbidden)
  2. *Causal Hierarchy:* Start from the Performance Goal, derive the required Geometry, then select Components (inverse design)
  3. *Stateless Execution:* Explicitly list all metals/groups every time (the database has no memory)
  4. *Stagnation Trap:* If performance plateaus for 3 iterations, abandon the current chemistry entirely and pivot to a fundamentally different mechanism

- **Design Toolbox:** A menu of available descriptors (geometry: di, df, sa, vf, density; electronic: oxidation_states, coordination_geometry for QMOF; components: node_metal, functional_groups, linker_length, building_block_properties)

- **Chain of Thought (3 steps):** Step 1: Identify the mechanism. Step 2: Derive the geometry. Step 3: Select the components.

- **Output Format:** Simple JSON with `meta_cognition.reasoning` (single text field) and `lesson_learnt` (single text field).

- **Memory:** Multi-turn conversation only. The LLM sees its full conversation history (all prior hypotheses and feedback) but receives no additional compressed summary.

**What it does NOT have:** No guidance on how to read the 4-beam feedback structure. No explicit rules for pattern extraction vs. anecdote-chasing. No exploration/exploitation budget management. The LLM relies entirely on its pretraining to decide how to interpret feedback and revise hypotheses.

#### v2.3.0 -- Reasoning Strategy (`prompts/agent1_v2.3.0*.md`)

Everything from v2.2.9, **plus** a new section titled **"Reasoning Strategy: Beat Bayesian Optimization"** containing six explicit rules:

**Rule A -- Extract Patterns, Not Individuals:**
When reading Beam 3 (Geometric Control) feedback, never anchor on a single high-performing structure. Read the Pattern Summary first. Target metals that appear at >20% frequency and backbones that dominate the top-10 list. A metal appearing once at rank #1 is an anecdote; a backbone appearing in 60% of the top-10 is a pattern.

> *Anti-pattern (forbidden):* "Beam 3 shows Dy+thiophene at 572, so I will use Dy+thiophene." This copies one data point.
>
> *Correct:* "Beam 3 shows diverse metals (Co 20%, Dy 10%, Eu 10%) but benzene_ring backbone dominates (60%). The mechanism may be aromatic backbone rigidity, not metal identity."

**Rule B -- Hypothesis Falsification (Scientific Method):**
Each iteration must test a specific mechanism, not just chase performance. Structure reasoning as:
- *Hypothesis:* "Property X drives performance because of mechanism Y."
- *Test:* "If X drives performance, then changing Z while keeping X should maintain performance."
- *Prediction:* "I expect performance > N because..."

After receiving feedback, explicitly evaluate whether the hypothesis was confirmed or falsified.

**Rule C -- Exploration Budget Management:**
Allocate the finite iteration budget strategically:
- *Iterations 1-2:* Broad exploration. Test 2-3 fundamentally different chemistry families. Cast a wide net with relaxed geometry. The goal is INFORMATION, not peak performance.
- *Iterations 3-4:* Focused exploitation. Double down on the most promising mechanism. Tighten geometry to the empirically validated window.
- *Iteration 5+:* Final refinement OR radical pivot if performance has plateaued.

**Rule D -- Diversify Chemistry per Iteration:**
Use `linker_branches` to test multiple chemistry families simultaneously within each iteration (like running parallel experiments). Example: `[Biphenyl, Naphthalene, Thiophene, Pyridine]` as separate branches -- one iteration tests four hypotheses.

**Rule E -- Read Beam Comparisons, Not Just Beam 1:**
- *Beam 1 vs Beam 2:* If Beam 1 >> Beam 2, your geometry constraints are adding value. If Beam 1 ~ Beam 2, geometry is irrelevant -- loosen it.
- *Beam 1 vs Beam 3:* If Beam 3 >> Beam 1, your chemistry is the bottleneck. Beam 3's geometry window contains better MOFs that your chemistry misses.
- *Beam 2 vs Beam 3:* If Beam 3 >> Beam 2, BOTH chemistry and geometry need work.

**Rule F -- Avoid Over-Constraining:**
The database is finite. Every constraint removes candidates. Apply the minimum constraints necessary to test the hypothesis. If an iteration returns 0 matches, the next iteration MUST use FEWER constraints, not different ones at the same specificity.

**Enhanced Output Format:**
- `meta_cognition` gains 3 new fields: `hypothesis_to_test` (the specific mechanism being tested), `prediction` (expected performance range and falsification criteria), `beam_analysis` (pattern extraction from beam summaries)
- `lesson_learnt` becomes a structured 4-field object:
  - `beam_comparison`: Which beam performed best and what does that imply?
  - `constraint_diagnosis`: How many candidates matched? Which constraint was most restrictive?
  - `pattern_extraction`: Which metals/backbones appear most frequently among top performers?
  - `strategy_change`: What specific change will be made next, stated as a testable prediction?

**Per-Database Variants:**
Three prompt files with database-specific rules appended:
- `agent1_v2.3.0.md` -- PorMake (base rules A-F only)
- `agent1_v2.3.0_qmof.md` -- QMOF (adds bandgap-specific rules for targeting specific eV ranges)
- `agent1_v2.3.0_hmof.md` -- hMOF (adds gas adsorption-specific rules for uptake/selectivity)

#### v2.3.1 -- Reflexion Only (`prompts/agent1_v2.3.1_reflexion_only.md`)

An ablation prompt designed to isolate the contribution of the structured output format from the reasoning rules.

- **Has:** The same structured output format as v2.3.0 (4-field `meta_cognition` with `hypothesis_to_test`, `prediction`, `beam_analysis`; 4-field `lesson_learnt` with `beam_comparison`, `constraint_diagnosis`, `pattern_extraction`, `strategy_change`)
- **Does NOT have:** No Rules A-F. No "Beat Bayesian Optimization" section. No per-database rules.
- **Universal:** One prompt for all three databases (PorMake, QMOF, hMOF)

The output format forces the LLM to reflect structurally (compare beams, diagnose constraints, extract patterns, state a testable strategy change) -- but without telling it *how* to do any of those things.

#### Ablation Design Summary

```
v2.2.9 (Baseline)          v2.3.1 (Reflexion Only)        v2.3.0 (Full Strategy)
  Simple output format   ->  Structured output format   ->  Structured output format
  No reasoning rules         No reasoning rules              Rules A-F
  No beam guidance           No beam guidance                Beam comparison rules
  No per-DB rules            No per-DB rules                 Per-DB rules (G/H)
```

| Comparison | Isolates | Question |
|------------|----------|----------|
| v2.2.9 -> v2.3.1 | Format effect | Does forcing the LLM to write structured reflections (beam comparisons, constraint diagnoses, pattern extraction) improve performance, even without telling it how? |
| v2.3.1 -> v2.3.0 | Rules effect | Do explicit reasoning rules (pattern extraction > anecdotes, hypothesis falsification, exploration budgets, beam comparison logic) add value beyond the structured format? |
| v2.2.9 -> v2.3.0 | Combined effect | What is the total improvement from both format and rules together? |

---

### 3. Bug Fixes

| Fix | File | Detail |
|-----|------|--------|
| Null connectivity guard | `core/sensitivity_analyzer.py` | Prevent crash when connectivity data is missing from matched MOFs |
| hMOF combinatorial space crash | `core/sensitivity_analyzer.py` | Added hMOF early return in `calculate_combinatorial_space()` — was falling through to PorMake's node/edge logic, crashing on `connectivity: None`. Killed CH4@iter6, CO2@iter8, XeKr@iter1 in batch runs. |
| Null connectivity field | `core/sensitivity_analyzer.py` | `get('connectivity') or []` instead of `get('connectivity', [])` to catch explicit `None` values from Agent 2 |
| cp949 encoding fix | `core/agent2_handler.py` | Handle Korean Windows encoding in file I/O |

### 4. v2.4: Chemistry-First Feedback, 4-Beam Diagnostics, and Agent Blinding

Major refactor of the feedback pipeline for publication readiness. Three categories of changes:

**Bug Fixes:**

| Fix | File | Detail |
|-----|------|--------|
| QMOF "Any" metal guard | `core/qmof_matchmaker.py` | Added missing `if any(m == "any" ...)` check — without this, exploratory QMOF queries with `metals: ["Any"]` silently returned zero results. PORMake and hMOF already had this guard. |
| SA negative-tag substring matching | `core/sensitivity_analyzer.py` | Replaced hand-rolled substring check (`if neg in combined_text`) with shared `check_negative_tags()` from `constraint_utils.py`. The old code used substring matching (so `"amine"` banned `"primary_amine"`), diverging from the matchmakers' exact set-membership logic. |

**4-Beam Database-Aware Feedback (feedback_generator.py):**

The old 3-beam diagnostic (Z, A, E) is replaced with database-aware 4-beam designs:

| Database | Beam 1 | Beam 2 | Beam 3 | Beam 4 | Diagnostic Question |
|----------|--------|--------|--------|--------|---------------------|
| PORMake/hMOF | Full hypothesis (Z) | Chemistry only (A) | Metal only (F) | Global baseline | Is geometry or chemistry the bottleneck? |
| QMOF | Full hypothesis (Z) | Metal control (F) | Linker control (G) | Global baseline | Is it metal d-electrons or linker conjugation? |

QMOF gets a different beam design because it has no geometry gate — Beams 1 and 2 would be identical under the PORMake/hMOF design, wasting a diagnostic slot. The QMOF-specific design isolates metal vs. linker electronic contributions instead.

**Agent 1 Blinding (prevent database identity inference):**

| Measure | Detail |
|---------|--------|
| Anonymous MOF labels | Chemistry profiles now show `MOF-1, MOF-2, ...` instead of `N419+E12`, `qmof-XXX`, `hmof-XXX` |
| Generic beam headers | "4-BEAM DIAGNOSTIC" / "4-BEAM ELECTRONIC DIAGNOSTIC" — no database name |
| Unified footer messages | "No entries match your Metal + Functional Group constraints" — no database name |
| Generic prompt instructions | Agent 1 prompt says "analyze beams as labeled" without specifying which beam configuration to expect |

**Prompt Changes:**

| File | Change |
|------|--------|
| `agent1_handler.py` | Reflexion prompt now describes beams generically, not hard-coded to chemistry+geometry design |
| `agent1_v2.3.1_reflexion_only.md` | `beam_analysis` and `beam_comparison` fields updated for generic 4-beam interpretation |
| `agent1_v2.2.9.md`, `agent1_v2.3.0_qmof.md` | Removed `(QMOF-only)` labels from electronic descriptor descriptions (blinding) |
| `agent2_v4.0.md` | Step 3 renamed "Geometry Predictions (Second-Stage Gate)" with note that geometry is not a primary filter |
| `agent2_handler.py` | `geometry_filter` validation softened — empty is valid for chemistry-first mode |

---

### 5. v2.5: hMOF Metal Constraint, Scientific Journal Removal

**hMOF metal constraint in user queries:**
The hMOF database only contains Cu/V/Zn/Zr metals (>99.6%). Without disclosing this, the LLM suggests valid but absent metals (Al, Fe, Cr, Ni) producing zero hits. Added "Limit metal nodes to Cu, V, Zn, and Zr only." to all 4 hMOF preset queries. This is constraint specification (what reagents are available), not bias (which to prefer).

Experimental validation (v2.3.1, 10 iterations):

| Experiment | Before: Zero-match rate | After: Zero-match rate | Before: Top-1 | After: Top-1 |
|---|---|---|---|---|
| H2_Storage | 80% (crashed iter 6) | **0%** (10/10 complete) | 53.7 | **58.58** |
| CH4_Storage | 0% | 0% | 174.4 | **261.2** |
| XeKr_Selectivity | 0% | 0% | 46.0 | **167.1** |
| CO2_Capture | 0% | 0% | 15.1 | **16.1** |

**Scientific Journal removed:**
The Scientific Journal was a cumulative text summary injected into the system prompt via `{SCIENTIFIC_JOURNAL}` placeholder each iteration. By iteration 10 it added ~7K tokens of compressed duplicates — information the LLM already had in its multi-turn chat context. Removed from all prompt files and the injection pipeline. The `lesson_learnt` output field is preserved for LLM self-reflection and experiment log analysis.

**Unified 4-beam feedback:**
Merged the separate QMOF branch in `feedback_generator.py` into the unified chemistry-first 4-beam design. For QMOF, Beam 1 ~ Beam 2 naturally (no geometry gate), which correctly signals "geometry irrelevant" without needing a separate beam layout.

### 6. Pending Work: LLM vs. Numerical Baselines Comparison

The 3 LLM prompt versions (v2.2.9, v2.3.0, v2.3.1) must still be compared against 4 numerical baseline strategies in a controlled head-to-head study:

| Baseline | Description |
|----------|-------------|
| **Random Search** | Pick N random MOFs, track cumulative best. No model, no learning. |
| **LHS** | Latin Hypercube Sampling in structural feature space with nearest-neighbor lookup. Systematic space-filling design, no learning. |
| **BO (Bayesian Optimization)** | GP surrogate (small DBs) or RF surrogate (large DBs) with Expected Improvement acquisition. The strongest numerical baseline -- builds an explicit model of the structure-property landscape. |
| **GA (Genetic Algorithm)** | Population-based evolutionary search with tournament selection, feature-space crossover, and Gaussian mutation. |

The comparison study will test the central hypothesis:

> **Does LLM pretraining knowledge provide a measurable sample-efficiency advantage over numerical optimization methods that learn entirely from scratch within each experiment?**

Key fairness considerations:
- The LLM observes ~30 MOFs per iteration (3 beams x 10 MOFs) vs. baselines observe 1 MOF per step. Baselines will be run at multiple budget tiers (N=10, 30, 100, 300) to bracket equivalent information budgets.
- The LLM has prior chemistry knowledge from pretraining. This is the central hypothesis being tested, not a confound to correct.
- The LLM operates in semantic/constraint space; baselines operate in numerical feature space.

Baseline infrastructure exists locally but is not yet committed pending completion of all experimental runs.

---

## Project Structure

```
.
├── run_experiment.py          # Interactive entry point (single experiment)
├── config.py                  # All configuration, data paths, metric registry
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not in git)
├── core/                      # Runtime modules
│   ├── agent0_handler.py      # Problem Consultant (interview)
│   ├── agent1_handler.py      # Hypothesis Generator (multi-turn)
│   ├── agent2_handler.py      # Constraint Extractor (stateless, with linker_branches validation)
│   ├── constraint_utils.py    # Tag/ontology parsing + linker branch matching
│   ├── feedback_generator.py  # Structured feedback for Agent 1
│   ├── hmof_matchmaker.py     # hMOF direct MOF matching (with branch support)
│   ├── llm_client.py          # Unified OpenAI/Gemini API client
│   ├── matchmaker.py          # PORMAKE component assembly matching (with branch support)
│   ├── memory_manager.py      # Experiment state persistence
│   ├── name_resolver.py       # Building block ID-to-name resolver
│   ├── qmof_matchmaker.py     # QMOF direct MOF matching (with branch support)
│   └── sensitivity_analyzer.py # Performance evaluation engine (with branch support)
├── prompts/                   # LLM system prompts
│   ├── agent0_v3.md           # Problem Consultant prompt
│   ├── agent1_v2.2.9.md       # Baseline LLM (no reasoning rules)
│   ├── agent1_v2.3.0.md       # Reasoning Strategy -- PorMake variant (Rules A-F)
│   ├── agent1_v2.3.0_qmof.md  # Reasoning Strategy -- QMOF variant (Rules A-F + bandgap rules)
│   ├── agent1_v2.3.0_hmof.md  # Reasoning Strategy -- hMOF variant (Rules A-F + gas adsorption rules)
│   ├── agent1_v2.3.1_reflexion_only.md  # Reflexion Only (structured format, no rules, universal)
│   └── agent2_v4.0.md         # Constraint Extractor (with linker_branches schema)
└── data/                      # Databases (large files via Git LFS)
    ├── pormake_bb_dictionary_v5.json
    ├── pormake_topo_dictionary_v3.json
    ├── unified_ontology.json
    ├── qmof.csv
    ├── qmof_ids_with_topology.txt
    ├── qmof_index_v2.json
    └── hMOF/
        └── hmof_index.json
```

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MAX_OUTPUT_TOKENS` | 32000 | Max tokens for LLM response |
| `LLM_REQUEST_TIMEOUT` | 120 | API timeout in seconds |
| `FEEDBACK_SAMPLE_SIZE` | 8 | Sample size per beam (8 x 4 beams x 10 iters = 320 samples) |
| `STOCHASTIC_SAMPLING` | True | Different samples each iteration |
| `AGENT0_MAX_TURNS` | 10 | Max interview turns for Agent 0 |
| `AGENT1_PROMPT_PATH` | `agent1_v2.3.1_reflexion_only.md` | Active Agent 1 prompt |
