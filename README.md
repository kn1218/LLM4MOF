# LLM2POR: Autonomous MOF Designer (v2.5)

An autonomous agent system that designs Metal-Organic Frameworks (MOFs) through iterative hypothesis generation, constraint extraction, and database-driven feedback. The system uses LLMs (GPT / Gemini) to propose MOF designs and evaluates them against real computational databases (QMOF, hMOF, PORMAKE).

**v2.5 status:** Agent 1 prompt locked at `agent1_v2.2.9.md` after a three-way ablation (v2.2.9 / v2.3.0 / v2.3.1) showed no measurable improvement from the v2.3.x reasoning rules. Retired prompts are retained by the maintainer locally for reproducibility of pre-v2.5 batches but are not shipped with this GitHub release. See [Version History](#version-history) for details.

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
    Creates structured 4-beam feedback for Agent 1
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
- `data/qmof_index_v2.json` (25 MB) — 20,373 QMOF MOFs for band gap mode
- `data/hMOF/hmof_index.json` (50 MB) — 51,163 hypothetical MOFs for gas adsorption mode
- `data/qmof.csv` (21 MB) — QMOF property database
- `data/total_characteristics&name_singleonly_20251203.csv` (1 MB) — PORMAKE master database (100 bar variant)
- `data/total_characteristics_h2_5bar_77K.csv` (1 MB) — PORMAKE master database (5 bar variant)

## Usage

```bash
python run_experiment.py
```

### Experiment Modes

You will be prompted to choose:

**[1] With Agent 0** — An AI consultant interviews you to clarify your design requirements before passing them to the hypothesis generator.

**[2] Direct Inquiry** (recommended) — Skip the interview and go straight to hypothesis generation with a preset or custom inquiry.

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

**hMOF metal constraint:** The hMOF database (Snurr group, 51K hypothetical MOFs) contains only Cu (25.6%), V (7.2%), Zn (62.1%), and Zr (4.7%) metal nodes. All other metals have <0.4% representation. The metal constraint is included in the user query (not the system prompt) so the LLM knows what materials are available without revealing database identity. This is constraint specification (what reagents are available), not bias (which to prefer).

### Feedback Types

After each iteration, choose a feedback type:

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
- `raw_user_input.txt` — Your original inquiry
- `experiment_log.txt` — Full run log
- `iteration_N/` — Per-iteration outputs (hypothesis, constraints, sensitivity reports)

---

## Core Architecture

### Constraint Engine: Branched Hypothesis Matching

Agent 1 frequently proposes alternative linker strategies in its hypotheses, e.g. *"use pyridine dicarboxylate OR ether-containing aromatics OR azolate linkers"*. The constraint engine supports this via the `linker_branches` schema in Agent 2's output: an array of alternative search branches with **AND-within-branch** semantics combined with **OR-between-branches**.

**Example — three alternative linker families expressed as branches:**

```json
{
  "linker_query": {
    "functional_groups": [],
    "linker_branches": [
      {"description": "pyridine dicarboxylate", "required_tags": ["Pyridine", "Carboxyl"]},
      {"description": "ether aromatic",         "required_tags": ["Ether", "Aromatic"]},
      {"description": "azolate",                "required_tags": ["Azolate"]}
    ]
  }
}
```

Each branch is searched independently. A linker matches if it satisfies ALL tags within ANY single branch. Empty `linker_branches` is the no-op default.

PORMAKE strips coordination tags (`Carboxyl`, `Carbonyl`, `Phosphonate`, `Sulfonate`) from branches before matching, because in PORMAKE's grammar these groups live on the **Node** SBU, not the Edge linker. QMOF and hMOF do not strip — they index whole MOFs where these tags are valid linker-level properties.

### 4-Beam Diagnostic Feedback

The default feedback type runs four parallel database searches per iteration and presents them side by side so Agent 1 can perform causal attribution:

| Database | Beam 1 | Beam 2 | Beam 3 | Beam 4 | Diagnostic Question |
|----------|--------|--------|--------|--------|---------------------|
| PORMAKE / hMOF | Full hypothesis (Z) | Chemistry only (A) | Metal only (F) | Global baseline | Is geometry or chemistry the bottleneck? |
| QMOF | Full hypothesis (Z) | Metal control (F) | Linker control (G) | Global baseline | Is it metal d-electrons or linker conjugation? |

QMOF gets a different beam design because it has no geometry gate — Beams 1 and 2 would be identical under the PORMAKE/hMOF design, wasting a diagnostic slot. The QMOF-specific design isolates metal vs. linker electronic contributions instead.

**Agent 1 blinding:** Chemistry profiles use anonymous `MOF-1, MOF-2, ...` labels (not the internal `N419+E12` / `qmof-XXX` / `hmof-XXX` IDs). Beam headers say "4-BEAM DIAGNOSTIC" with no database name. The orchestrator's prompts say "analyze beams as labeled" without specifying which beam configuration to expect. This prevents Agent 1 from inferring which database is active and hardcoding strategies.

### Agent 1 Prompt: Locked at v2.2.9

Agent 1 uses `prompts/agent1_v2.2.9.md` for all three database modes (PORMAKE, QMOF, hMOF). The prompt encodes four core principles:

1. **Mechanism-Grounded Reasoning** — every component choice must be justified by chemical rationale, not popularity. ("Use Zr because it's popular" is forbidden.)
2. **Causal Hierarchy / Inverse Design** — start from the Performance Goal, derive the required Geometry, then select Components.
3. **Stateless Execution** — explicitly list all metals and functional groups in every iteration. The pipeline has no memory of previous hypotheses other than what is in the multi-turn conversation buffer.
4. **Scientific Skepticism / Radical Pivots** — if performance plateaus for 3 iterations, abandon the current chemistry entirely and pivot to a fundamentally different mechanism.

The prompt's output schema has 8 fields: `meta_cognition.reasoning`, `target_application`, `hypothesis_mechanism`, `ideal_pore_geometry`, `node_composition`, `linker_composition`, `novelty_justification`, `lesson_learnt` — all rich text.

**Memory model:** Multi-turn conversation only. Agent 1 sees its full conversation history (all prior hypotheses and feedback) via the `LLMClient.messages` buffer. There is no external "Scientific Journal" injection — that mechanism existed in earlier versions and was removed in v2.5 (it duplicated information already in the multi-turn context and cost ~7K tokens per iteration by iteration 10).

For the deep technical reference on every component (matchmaker filter pipelines, AND/OR/NOT/branch logic, the schema-to-logic contract, worked examples), see `docs/REPORT_2_System_Logic_and_Methodology.md`.

---

## Version History

### v2.5 (current)

- **Agent 1 prompt locked at v2.2.9.** Three-way ablation (v2.2.9 / v2.3.0 / v2.3.1) found no measurable improvement from the v2.3.x "Reasoning Strategy" rules (Rules A-F) or the v2.3.1 reflexion-only structured output. Retired prompts moved out of the public tree (retained locally by the maintainer for batch reproducibility, not shipped with the GitHub release).
- **hMOF metal constraint** added to all 4 hMOF preset queries (Cu/V/Zn/Zr only). Eliminated zero-match failures (H2_Storage went from 80% zero-match rate to 0%) and improved Top-1 across all hMOF targets.
- **Scientific Journal removed.** The cumulative `{SCIENTIFIC_JOURNAL}` placeholder injection is gone. Agent 1 relies on the multi-turn conversation buffer plus the iteration feedback string.
- **Unified 4-beam feedback for QMOF.** Merged the separate QMOF feedback branch into the unified design. For QMOF, Beam 1 ≈ Beam 2 naturally (no geometry gate), which correctly signals "geometry irrelevant" without a separate beam layout.
- **hMOF sensitivity crash fix** — `core/sensitivity_analyzer.py` was falling through to PORMAKE's node/edge logic on hMOF and crashing on `connectivity: None`. Killed multiple iterations across the previous batch (CH4@iter6, CO2@iter8, XeKr@iter1) until fixed.

### v2.4

- **Chemistry-first feedback** with the 4-beam diagnostic and database-aware variants.
- **Agent 1 blinding** — anonymous MOF labels, generic beam headers, unified footer messages, generic prompt instructions. Prevents Agent 1 from inferring database identity.
- **Bug fixes:** QMOF `"Any"` metal guard added (matched PORMAKE/hMOF behavior); SA negative-tag substring matching replaced with shared `check_negative_tags()` from `constraint_utils.py` for consistency with the matchmakers.

### v2.3.x — Retired ablations

Three Agent 1 prompt variants tested in head-to-head comparison and rejected:

- **v2.3.0** (`agent1_v2.3.0.md` + `_qmof.md` + `_hmof.md`): Added a "Reasoning Strategy: Beat Bayesian Optimization" section with six explicit rules (pattern extraction over anecdotes, hypothesis falsification, exploration budget management, beam comparison rules, etc.). Plus per-database variants for QMOF and hMOF.
- **v2.3.1** (`agent1_v2.3.1_reflexion_only.md`): The structured output format from v2.3.0 (4-field `meta_cognition`, 4-field `lesson_learnt`) but **without** the explicit rules. Designed to isolate the "format effect" from the "rules effect".

All three were locally available simultaneously and chosen at runtime via a `set_agent1_strategy()` registry. Result: no measurable improvement over v2.2.9 for either v2.3.0 or v2.3.1. The registry, the strategy switch, and the per-database routing have all been removed in v2.5. The retired prompt files are retained locally by the maintainer (not in this GitHub release) solely so pre-v2.5 batch experiments can be rerun for reproducibility.

### v2.x — Branched Hypothesis Matching

The `linker_branches` schema was introduced in earlier v2.x work to fix an information-loss bug where Agent 2's flat AND/OR could only express the lowest-common-denominator tag across alternative linker strategies. Agent 1 frequently proposes 2-4 alternative chemistries per iteration; before branched matching, all 178 sampled `agent2_output.json` files collapsed these into a single tag like `Aromatic`, which matched 52-72% of every database and produced essentially unfiltered results. The OR-of-ANDs branch schema preserves the full alternative structure. This is now stable and used by all three matchmakers.

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
│   ├── agent2_handler.py      # Constraint Extractor (stateless)
│   ├── constraint_utils.py    # Tag/ontology parsing + AND/OR/NOT/branch logic
│   ├── feedback_generator.py  # Structured feedback for Agent 1
│   ├── matchmaker.py          # PORMAKE component assembly matching
│   ├── qmof_matchmaker.py     # QMOF whole-MOF matching
│   ├── hmof_matchmaker.py     # hMOF whole-MOF matching
│   ├── sensitivity_analyzer.py # Performance evaluation engine (22 filter sets)
│   ├── memory_manager.py      # Experiment state persistence
│   ├── name_resolver.py       # Building block ID-to-name resolver
│   └── llm_client.py          # Unified OpenAI/Gemini API client
├── prompts/                   # Active LLM system prompts
│   ├── agent0_v3.md           # Problem Consultant prompt
│   ├── agent1_v2.2.9.md       # ACTIVE Agent 1 prompt (locked v2.5)
│   └── agent2_v4.0.md         # Constraint Extractor (with linker_branches schema)
└── data/                      # Databases (large files via Git LFS)
    ├── pormake_bb_dictionary_v5.json
    ├── pormake_topo_dictionary_v3.json
    ├── unified_ontology.json
    ├── qmof.csv
    ├── qmof_ids_with_topology.txt
    ├── qmof_index_v2.json
    ├── total_characteristics&name_singleonly_20251203.csv
    ├── total_characteristics_h2_5bar_77K.csv
    └── hMOF/
        └── hmof_index.json
```

The repository ships only the production runtime. All research artifacts (paper drafts, presentations, analysis scripts, figure outputs, comparison-study results, retired prompts, archived experiment batches) live under a local-only `research/` tree on the maintainer's machine and are excluded from this GitHub release via `.gitignore`. Experiment outputs from `python run_experiment.py` are written to a local `experiments/` directory, which is also gitignored.

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MAX_OUTPUT_TOKENS` | 32000 | Max tokens for LLM response |
| `LLM_REQUEST_TIMEOUT` | 120 | API timeout in seconds |
| `FEEDBACK_SAMPLE_SIZE` | 8 | Sample size per beam (8 × 4 beams × 10 iters = 320 samples) |
| `STOCHASTIC_SAMPLING` | True | Different samples each iteration |
| `AGENT0_MAX_TURNS` | 10 | Max interview turns for Agent 0 |
| `AGENT1_PROMPT_PATH` | `prompts/agent1_v2.2.9.md` | Active Agent 1 prompt (locked at v2.2.9 for v2.5) |
| `AGENT2_PROMPT_PATH` | `prompts/agent2_v4.0.md` | Active Agent 2 prompt |
