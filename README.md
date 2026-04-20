# LLM2POR: Autonomous MOF Designer (v2.5)

An autonomous agent system that designs Metal-Organic Frameworks (MOFs) through iterative hypothesis generation, constraint extraction, and database-driven feedback. The system uses LLMs (GPT / Gemini) to propose MOF designs and evaluates them against real computational databases (QMOF, hMOF, PORMAKE).

**v2.5 status:** Agent 1 prompt is `agent1_v2.2.9.2.md` — a database-agnostic prompt with concrete examples and incremental constraint discipline. Agent 1 receives the raw user query with no hidden injection; unit and pressure info are carried naturally in the query text. Retired prompts are in `prompts/_archive/` locally.

## Changelog

### BB Dictionary v5 → v6 (2026-04-20)

Resolved all 333 `has_open_metal_site: null` values in `pormake_bb_dictionary_v6.json`:

- **314 organic BBs** (no metals): set to `false` — purely organic linkers cannot have open metal sites.
- **8 metal-containing BBs** set to `true` (coordinatively unsaturated):
  E56 (Cd, 2N), E74 (Rh, 2-coord), E103 (2×Ir), E105 (Ni, 2O), E172 (Mn, 2O), E191 (Ni, 2N), E207 (Cu₂ paddle-wheel), E230 (Cu, 2N).
- **11 metal-containing BBs** set to `false` (coordination sphere saturated):
  E54 (Ag linear), E60 (Cu N4 macrocycle), E99 (Ag linear), E109 (Ni octahedral 6-coord), E129 (Cu 4-coord + NCS), E133 (Mn 5-coord + Cl), E169 (Ag linear), E197 (Cu 6-coord), E208 (In 5-coord), E210 (Cu 4N), E213 (B tetrahedral).

**Final distribution:** 241 true / 626 false / 0 null (867 total). v5 archived in `data/`.

## How It Works

```
User Inquiry ("Design a MOF to maximize gravimetric H2 storage in mol/kg at 77K and 100 bar.")
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
# Install git-lfs if needed
conda install -c conda-forge git-lfs

# Download large files
git lfs pull
```

This downloads the large database files (~175 MB total):
- `data/qmof_index_v2.json` (25 MB) — 20,373 QMOF MOFs for band gap mode
- `data/hMOF/hmof_index.json` (50 MB) — 51,163 hypothetical MOFs for gas adsorption mode
- `data/qmof.csv` (21 MB) — QMOF property database
- `core/mof2zeo/ckpt/epoch=478-step=213634.ckpt` (76 MB) — mof2zeo geometry prediction model
- PorMake H2 markscheme CSVs (~1 MB each, 6 files):

| CSV File | Pressure | Unit |
|----------|----------|------|
| `total_characteristics_h2_100bar_77K.csv` | 100 bar | cm3(STP)/cm3 (volumetric) |
| `total_characteristics_h2_100bar_77K_mol_kg.csv` | 100 bar | mol/kg (gravimetric) |
| `total_characteristics_h2_100bar_77K_gperL.csv` | 100 bar | g/L (gravimetric) |
| `total_characteristics_h2_5bar_77K.csv` | 5 bar | cm3(STP)/cm3 (volumetric) |
| `total_characteristics_h2_5bar_77K_mol_kg.csv` | 5 bar | mol/kg (gravimetric) |
| `total_characteristics_h2_5bar_77K_gperL.csv` | 5 bar | g/L (gravimetric) |

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
| **PorMake H2 (100 bar, 77K)** |||
| 1 | Volumetric | Maximize volumetric H2 storage capacity (cm3/cm3) | PORMAKE |
| 2 | Gravimetric | Maximize gravimetric H2 storage capacity (mol/kg) | PORMAKE |
| 3 | Gravimetric | Maximize H2 storage density (g/L) | PORMAKE |
| **PorMake H2 (5 bar, 77K)** |||
| 4 | Volumetric | Maximize volumetric H2 storage capacity (cm3/cm3) | PORMAKE |
| 5 | Gravimetric | Maximize gravimetric H2 storage capacity (mol/kg) | PORMAKE |
| 6 | Gravimetric | Maximize H2 storage density (g/L) | PORMAKE |
| **QMOF Band Gap** |||
| 7 | Band Gap | Optimal band gap for visible-light water splitting | QMOF |
| 8 | Band Gap | Band gap between 3-4 eV | QMOF |
| 9 | Band Gap | Band gap for UV activity | QMOF |
| 10 | Band Gap | Band gap below 0.1 eV | QMOF |
| 11 | Band Gap | Band gap above 4 eV | QMOF |
| **hMOF Gas Adsorption** |||
| 12 | CH4 Storage | High methane storage at 298K, 35 bar. **Cu/V/Zn/Zr only.** | hMOF |
| 13 | CO2 Capture | CO2 capture at 2.5 bar, 298K. **Cu/V/Zn/Zr only.** | hMOF |
| 14 | Xe/Kr | High Xe/Kr selectivity at 1 bar. **Cu/V/Zn/Zr only.** | hMOF |
| 15 | H2 Storage | High H2 uptake at 100 bar, 77K. **Cu/V/Zn/Zr only.** | hMOF |
| 16 | Custom | Type your own design inquiry | Auto-detected |

**Unit-aware queries:** PorMake queries carry unit and pressure information naturally in the query text (e.g. "maximize gravimetric H2 storage capacity in mol/kg at 77K and 100 bar"). The system detects unit keywords and routes to the correct CSV automatically. No hidden injection to Agent 1.

**hMOF metal constraint:** The hMOF database (Snurr group, 51K hypothetical MOFs) contains only Cu (25.6%), V (7.2%), Zn (62.1%), and Zr (4.7%) metal nodes. The metal constraint is included in the user query so the LLM knows what materials are available without revealing database identity.

### Batch / Automated Mode

For non-interactive runs (scripting, ablation studies):

```bash
python run_experiment.py --auto \
  --inquiry "Design a MOF to maximize volumetric H2 storage capacity at 77K and 100 bar." \
  --iterations 10 \
  --mode direct \
  --database pormake
```

Additional CLI flags for ablation control:
- `--agent1-prompt <filename>` — Override Agent 1 prompt file
- `--pormake-unit {volumetric,molkg,gperL}` — Force unit variant
- `--pormake-pressure {5bar,100bar}` — Force pressure variant
- `--database {pormake,hmof,qmof}` — Force database routing

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
- `experiment_meta.json` — Run metadata (prompt, query, CSV path, unit, model)
- `experiment_log.txt` — Full run log
- `iteration_N/` — Per-iteration outputs:
  - `agent1_output.json` — Agent 1 hypothesis
  - `agent2_output.json` — Agent 2 constraints
  - `beam_data.csv` — All beam candidates with performance values
  - `Sensitivity_Report_iterN.csv` — 22-filter-set evaluation
  - `feedback_selected.txt` — Feedback text sent to Agent 1

---

## Core Architecture

### Agent Pipeline (Clean Pipe)

Agent 1 receives the raw user query with no hidden injection. Unit, pressure, and application information are carried in the query text itself. The handler is a pass-through:

```
user_inquiry = "Design a MOF to maximize gravimetric H2 storage in mol/kg at 77K and 100 bar."
    -> Agent 1 reads unit info from query text naturally
    -> Agent 2 extracts constraints (stateless, single-turn)
    -> Matchmaker finds candidates in the unit-matched CSV
    -> Feedback labels use correct unit from config
```

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

**Agent 1 blinding:** Chemistry profiles use anonymous `MOF-1, MOF-2, ...` labels (not the internal `N419+E12` / `qmof-XXX` / `hmof-XXX` IDs). Beam headers say "4-BEAM DIAGNOSTIC" with no database name. This prevents Agent 1 from inferring which database is active and hardcoding strategies.

### Agent 1 Prompt: v2.2.9.2

Agent 1 uses `prompts/agent1_v2.2.9.2.md` for all three database modes (PORMAKE, QMOF, hMOF). The prompt encodes four core principles:

1. **Mechanism-Grounded Reasoning** — every component choice must be justified by chemical rationale, not popularity.
2. **Causal Hierarchy / Inverse Design** — start from the Performance Goal, derive the required Geometry, then select Components.
3. **Stateless Execution** — explicitly list all metals and functional groups in every iteration. The pipeline has no memory other than the multi-turn conversation buffer.
4. **Scientific Skepticism / Radical Pivots** — if performance plateaus, abandon the current chemistry and pivot to a fundamentally different mechanism.

The prompt's output schema has 8 fields: `meta_cognition.reasoning`, `target_application`, `hypothesis_mechanism`, `ideal_pore_geometry`, `node_composition`, `linker_composition`, `novelty_justification`, `lesson_learnt` — all rich text.

**Memory model:** Multi-turn conversation only. Agent 1 sees its full conversation history (all prior hypotheses and feedback) via the `LLMClient.messages` buffer. No external injection.

---

## Version History

### v2.5 (current)

- **Agent 1 prompt updated to v2.2.9.2.** Database-agnostic with concrete examples and incremental constraint discipline. Chemistry hint annotations on geometric descriptors removed to prevent database-identity leaks.
- **Unit-diversified query system.** Interactive menu expanded to 16 options covering 6 PorMake H2 variants (2 pressures x 3 units), 5 QMOF bandgap queries, and 4 hMOF gas adsorption targets. Unit/pressure info carried naturally in query text.
- **Clean agent pipeline.** Agent 1 handler simplified to pass-through — sends raw user query to LLM with no appended guidance or injection.
- **6 PorMake H2 markscheme CSVs** — per-condition files for volumetric (cm3/cm3), gravimetric (mol/kg), and mass-density (g/L) at both 100 bar and 5 bar.
- **hMOF metal constraint** added to all 4 hMOF preset queries (Cu/V/Zn/Zr only). Eliminated zero-match failures.
- **Scientific Journal removed.** Agent 1 relies on the multi-turn conversation buffer plus iteration feedback.
- **Matchmaker null-safety** — defaults connectivity to [3,4,6,8,12] when Agent 2 returns None instead of crashing.
- **Unified 4-beam feedback for QMOF.**

### v2.4

- **Chemistry-first feedback** with the 4-beam diagnostic and database-aware variants.
- **Agent 1 blinding** — anonymous MOF labels, generic beam headers, unified footer messages.

### v2.3.x — Retired ablations

Three Agent 1 prompt variants tested in head-to-head comparison and rejected:

- **v2.3.0**: Added "Reasoning Strategy: Beat Bayesian Optimization" rules + per-database variants.
- **v2.3.1**: Structured output format from v2.3.0 without the explicit rules (isolating format effect from rules effect).

No measurable improvement over v2.2.9 for either variant. Retired prompts retained locally for reproducibility.

### v2.x — Branched Hypothesis Matching

The `linker_branches` schema was introduced to fix an information-loss bug where Agent 2's flat AND/OR collapsed alternative linker strategies into a single tag. The OR-of-ANDs branch schema preserves the full alternative structure and is used by all three matchmakers.

---

## Project Structure

```
.
├── run_experiment.py          # Interactive + batch entry point (16-option menu)
├── run_live_experiment.py     # Live simulation pipeline (RASPA3 + Zeo++)
├── run_prepare_step.py        # HPC orchestration: prepare step
├── run_collect_step.py        # HPC orchestration: collect step
├── config.py                  # Configuration, data paths, unit/pressure routing
├── setup.py                   # Package installer
├── requirements.txt           # Python dependencies
├── setup.py                   # pip install -e .
├── .env                       # API keys (not in git)
├── core/                      # Runtime modules
│   ├── agent1_handler.py      # Hypothesis Generator (multi-turn, clean pipe)
│   ├── agent2_handler.py      # Constraint Extractor (stateless)
│   ├── agent3_260324.py       # Geometry Predictor (mof2zeo)
│   ├── constraint_utils.py    # Tag/ontology parsing + AND/OR/NOT/branch logic
│   ├── feedback_generator.py  # Structured feedback for Agent 1
│   ├── matchmaker.py          # PORMAKE component assembly matching
│   ├── qmof_matchmaker.py     # QMOF whole-MOF matching
│   ├── hmof_matchmaker.py     # hMOF whole-MOF matching
│   ├── sensitivity_analyzer.py # Performance evaluation engine (22 filter sets)
│   ├── memory_manager.py      # Experiment state persistence
│   ├── name_resolver.py       # Building block ID-to-name resolver
│   ├── qmof_matchmaker.py     # QMOF direct MOF matching
│   ├── sensitivity_analyzer.py # Performance evaluation engine
│   ├── run_simulation.py      # Full simulation pipeline
│   ├── filter_candidate.py     # mof2zeo candidate filtering
│   ├── llm_client.py          # Unified OpenAI/Gemini API client
│   ├── mof2zeo/               # Geometry prediction model
│   │   ├── ckpt/              # Model checkpoint (LFS)
│   │   ├── data/               # Training data
│   │   └── scaler/             # Feature scalers
│   └── simulation/            # Simulation pipeline
│       ├── generate_mofs.py   # CIF generation from PORMAKE
│       ├── opt/                # LAMMPS optimization
│       └── gcmc/              # RASPA3 GCMC simulations
├── prompts/                   # Active LLM system prompts
│   ├── agent1_v2.2.9.2.md     # ACTIVE Agent 1 prompt
│   └── agent2_v4.0.md         # Constraint Extractor (with linker_branches schema)
└── data/                      # Databases (large files via Git LFS)
    ├── pormake_bb_dictionary_v6.json
    ├── pormake_topo_dictionary_v3.json
    ├── unified_ontology.json
    ├── qmof.csv
    ├── qmof_ids_with_topology.txt
    ├── qmof_index_v2.json
    ├── total_characteristics_h2_100bar_77K.csv
    ├── total_characteristics_h2_100bar_77K_mol_kg.csv
    ├── total_characteristics_h2_100bar_77K_gperL.csv
    ├── total_characteristics_h2_5bar_77K.csv
    ├── total_characteristics_h2_5bar_77K_mol_kg.csv
    ├── total_characteristics_h2_5bar_77K_gperL.csv
    └── hMOF/
        └── hmof_index.json
```

The repository ships only the production runtime. All research artifacts (paper drafts, presentations, analysis scripts, figure outputs, retired prompts, experiment batches) live under local-only directories and are excluded from this GitHub release via `.gitignore`.

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MAX_OUTPUT_TOKENS` | 32000 | Max tokens for LLM response |
| `LLM_REQUEST_TIMEOUT` | 120 | API timeout in seconds |
| `FEEDBACK_SAMPLE_SIZE` | 8 | Sample size per beam (8 x 4 beams x 10 iters = 320 samples) |
| `STOCHASTIC_SAMPLING` | True | Different samples each iteration |
| `AGENT1_PROMPT_PATH` | `prompts/agent1_v2.2.9.2.md` | Active Agent 1 prompt |
| `AGENT2_PROMPT_PATH` | `prompts/agent2_v4.0.md` | Active Agent 2 prompt |
