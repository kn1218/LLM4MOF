# LLM2POR: Autonomous MOF Designer v3

An autonomous agent system that designs Metal-Organic Frameworks (MOFs) through iterative hypothesis generation, constraint extraction, and database-driven feedback. The system uses LLMs (GPT / Gemini) to propose MOF designs and evaluates them against real computational databases (QMOF, hMOF, PORMAKE).

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
OPENAI_API_KEY=sk-proj-...   # Your OpenAI API key
GEMINI_API_KEY=AIza...        # Your Gemini API key (if using Gemini)
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

**[1] With Agent 0** (recommended) - An AI consultant interviews you to clarify your design requirements before passing them to the hypothesis generator.

**[2] Direct Inquiry** - Skip the interview and go straight to hypothesis generation with a preset or custom inquiry.

### Available Inquiry Types

| # | Category | Inquiry | Database |
|---|----------|---------|----------|
| 1 | H2 Storage | High capacity hydrogen storage at 77K | PORMAKE |
| 2 | Band Gap | Optimal band gap for visible-light water splitting | QMOF |
| 3 | Band Gap | Band gap between 3-4 eV | QMOF |
| 4 | Band Gap | Band gap for UV activity | QMOF |
| 5 | Band Gap | Band gap below 0.1 eV | QMOF |
| 6 | Band Gap | Band gap above 4 eV | QMOF |
| 7 | CH4 Storage | High methane storage at 298K | hMOF |
| 8 | CO2 Capture | CO2 capture at low pressure | hMOF |
| 9 | Xe/Kr Selectivity | High Xe/Kr selectivity | hMOF |
| 10 | H2 Storage | High H2 uptake at 100 bar 77K | hMOF |
| 11 | Custom | Type your own design inquiry | Auto-detected |

### Feedback Types

After each iteration, choose a feedback strategy:

| # | Type | Description |
|---|------|-------------|
| 1 | 3-Beam Diagnostic | Tests complete hypothesis against controls (default) |
| 2 | Universe Baseline | Samples across all DB; useful when 0 candidates found |
| 3 | Geometric Optimizer | Tests random vs constrained geometry |
| 4 | Chemical Pivot | Tests random metals vs your geometry |
| 5 | Best vs Worst | Stratified sampling to find patterns |
| 6 | Hypothesis Validation | Tests only the complete hypothesis block |
| 7 | Virtual Synthesis | Lab synthesis simulation |

Type `quit` at any feedback prompt to end the experiment.

### Output

Results are saved to `experiments/exp_YYYYMMDD_HHMM_{mode}/`:
- `raw_user_input.txt` - Your original inquiry
- `experiment_log.txt` - Full run log
- `iteration_N/` - Per-iteration outputs (hypothesis, constraints, sensitivity reports)

## Project Structure

```
.
├── run_experiment.py          # Entry point
├── config.py                  # All configuration and data paths
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not in git)
├── core/                      # Runtime modules
│   ├── agent0_handler.py      # Problem Consultant (interview)
│   ├── agent1_handler.py      # Hypothesis Generator (multi-turn)
│   ├── agent2_handler.py      # Constraint Extractor (stateless)
│   ├── constraint_utils.py    # Tag/ontology parsing utilities
│   ├── feedback_generator.py  # Structured feedback for Agent 1
│   ├── hmof_matchmaker.py     # hMOF direct MOF matching
│   ├── llm_client.py          # Unified OpenAI/Gemini API client
│   ├── matchmaker.py          # PORMAKE component assembly matching
│   ├── memory_manager.py      # Experiment state persistence
│   ├── name_resolver.py       # Building block ID-to-name resolver
│   ├── qmof_matchmaker.py     # QMOF direct MOF matching
│   └── sensitivity_analyzer.py # Performance evaluation engine
├── prompts/                   # LLM system prompts
│   ├── agent0_v3.md
│   ├── agent1_v2.2.9.md
│   └── agent2_v4.0.md
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
| `FEEDBACK_SAMPLE_SIZE` | 10 | Sample size for 3-Beam Diagnostic |
| `STOCHASTIC_SAMPLING` | True | Different samples each iteration |
| `AGENT0_MAX_TURNS` | 10 | Max interview turns for Agent 0 |
