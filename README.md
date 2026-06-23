# LLM4MOF — Autonomous MOF Designer

An autonomous agent system that designs Metal–Organic Frameworks (MOFs) through iterative
hypothesis generation, constraint extraction, and feedback. An LLM (OpenAI GPT)
proposes MOF designs that are evaluated against real computational MOF databases (PORMAKE-assembled
H₂ markschemes, QMOF, hMOF) or, optionally, against a full live simulation pipeline on HPC
(PORMAKE → LAMMPS → Zeo++ → RASPA3).

## How it works

```
User inquiry  ("Maximize gravimetric H2 storage in mol/kg at 77K and 100 bar.")
      │
      ▼
[Agent 1] Hypothesis generator   — proposes metals, linkers, target pore geometry (multi-turn)
      │
      ▼
[Agent 2] Constraint extractor   — converts the hypothesis into searchable database filters
      │
      ▼
[Matchmaker]                     — finds matching MOFs / building-block assemblies in the database
      │
      ▼
[Sensitivity analyzer]           — scores how well the hypothesis performs
      │
      ▼
[Feedback generator]             — builds a blinded 4-beam diagnostic and feeds it back to Agent 1
      │
      └──────────────────────────  Agent 1 refines the hypothesis (loop)
```

The default feedback is a **4-beam diagnostic**: Full hypothesis (Z), Chemistry-only (A),
Metal-only (F), and a Global baseline — presented with anonymous MOF labels and generic headers so
Agent 1 cannot infer which database is active.

## Setup

### 1. Python environment

Requires **Python 3.10+**.

```bash
python -m venv llm4mof
# Windows:
llm4mof\Scripts\activate
# macOS/Linux:
source llm4mof/bin/activate

pip install -r requirements.txt
```

### 2. API keys

Copy `.env.example` to `.env` and fill in your key. `.env` is git-ignored and never committed.

```bash
OPENAI_API_KEY=...
```

- OpenAI: https://platform.openai.com/api-keys

The active model is set in `config.py` (`OPENAI_MODEL`).

### 3. Large data files (Git LFS)

Four large files ship via **Git LFS**. After cloning:

```bash
git lfs install
git lfs pull
```

| File | Size | Purpose |
|------|------|---------|
| `core/mof2zeo/ckpt/epoch=487-step=1039440.ckpt` | 78 MB | mof2zeo geometry-prediction model |
| `data/hMOF/hmof_index.json` | 66 MB | hMOF gas-adsorption database |
| `data/qmof_index_v2.json` | 26 MB | QMOF band-gap index |
| `data/qmof.csv` | 21 MB | QMOF property table |

The six `data/total_characteristics_h2_*.csv` PORMAKE H₂ markschemes are small and ship as normal
files (no LFS).

If the LFS quota is exhausted, the same files are archived on Zenodo with a DOI and SHA-256
checksums — see [`DATA.md`](DATA.md).

## Usage — database mode (no cluster required)

```bash
python run_experiment.py
```

You choose an inquiry from an 11-option menu: 4 PORMAKE H₂ targets (volumetric / gravimetric ×
5 / 100 bar), 4 hMOF gas-adsorption targets (CH₄, CO₂, Xe/Kr, H₂), 2 QMOF band-gap targets, or a
custom inquiry. Unit and pressure are carried naturally in the query text and routed to the correct
database automatically.

Non-interactive / batch:

```bash
python run_experiment.py --auto \
  --inquiry "Design a MOF to maximize volumetric H2 storage capacity at 77K and 100 bar." \
  --iterations 10 --database pormake
```

Useful flags: `--feedback-type N`, `--agent1-prompt <file>`,
`--pormake-unit {volumetric,molkg,gperL}`, `--pormake-pressure {5bar,100bar}`,
`--database {pormake,hmof,qmof}`, `--agent1-temp`, `--agent2-temp`.

### Output

Each run writes to `experiments/exp_YYYYMMDD_HHMM_{mode}/` with per-iteration `agent1_output.json`,
`agent2_output.json`, `beam_data.csv`, a sensitivity report, and the feedback text sent to Agent 1.

## Usage — live simulation mode (requires HPC)

`run_live_experiment.py` runs the full **PORMAKE → LAMMPS → Zeo++ → RASPA3** pipeline on an HPC
cluster (via SSH/qsub) instead of the pre-computed markschemes. This path **requires a configured
cluster** (RASPA3, LAMMPS, Zeo++, a PBS scheduler, and an SSH host entry); it is included for
transparency and reproducibility of the live-simulation results, and is not runnable on a laptop.

```bash
python run_live_experiment.py --hpc --pressure 5 \
  --inquiry "Design a MOF for high hydrogen storage at 5 bar and 77K" --iterations 10
```

Other live-mode flags: `--smoke` (quick validation), `--no-zeo`, `--adsorbate`, `--temperature`,
`--prepare` / `--collect` / `--resume` (step control), `--job-prefix`, `--node-prop`.

HPC settings (host, base dir, scheduler) are in the `LIVE SIMULATION CONFIGURATION` section of
`config.py`. The cluster-side scripts live in `hpc/`; local orchestration lives in `core/hpc/`.

> **Note:** the `hpc/` scripts are the authors' PBS/Torque batch scripts, provided for transparency
> and reproducibility — they are not runnable as-is elsewhere. Override the documented variables
> (`CONDA_ENV`, `SUBMIT_CMD`, `STATUS_CMD`, `LAMMPS_BIN`, `NODE_PROP`) and the `#PBS` directives to
> match your own cluster and scheduler.

## Configuration highlights (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `AGENT1_PROMPT_PATH` | `prompts/agent1_v3.0_production.md` | Active Agent 1 prompt |
| `AGENT2_PROMPT_PATH` | `prompts/agent2_v4.1.md` | Active Agent 2 prompt |
| `FEEDBACK_SAMPLE_SIZE` | 10 | Samples per beam |
| `STOCHASTIC_SAMPLING` | True | New samples each iteration |
| `STRATIFIED_SAMPLING` | True | Metal-stratified feedback sampling (env: `LLM4MOF_STRATIFIED_SAMPLING`) |
| `USE_MEMORY_LEDGER` | True | Facts-only design-memory prepend (env: `LLM4MOF_USE_MEMORY_LEDGER`) |

## Repository layout

```
.
├── run_experiment.py          # Database-mode entry point (interactive + batch)
├── run_live_experiment.py     # Live HPC simulation entry point
├── run_prepare_step.py        # HPC orchestration: prepare
├── run_collect_step.py        # HPC orchestration: collect
├── config.py                  # Paths, models, unit/pressure routing, toggles
├── setup.py  requirements.txt
├── core/                      # Runtime modules (agents, matchmakers, feedback, mof2zeo, simulation, hpc)
├── prompts/                   # Active Agent 1 / Agent 2 prompts
├── data/                      # Databases (4 large files via Git LFS)
├── hpc/                       # Cluster-side scripts (uploaded and run on HPC)
└── scripts/                   # build_canonical_db.py — rebuilds the shipped data files
```

## Code and data availability

- **Code** — this repository (MIT-licensed; see `LICENSE`).
- **Data** — the three evaluation databases (PORMAKE, hMOF, QMOF) and the mof2zeo model checkpoint
  ship in-repo via Git LFS and are also archived on Zenodo with a permanent DOI and SHA-256
  checksums. See [`DATA.md`](DATA.md) for the file manifest, integrity hashes, and third-party
  database citations.

See `PROVENANCE.md` for how this repository was derived. Licensed under MIT — see `LICENSE`.
