# LLM4MOF: Interpretable Inverse Design of MOFs with LLM Agents

A closed-loop multi-agent framework for inverse design of Metal–Organic Frameworks (MOFs).
Language-model agents (OpenAI GPT) propose interpretable design hypotheses, translate them into
searchable constraints, and refine them over ten autonomous iterations against either precomputed
MOF property databases (**database mode**) or a full live-simulation pipeline on HPC
(**discovery mode**: PORMAKE → LAMMPS → Zeo++ → RASPA3).

This repository accompanies the preprint *"Interpretable Inverse Design of Metal–Organic Frameworks
with Large Language Model Agents"* (Nam, Han, Kim).

## How it works
User query  ("Maximize gravimetric H2 storage in mol/kg at 77 K and 100 bar.")

│

▼

[Agent 1] Hypothesis generator   — proposes metal nodes, linkers, target pore geometry (multi-turn)

│

▼

[Agent 2] Constraint translator  — converts the hypothesis into searchable database constraints

│

▼

[Matchmaker]                     — applies constraints; organizes candidates into four diagnostic beams

│

▼

[Hypothesis testing]             — retrieves properties (database mode) or runs live simulation (discovery mode)

│

▼

[Feedback generator]             — builds blinded beam feedback + memory ledger, returns it to Agent 1

│

└──────────────────────────  Agent 1 refines the hypothesis (loop ×10)

The Matchmaker organizes candidates into a **4-beam diagnostic** that isolates which design axis drives
performance:

| Beam | Name | Constraints applied |
|------|------|---------------------|
| Beam 1 | Full hypothesis | Full hypothesis (geometry + chemistry + metal) |
| Beam 2 | Metal–linker chemistry | Chemistry only; geometry window removed |
| Beam 3 | Metal only | Metal only; linker and geometry unconstrained |
| Beam 4 | Random baseline | Unconstrained sampling from the full design space |

Beams are presented to Agent 1 under anonymized labels (internally `Z` / `A` / `F` / baseline) with
generic headers, so Agent 1 cannot infer which database is active or look up structures externally.

## Setup

### 1. Python environment

Requires **Python 3.10+**. Database mode needs only pure-Python packages, so any of `venv` + pip, `uv`,
or `conda` works — pick one:

```bash
# --- venv + pip ---
python -m venv llm4mof
llm4mof\Scripts\activate        # Windows   (macOS/Linux: source llm4mof/bin/activate)
pip install -r requirements.txt

# --- or uv (fast) ---
uv venv && uv pip install -r requirements.txt

# --- or conda ---
conda create -n llm4mof python=3.11 -y && conda activate llm4mof
pip install -r requirements.txt
```

`requirements.txt` is **database-mode only** and installs cleanly with no compiler or cluster.
Discovery / live-simulation mode needs extra Python packages **and** the external RASPA3 + LAMMPS
engines — see [`requirements-live.txt`](requirements-live.txt) and the "discovery mode" section below.
For that path **conda is recommended**, since RASPA3/LAMMPS are compiled binaries best obtained from
conda-forge or HPC modules (uv/pip cannot provide them).

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
| `core/mof2zeo/ckpt/epoch=487-step=1039440.ckpt` | 78 MB | MOF2Zeo geometry-prediction model |
| `data/hMOF/hmof_index.json` | 66 MB | hMOF gas-adsorption database |
| `data/qmof_index_v2.json` | 26 MB | QMOF band-gap index |
| `data/qmof.csv` | 21 MB | QMOF property table |

The six `data/total_characteristics_h2_*.csv` PORMAKE H₂ property tables are small and ship as normal
files (no LFS).

If the LFS quota is exhausted, the same files are archived on Zenodo with a DOI and SHA-256
checksums — see [`DATA.md`](docs/DATA.md).

## Usage — database mode (no cluster required)

```bash
python run_experiment.py
```

You choose a query from an 11-option menu: 4 PORMAKE H₂ targets (volumetric / gravimetric ×
5 / 100 bar), 4 hMOF gas-adsorption targets (CH₄, CO₂, Xe/Kr, H₂), 2 QMOF band-gap targets, or a
custom query. Unit and pressure are carried in the query text and routed to the correct database
automatically.

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
`agent2_output.json`, `beam_data.csv`, a feedback report, and the feedback text sent to Agent 1.

## Usage — discovery mode / live simulation (requires HPC)

`run_live_experiment.py` runs the full **PORMAKE → LAMMPS → Zeo++ → RASPA3** pipeline on an HPC
cluster (via SSH/qsub) instead of the precomputed property tables. This path **requires a configured
cluster** (RASPA3, LAMMPS, Zeo++, a PBS scheduler, and an SSH host entry); it is included for
transparency and reproducibility of the discovery-mode results, and is not runnable on a laptop.

Install the extra dependencies first (on top of `requirements.txt`):

```bash
pip install -r requirements.txt -r requirements-live.txt
# plus the external engines RASPA3 + LAMMPS — see requirements-live.txt
```

```bash
python run_live_experiment.py --hpc --pressure 5 \
  --inquiry "Design a MOF for high hydrogen storage at 5 bar and 77K" --iterations 10
```

Other live-mode flags: `--smoke` (quick validation), `--no-zeo`, `--adsorbate`, `--temperature`,
`--prepare` / `--collect` / `--resume` (step control), `--job-prefix`, `--node-prop`.

HPC settings (host, base dir, scheduler) are in the `LIVE SIMULATION CONFIGURATION` section of
`config.py`. The cluster-side scripts live in `hpc/`; local orchestration lives in `core/hpc/`.

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
.

├── run_experiment.py          # Database-mode entry point (interactive + batch)

├── run_live_experiment.py     # Live HPC simulation entry point (discovery mode)

├── config.py                  # Paths, models, unit/pressure routing, toggles

├── setup.py  requirements.txt  requirements-live.txt

├── core/                      # Runtime modules (agents, matchmaker, feedback, mof2zeo, simulation, hpc)

├── prompts/                   # Active Agent 1 / Agent 2 prompts

├── data/                      # Databases (large files via Git LFS)

├── hpc/                       # Cluster-side scripts + HPC step helpers (run_prepare_step / run_collect_step)

├── scripts/                   # build_canonical_db.py — rebuilds the shipped data files

├── paper/                     # Publication figures + figure-to-data map

└── docs/                      # DATA.md (data manifest) · PROVENANCE.md

## Code and data availability

- **Code** — this repository (MIT-licensed; see `LICENSE`).
- **Data** — the three evaluation databases (PORMAKE, hMOF, QMOF) and the MOF2Zeo model checkpoint
  ship in-repo via Git LFS and are also archived on Zenodo with a permanent DOI and SHA-256
  checksums. See [`DATA.md`](docs/DATA.md) for the file manifest, integrity hashes, and third-party
  database citations.

See [`PROVENANCE.md`](docs/PROVENANCE.md) for how this repository was derived. Licensed under MIT — see `LICENSE`.

## Citation

```bibtex
@article{nam2026llm4mof,
  title         = {Interpretable Inverse Design of Metal--Organic Frameworks with Large Language Model Agents},
  author        = {Nam, Kyungmin and Han, Seunghee and Kim, Jihan},
  journal       = {arXiv preprint arXiv:2606.29459},
  year          = {2026},
  eprint        = {2606.29459},
  archivePrefix = {arXiv}
}
```

Preprint: <https://arxiv.org/abs/2606.29459>
