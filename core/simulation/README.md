# MOF Simulation Pipeline

Automated pipeline for Metal-Organic Framework (MOF) simulation including geometry prediction, CIF generation, LAMMPS optimization, and RASPA3 GCMC simulations.

## Structure

```
core/
├── filter_candidate.py     # mof2zeo candidate filtering (geometry prediction)
├── run_simulation.py       # Full pipeline orchestrator
├── mof2zeo/                # Geometry prediction model
│   ├── ckpt/               # Model checkpoint
│   └── config.yaml
└── simulation/             # Simulation modules
    ├── generate_mofs.py    # CIF generation from PORMAKE
    ├── opt/                # LAMMPS optimization
    │   └── optimize.py
    └── gcmc/              # RASPA3 GCMC simulations
        ├── run_raspa.py
        ├── analyze.py
        └── raspa_utils.py
```

## Full Pipeline (run_simulation.py)

Runs the complete pipeline: matchmaker → filter_candidate → generate_mofs → optimize → RASPA3

```bash
python core/run_simulation.py \
  --input_json agent2_output.json \
  --output_dir ./results \
  --num_mofs 10
```

Arguments:
- `--input_json` (required): Path to agent2_output.json from Agent 2
- `--output_dir` (optional): Output directory, default: `test_run_simulation`
- `--num_mofs` (optional): Number of MOFs to generate, default: 10

## Individual Steps

### Step 0: Filter Candidates (mof2zeo)

Filter MOF candidates using geometry prediction model:

```bash
python core/filter_candidate.py \
  --constraints agent2_output.json \
  --output test_result_agent3.json \
  --top_n 10
```

Arguments:
- `--constraints` (optional): Combined constraints JSON (agent2 output format)
- `--matchmaker` (optional): Matchmaker JSON file (legacy format)
- `--output` (optional): Output JSON file, default: `test_result_agent3.json`
- `--top_n` (optional): Number of candidates to generate, default: 10

### Step 1: Generate MOF CIFs (PORMAKE)

```bash
python core/simulation/generate_mofs.py \
  --mof-dir ./cif_output \
  --result_file test_result_agent3.json \
  --max 10 \
  --db-path /path/to/pormake_db.json
```

Arguments:
- `--mof-dir` (required): Output directory for CIF files
- `--result_file` (required): Input JSON file with ranked MOFs
- `--max` (optional): Maximum number of MOFs to generate, default: 20
- `--db-path` (optional): Path to PORMAKE database

### Step 2: LAMMPS Optimization

```bash
python core/simulation/opt/optimize.py \
  --cif-dir ./cif_output \
  --output-dir ./optimization
```

### Step 3: RASPA3 GCMC Simulation

```bash
python core/simulation/gcmc/run_raspa.py \
  --mof-dir ./optimized_cifs \
  --output-dir ./gcmc_results \
  --temperature 77 \
  --pressure 500000 \
  --cycles 10000
```

## Output Directories

When using the full pipeline:

```
output_dir/
├── 0_mof_candidates/    # filter_candidate output (ranked MOFs)
├── 1_mof_generation/   # PORMAKE CIF generation
├── 2_opt/               # LAMMPS optimization
│   ├── lammps_data/    # LAMMPS data files
│   ├── optimized_cifs/  # Optimized CIF files
│   └── opt_lammps_data/
└── 3_gcmc/             # RASPA3 results
    └── [mof_name]/
        ├── output/      # RASPA output
        └── DONE.txt    # Completion flag
```

## Time Tracking

The full pipeline prints timing information for each step:

```
[Step 1/5] Running matchmaker... (X.Xs)
[Step 2/5] Filtering candidates with mof2zeo... (X.Xs)
[Step 3/5] Generating MOF CIFs... (X.Xs)
[Step 4/5] Running LAMMPS optimization... (X.Xs)
[Step 5/5] Running RASPA3 GCMC... (X.Xs)
Total time: X.Xs
```

## RASPA3 Auto-Detection

The pipeline automatically detects the RASPA3 binary:

1. First checks if `raspa` module is installed in current environment
2. Falls back to `which raspa3`
3. Falls back to conda environment bin directory

To specify manually:

```bash
python core/simulation/gcmc/run_raspa.py \
  --raspa3 /path/to/raspa3 \
  --mof-dir ./cifs
```

## Requirements

- Python 3.9+
- RASPA3 (conda install raspa3 -c conda-forge)
- LAMMPS with lammps-interface
- PORMAKE (pip install git+https://github.com/Sangwon91/PORMAKE.git)
- mof2zeo checkpoint (included via Git LFS)

## Environment Setup

```bash
# Create and activate environment
conda create -n llm2por python=3.11
conda activate llm2por

# Install dependencies
pip install numpy pandas scipy requests openai python-dotenv

# Install RASPA3
conda install raspa3 -c conda-forge

# Install LLM2POR in editable mode
cd LLM2POR
pip install -e .

# Download large files (checkpoints, databases)
git lfs pull
```

## Parallel Execution

LAMMPS and RASPA3 jobs run in parallel (background) for each MOF:

- Each MOF runs as a separate background process
- Use `htop` or `ps aux | grep raspa3` to monitor
- Completion is tracked via `DONE.txt` files