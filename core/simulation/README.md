# MOF Simulation Pipeline

Automated pipeline for Metal-Organic Framework (MOF) simulation including geometry prediction, CIF generation, LAMMPS optimization, and RASPA3 GCMC simulations.

## Structure

```
simulation/
├── generate_mofs.py     # Generate CIF files from PORMAKE
├── opt/                 # LAMMPS optimization
│   └── optimize.py
└── gcmc/              # RASPA3 GCMC simulations
    ├── run_raspa.py
    ├── analyze.py
    └── raspa_utils.py
```

## Usage

### Full Pipeline (via run_simulation.py)

```bash
python core/run_simulation.py \
  --input_json agent2_output.json \
  --output_dir ./test_run_simulation \
  --num_mofs 10
```

### Individual Steps

#### 1. Generate MOF CIFs (PORMAKE)

```bash
python core/simulation/generate_mofs.py \
  --mof-dir ./cif_output \
  --result_file test_result_agent3.json \
  --max 10
```

#### 2. LAMMPS Optimization

```bash
python core/simulation/opt/optimize.py \
  --cif-dir ./cif_output \
  --output-dir ./optimization
```

#### 3. RASPA3 GCMC Simulation

```bash
python core/simulation/gcmc/run_raspa.py \
  --mof-dir ./optimized_cifs \
  --output-dir ./gcmc_results \
  --temperature 77 \
  --pressure 500000 \
  --cycles 10000
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

## Requirements

- Python 3.9+
- RASPA3 (conda install raspa3 -c conda-forge)
- LAMMPS with lammps-interface
- PORMAKE (pip install git+https://github.com/Sangwon91/PORMAKE.git)

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
```

## Parallel Execution

LAMMPS and RASPA3 jobs run in parallel (background) for each MOF:

- Each MOF runs as a separate background process
- Use `htop` or `ps aux | grep raspa3` to monitor
- Completion is tracked via `DONE.txt` files
