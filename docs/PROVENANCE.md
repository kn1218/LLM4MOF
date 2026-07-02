# Provenance

This repository (**LLM4MOF**) is a cleaned, publication-ready extraction of an internal research
production runtime.

| | |
|---|---|
| Source branch / commit | `master` @ `dbe9c13` (pre-publication internal history) |
| Extraction date | 2026-06-24 |
| KEEP manifest | 85 files |
| KEEP manifest SHA-256 | `5ddb813921f744ef44fe885ad0bc42bf4c89a8f8ea4f1f6c2431633c11d44e1a` |

## What was done

1. **Selective copy.** Only the runnable production surface (the 85-file KEEP manifest) was copied
   from the source working tree, with Git LFS content materialized. Research artifacts, experiment
   batches, archived/ablation code and prompts, internal handoff notes, and developer tooling were
   excluded.
2. **Dead-code/comment cleanup.** Removed unused config constants (`QMOF_JSONS_V3_DIR`,
   `QMOF_BB_FILTERED_PATH` — neither was referenced at runtime) and corrected stale documentation
   (active prompts are `prompts/agent1_v3.0_production.md` and `prompts/agent2_v4.1.md`).
3. **Fresh history.** Re-initialized as a new git repository with a single initial commit; the
   439 MB of legacy blob history from the source working tree was intentionally dropped.

## HPC environment note

The scripts in `hpc/` and `core/hpc/` are the authors' PBS/Torque batch scripts, included for
transparency and reproducibility. They are **not runnable as-is** on other systems. Key settings are
exposed as overridable shell variables (`CONDA_ENV`, `SUBMIT_CMD`, `STATUS_CMD`, `LAMMPS_BIN`,
`NODE_PROP`) with generic defaults; set these and the `#PBS` directives to match your own cluster.
The Python-side equivalents live in the HPC section of `config.py` (`HPC_HOST`, `HPC_SUBMIT_CMD`,
`HPC_STATUS_CMD`, `HPC_NODE_PROPERTY`, …).

## Not bundled

mof2zeo training data (`core/mof2zeo/data/train.csv`, `valid.csv`) is not shipped due to size.
`core/mof2zeo/train.py` is retained to document how the shipped checkpoint was produced; the
training data is available on request.
