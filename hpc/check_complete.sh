#!/bin/bash
# -----------------------------------------------------------------------------
# NOTE: This is one of the authors' HPC batch scripts (PBS/Torque scheduler),
# provided for transparency and reproducibility. It is NOT runnable as-is on
# other systems. Override CONDA_ENV, SUBMIT_CMD, STATUS_CMD, LAMMPS_BIN and the
# #PBS directives to match your own cluster before use.
# -----------------------------------------------------------------------------
# =============================================================================
# check_complete.sh — Check if all HPC jobs finished
#
# Usage:
#   bash check_complete.sh batch_manifest.json results/
#
# Prints: ALL_DONE | PENDING done/total
# On ALL_DONE, also runs aggregate_results.py
# =============================================================================

MANIFEST="${1:?Usage: check_complete.sh <manifest.json> [results_dir]}"
RESULTS_DIR="${2:-results}"
CONDA_ENV="${CONDA_ENV:-llm4mof}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

N_JOBS=$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(m['n_jobs'])")
DONE_COUNT=$(ls "${RESULTS_DIR}"/*.DONE 2>/dev/null | wc -l)

if [ "$DONE_COUNT" -ge "$N_JOBS" ]; then
    # Aggregate results
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
    python3 "${SCRIPT_DIR}/aggregate_results.py" --manifest "$MANIFEST" --results-dir "$RESULTS_DIR"
    echo "ALL_DONE"
else
    echo "PENDING ${DONE_COUNT}/${N_JOBS}"
fi
