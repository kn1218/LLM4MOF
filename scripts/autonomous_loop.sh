#!/bin/bash
# =============================================================================
# autonomous_loop.sh — Full autonomous iteration loop
#
# Orchestrates: local --prepare → scp → qas → poll → scp results → --collect
#
# Usage:
#   bash scripts/autonomous_loop.sh <experiment_id> [max_iterations]
#
# Prerequisites:
#   - SSH key auth to dirac1 working
#   - HPC has ~/llm2por/hpc/ with run_mof_sim.py etc.
#   - HPC has ~/llm2por/forcefields/UFF_H2/ with forcefield files
#   - Local conda env llm2auto activated (or use PYTHON var below)
# =============================================================================

set -e

EXP_ID="${1:?Usage: autonomous_loop.sh <experiment_id> [max_iterations]}"
MAX_ITER="${2:-3}"
POLL_INTERVAL=300  # seconds (5 minutes)

# --- Configuration ---
HPC_HOST="dirac1"
HPC_USER="kn1218"
HPC_BASE="~/llm2por"
LOCAL_BASE="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"  # Override with full path if needed
EXPERIMENTS_DIR="${LOCAL_BASE}/experiments"

echo "============================================================"
echo "  LLM2POR AUTONOMOUS LOOP"
echo "  Experiment: $EXP_ID"
echo "  Max Iterations: $MAX_ITER"
echo "  HPC: ${HPC_USER}@${HPC_HOST}:${HPC_BASE}"
echo "  Poll interval: ${POLL_INTERVAL}s"
echo "============================================================"

for ITER in $(seq 1 "$MAX_ITER"); do
    echo ""
    echo "======================== ITERATION $ITER ========================"

    ITER_DIR="${EXPERIMENTS_DIR}/${EXP_ID}/iter_${ITER}"
    HPC_ITER_DIR="${HPC_BASE}/${EXP_ID}/iter_${ITER}"

    # --- Phase 1: Local prepare (LLM agents + matchmaker + manifest) ---
    echo "[Phase 1] Running local --prepare..."
    $PYTHON "${LOCAL_BASE}/run_live_experiment.py" --resume "$EXP_ID" --prepare
    echo "[Phase 1] Done."

    MANIFEST="${ITER_DIR}/batch_manifest.json"
    if [ ! -f "$MANIFEST" ]; then
        echo "[ERROR] No manifest generated at $MANIFEST. Stopping."
        exit 1
    fi

    N_JOBS=$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(m['n_jobs'])")
    echo "[Phase 1] Manifest: $N_JOBS jobs"

    if [ "$N_JOBS" -eq 0 ]; then
        echo "[WARN] No jobs to simulate (all beams empty). Running --collect with empty results."
        # Create empty batch_results.json
        mkdir -p "${ITER_DIR}/hpc_results"
        python3 -c "
import json
with open('$MANIFEST') as f: m = json.load(f)
r = {'experiment_id': m['experiment_id'], 'iteration': m['iteration'],
     'n_jobs': 0, 'n_success': 0, 'n_fail': 0, 'n_missing': 0,
     'total_wall_seconds': 0, 'results': []}
with open('${ITER_DIR}/hpc_results/batch_results.json', 'w') as f:
    json.dump(r, f, indent=2)
"
        $PYTHON "${LOCAL_BASE}/run_live_experiment.py" --resume "$EXP_ID" --collect
        continue
    fi

    # --- Phase 2: Upload manifest + submit on HPC ---
    echo "[Phase 2] Uploading to HPC..."
    ssh "$HPC_HOST" "mkdir -p ${HPC_ITER_DIR}/results"
    scp "$MANIFEST" "${HPC_HOST}:${HPC_ITER_DIR}/batch_manifest.json"
    echo "[Phase 2] Submitting via qas..."
    ssh "$HPC_HOST" "cd ${HPC_ITER_DIR} && bash ${HPC_BASE}/hpc/submit_iteration.sh batch_manifest.json results"
    echo "[Phase 2] Jobs submitted."

    # --- Phase 3: Poll for completion ---
    echo "[Phase 3] Polling for completion (every ${POLL_INTERVAL}s)..."
    while true; do
        STATUS=$(ssh "$HPC_HOST" "cd ${HPC_ITER_DIR} && bash ${HPC_BASE}/hpc/check_complete.sh batch_manifest.json results" 2>/dev/null | tail -1)
        echo "  [$(date +%H:%M:%S)] $STATUS"

        if [[ "$STATUS" == "ALL_DONE" ]]; then
            break
        fi

        sleep "$POLL_INTERVAL"
    done
    echo "[Phase 3] All jobs complete."

    # --- Phase 4: Download results + local collect ---
    echo "[Phase 4] Downloading results..."
    mkdir -p "${ITER_DIR}/hpc_results"
    scp "${HPC_HOST}:${HPC_ITER_DIR}/results/batch_results.json" "${ITER_DIR}/hpc_results/"
    echo "[Phase 4] Running --collect..."
    $PYTHON "${LOCAL_BASE}/run_live_experiment.py" --resume "$EXP_ID" --collect
    echo "[Phase 4] Iteration $ITER complete."
done

echo ""
echo "============================================================"
echo "  AUTONOMOUS LOOP COMPLETE"
echo "  Experiment: $EXP_ID"
echo "  Iterations: $MAX_ITER"
echo "============================================================"
