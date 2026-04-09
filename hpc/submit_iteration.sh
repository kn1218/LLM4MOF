#!/bin/bash
# =============================================================================
# submit_iteration.sh — Generate per-MOF qsub scripts and submit via qas
#
# Usage:
#   cd ~/llm2por/exp_xxx/iter_1
#   bash ~/llm2por/hpc/submit_iteration.sh batch_manifest.json results [node_prop]
#
# All paths in generated .qsub files are ABSOLUTE to avoid PBS cwd issues.
# =============================================================================

set -e

MANIFEST="${1:?Usage: submit_iteration.sh <manifest.json> [output_dir] [node_property]}"
OUTPUT_DIR="${2:-results}"
NODE_PROP="${3:-ac}"  # dirac1 node property: aa, ab, ac, amd, ax, xeonphi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Resolve all paths to absolute
MANIFEST_ABS="$(cd "$(dirname "$MANIFEST")" && pwd)/$(basename "$MANIFEST")"
WORK_DIR="$(pwd)"
OUTPUT_ABS="${WORK_DIR}/${OUTPUT_DIR}"

# Read number of jobs from manifest
N_JOBS=$(python3 -c "import json; m=json.load(open('$MANIFEST_ABS')); print(m['n_jobs'])")

if [ "$N_JOBS" -eq 0 ]; then
    echo "[submit] No jobs in manifest. Nothing to submit."
    exit 0
fi

echo "[submit] Manifest: $MANIFEST_ABS ($N_JOBS jobs)"
echo "[submit] Output: $OUTPUT_ABS"
echo "[submit] Work dir: $WORK_DIR"

# Create directories
QSUB_DIR="${OUTPUT_ABS}/qsub_scripts"
mkdir -p "$QSUB_DIR" "${OUTPUT_ABS}/logs"

# Generate one .qsub file per job (all paths absolute)
for i in $(seq 0 $((N_JOBS - 1))); do
    FILENAME=$(python3 -c "import json; m=json.load(open('$MANIFEST_ABS')); print(m['jobs'][$i]['filename'])")

    cat > "${QSUB_DIR}/job_${i}_${FILENAME}.qsub" << QEOF
#!/bin/bash
#PBS -N llm2por_${FILENAME}
#PBS -l nodes=1:ppn=1:${NODE_PROP}
#PBS -l walltime=02:00:00
#PBS -o ${OUTPUT_ABS}/logs/${FILENAME}.out
#PBS -e ${OUTPUT_ABS}/logs/${FILENAME}.err

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate llm2auto

cd ${WORK_DIR}

python ${SCRIPT_DIR}/run_mof_sim.py \\
    --manifest ${MANIFEST_ABS} \\
    --job-index ${i} \\
    --output-dir ${OUTPUT_ABS}
QEOF
done

echo "[submit] Generated $N_JOBS qsub scripts in $QSUB_DIR"

# Submit all via qas
echo "[submit] Submitting via qas..."
qas ${QSUB_DIR}/*.qsub

echo "[submit] All $N_JOBS jobs submitted. Monitor with: myqstat && myqinfo"
