#!/bin/bash
# -----------------------------------------------------------------------------
# NOTE: This is one of the authors' HPC batch scripts (PBS/Torque scheduler),
# provided for transparency and reproducibility. It is NOT runnable as-is on
# other systems. Override CONDA_ENV, SUBMIT_CMD, STATUS_CMD, LAMMPS_BIN and the
# #PBS directives to match your own cluster before use.
# -----------------------------------------------------------------------------
# =============================================================================
# submit_iteration.sh — Generate per-MOF qsub scripts and submit via $SUBMIT_CMD (default qsub)
#
# Usage:
#   cd ~/llm4mof/exp_xxx/iter_1
#   bash ~/llm4mof/hpc/submit_iteration.sh batch_manifest.json results [node_prop] [--use-zeo] [--adsorbate X ...]
#
# All paths in generated .qsub files are ABSOLUTE to avoid PBS cwd issues.
# =============================================================================

set -e

MANIFEST="${1:?Usage: submit_iteration.sh <manifest.json> [output_dir] [node_property] [--use-zeo] [--adsorbate X --xe-molfrac Y --temperature T --pressure P ...]}"
OUTPUT_DIR="${2:-results}"
NODE_PROP="${3:-ac}"  # cluster-specific node property (optional); adjust for your scheduler
USE_ZEO="${4:-}"      # optional: "--use-zeo" to enable Zeo++ post-processing
JOB_PREFIX="${JOB_PREFIX:-llm4mof}"  # PBS job name prefix; set by caller for parallel-experiment isolation
CONDA_ENV="${CONDA_ENV:-llm4mof}"          # conda env to activate on compute nodes
SUBMIT_CMD="${SUBMIT_CMD:-qsub}"           # batch submit command (set to your scheduler's)
STATUS_CMD="${STATUS_CMD:-qstat}"          # job-status command (set to your scheduler's)
LAMMPS_BIN="${LAMMPS_BIN:-/path/to/lammps/bin}"  # dir containing the LAMMPS executable
# All remaining args ($5+) are passed through directly to run_mof_sim.py
SIM_ARGS="${@:5}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Resolve Zeo++ binary path (expected alongside HPC scripts)
ZEOPP_BIN="${SCRIPT_DIR}/network"
ZEO_ARGS=""
if [ "$USE_ZEO" = "--use-zeo" ] && [ -f "$ZEOPP_BIN" ]; then
    ZEO_ARGS="--use-zeo --zeopp-bin ${ZEOPP_BIN}"
    echo "[submit] Zeo++ enabled: ${ZEOPP_BIN}"
elif [ "$USE_ZEO" = "--use-zeo" ]; then
    echo "[submit] WARNING: --use-zeo requested but binary not found at ${ZEOPP_BIN}"
fi

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
#PBS -N ${JOB_PREFIX}_${FILENAME}
#PBS -l nodes=1:ppn=1:${NODE_PROP}
#PBS -l walltime=04:00:00
#PBS -q long
#PBS -o ${OUTPUT_ABS}/logs/${FILENAME}.out
#PBS -e ${OUTPUT_ABS}/logs/${FILENAME}.err

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export PATH="${LAMMPS_BIN}:\$PATH"

cd ${WORK_DIR}

python ${SCRIPT_DIR}/run_mof_sim.py \\
    --manifest ${MANIFEST_ABS} \\
    --job-index ${i} \\
    --output-dir ${OUTPUT_ABS} \\
    ${ZEO_ARGS} \\
    ${SIM_ARGS}
QEOF
done

echo "[submit] Generated $N_JOBS qsub scripts in $QSUB_DIR"

# Submit in batches of 48 via $SUBMIT_CMD (avoid per-user queue limit)
BATCH_SIZE=48
echo "[submit] Submitting $N_JOBS jobs in batches of $BATCH_SIZE..."

all_qsubs=($(ls ${QSUB_DIR}/*.qsub | sort))
total=${#all_qsubs[@]}
submitted=0

for ((i=0; i<total; i+=BATCH_SIZE)); do
    batch=("${all_qsubs[@]:$i:$BATCH_SIZE}")
    echo "[submit] Batch $((i/BATCH_SIZE + 1)): submitting ${#batch[@]} jobs..."
    "$SUBMIT_CMD" "${batch[@]}"
    submitted=$((submitted + ${#batch[@]}))
    if [ $((i + BATCH_SIZE)) -lt $total ]; then
        echo "[submit] Sleeping 5s before next batch..."
        sleep 5
    fi
done

echo "[submit] $submitted/$N_JOBS jobs submitted. Monitor with: $STATUS_CMD"
