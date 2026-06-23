#!/bin/bash
# -----------------------------------------------------------------------------
# NOTE: This is one of the authors' HPC batch scripts (PBS/Torque scheduler),
# provided for transparency and reproducibility. It is NOT runnable as-is on
# other systems. Override CONDA_ENV, SUBMIT_CMD, STATUS_CMD, LAMMPS_BIN and the
# #PBS directives to match your own cluster before use.
# -----------------------------------------------------------------------------
# =============================================================================
# submit_iteration_packed.sh — Pack 4 MOF jobs per qsub (packed mode, ppn=4)
#
# Usage:
#   cd ~/llm4mof/exp_xxx/iter_1
#   bash ~/llm4mof/hpc/submit_iteration_packed.sh batch_manifest.json results [node_prop] [--use-zeo] [sim_args...]
#
# Differences from submit_iteration.sh (single-job):
#   - 4 jobs packed per qsub (#PBS -l nodes=1:ppn=4)
#   - Uses $SUBMIT_CMD for batch submission (same as single-job)
#   - Parallel execution via & ... wait
# =============================================================================

set -e

MANIFEST="${1:?Usage: submit_iteration_packed.sh <manifest.json> [output_dir] [node_property] [--use-zeo] [...]}"
OUTPUT_DIR="${2:-results}"
NODE_PROP="${3:-aa}"
USE_ZEO="${4:-}"
SIM_ARGS="${@:5}"
JOB_PREFIX="${JOB_PREFIX:-llm4mof}"  # PBS job name prefix; set by caller for parallel-experiment isolation
CONDA_ENV="${CONDA_ENV:-llm4mof}"          # conda env to activate on compute nodes
SUBMIT_CMD="${SUBMIT_CMD:-qsub}"           # batch submit command (set to your scheduler's)
STATUS_CMD="${STATUS_CMD:-qstat}"          # job-status command (set to your scheduler's)
LAMMPS_BIN="${LAMMPS_BIN:-/path/to/lammps/bin}"  # dir containing the LAMMPS executable
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ZEOPP_BIN="${SCRIPT_DIR}/network"
ZEO_ARGS=""
if [ "$USE_ZEO" = "--use-zeo" ] && [ -f "$ZEOPP_BIN" ]; then
    ZEO_ARGS="--use-zeo --zeopp-bin ${ZEOPP_BIN}"
    echo "[submit] Zeo++ enabled: ${ZEOPP_BIN}"
elif [ "$USE_ZEO" = "--use-zeo" ]; then
    echo "[submit] WARNING: --use-zeo requested but binary not found at ${ZEOPP_BIN}"
fi

MANIFEST_ABS="$(cd "$(dirname "$MANIFEST")" && pwd)/$(basename "$MANIFEST")"
WORK_DIR="$(pwd)"
OUTPUT_ABS="${WORK_DIR}/${OUTPUT_DIR}"

N_JOBS=$(python3 -c "import json; m=json.load(open('$MANIFEST_ABS')); print(m['n_jobs'])")

if [ "$N_JOBS" -eq 0 ]; then
    echo "[submit] No jobs in manifest. Nothing to submit."
    exit 0
fi

JOBS_PER_QSUB=4
N_QSUBS=$(( (N_JOBS + JOBS_PER_QSUB - 1) / JOBS_PER_QSUB ))

echo "[submit] Manifest: $MANIFEST_ABS ($N_JOBS jobs → $N_QSUBS qsubs, 4 jobs/qsub)"
echo "[submit] Output: $OUTPUT_ABS"
echo "[submit] Mode: packed (ppn=4:${NODE_PROP})"

QSUB_DIR="${OUTPUT_ABS}/qsub_scripts"
mkdir -p "$QSUB_DIR" "${OUTPUT_ABS}/logs"

# Generate qsub scripts (Python for clean job-grouping logic)
python3 << PYEOF
import json, os

manifest = json.load(open("$MANIFEST_ABS"))
jobs = manifest["jobs"]
n_jobs = len(jobs)
jobs_per_qsub = $JOBS_PER_QSUB
script_dir = "$SCRIPT_DIR"
work_dir = "$WORK_DIR"
output_abs = "$OUTPUT_ABS"
node_prop = "$NODE_PROP"
zeo_args = "$ZEO_ARGS"
sim_args = "$SIM_ARGS"
qsub_dir = "$QSUB_DIR"
manifest_abs = "$MANIFEST_ABS"
job_prefix = "$JOB_PREFIX"
n_qsubs = (n_jobs + jobs_per_qsub - 1) // jobs_per_qsub

for q in range(n_qsubs):
    start = q * jobs_per_qsub
    batch = jobs[start:start + jobs_per_qsub]
    batch_indices = list(range(start, start + len(batch)))

    qsub_path = os.path.join(qsub_dir, f"llm4mof_{q}_{batch[0]['filename']}.qsub")

    # Use the conda env's python (resolved dynamically after conda activate)
    python_bin = "python"

    lines = [
        "#!/bin/bash",
        f"#PBS -N {job_prefix}_{q}_{batch[0]['filename'][:20]}",
        "#PBS -r n",
        "#PBS -q long",
        f"#PBS -l nodes=1:ppn=4:{node_prop}",
        "#PBS -o /dev/null",
        "#PBS -e /dev/null",
        "",
        "source ~/anaconda3/etc/profile.d/conda.sh",
        f"conda activate {os.environ.get('CONDA_ENV', 'llm4mof')}",
        "",
        f"cd {work_dir}",
        f"mkdir -p {output_abs}/logs",
        "",
    ]

    for idx, job in zip(batch_indices, batch):
        fname = job["filename"]
        lines.append(
            f"({python_bin} {script_dir}/run_mof_sim.py "
            f"--manifest {manifest_abs} "
            f"--job-index {idx} "
            f"--output-dir {output_abs} "
            f"{zeo_args} {sim_args} "
            f"> {output_abs}/logs/{fname}.out 2>&1) &"
        )

    lines += ["wait", ""]

    with open(qsub_path, "w") as f:
        f.write("\n".join(lines))

print(f"[submit] Generated {n_qsubs} qsub scripts in {qsub_dir}")
PYEOF

# Submit in batches to avoid PBS per-user queue limit (the cluster has limited nodes of this type)
BATCH_SIZE=20
all_qsubs=($(ls ${QSUB_DIR}/*.qsub | sort))
total_qsubs=${#all_qsubs[@]}
submitted=0

echo "[submit] Submitting $total_qsubs qsub scripts in batches of $BATCH_SIZE..."
for ((i=0; i<total_qsubs; i+=BATCH_SIZE)); do
    batch=("${all_qsubs[@]:$i:$BATCH_SIZE}")
    echo "[submit] Batch $((i/BATCH_SIZE + 1)): submitting ${#batch[@]} qsubs (jobs $((i*4+1))-$(( (i+${#batch[@]})*4 )))..."
    "$SUBMIT_CMD" "${batch[@]}"
    submitted=$((submitted + ${#batch[@]}))
    if [ $((i + BATCH_SIZE)) -lt $total_qsubs ]; then
        echo "[submit] Sleeping 10s before next batch..."
        sleep 10
    fi
done

echo "[submit] $submitted/$total_qsubs qsubs submitted ($N_JOBS total jobs). Monitor with: $STATUS_CMD"
