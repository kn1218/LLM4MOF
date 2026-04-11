#!/bin/bash
# Poll HPC until done, then aggregate + download
# Usage: bash hpc_poll.sh <experiment> <iteration>
EXP="${1:?Usage: hpc_poll.sh <experiment> <iteration>}"
ITER="${2:?Usage: hpc_poll.sh <experiment> <iteration>}"
N_JOBS="${3:-104}"
HPC_DIR="~/llm2por/${EXP}/iter_${ITER}"
LOCAL_DIR="experiments/${EXP}/iter_${ITER}"

echo "[Poll] Monitoring ${EXP}/iter_${ITER} (${N_JOBS} jobs)"

while true; do
    sleep 300
    DONE=$(ssh -o ConnectTimeout=10 dirac1 "ls ${HPC_DIR}/results/*.DONE 2>/dev/null | wc -l" 2>/dev/null)
    DONE=${DONE//[^0-9]/}
    echo "[Poll] $(date +%H:%M) ${DONE}/${N_JOBS} done"
    if [ "${DONE}" -ge "${N_JOBS}" ] 2>/dev/null; then
        echo "[Poll] ALL COMPLETE"
        break
    fi
done

# Aggregate
echo "[Poll] Aggregating..."
ssh -o ConnectTimeout=10 dirac1 "cd ${HPC_DIR} && source ~/anaconda3/etc/profile.d/conda.sh && conda activate llm2auto && python ~/llm2por/hpc/aggregate_results.py --manifest batch_manifest.json --results-dir results" 2>&1 | tail -2

# Download
mkdir -p "${LOCAL_DIR}/hpc_results"
scp dirac1:${HPC_DIR}/results/batch_results.json "${LOCAL_DIR}/hpc_results/batch_results.json" 2>/dev/null
echo "[Poll] Downloaded to ${LOCAL_DIR}/hpc_results/batch_results.json"
echo "READY_FOR_COLLECT"
