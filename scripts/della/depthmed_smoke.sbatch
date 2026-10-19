#!/bin/bash
#SBATCH --job-name=depthmed-smoke
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=24G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out

# Smoke test of depthmed.sbatch: 2 models (attnres and deepnorm, separate heads; the costliest designs), layers 1 and 48
# skipped, one worker per GPU, GPU utilization logged every 10 s. Output: tmp/pls_status/depthmed/smoke/.
set -uo pipefail
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
WT=/scratch/gpfs/GROUP/USER/project/marin-pls-auto10
OUT=/scratch/gpfs/GROUP/USER/tmp/pls_status/depthmed/smoke
CFG=/scratch/gpfs/GROUP/USER/tmp/pls_status/depthmed/cfg
LOGD=$OUT/logs/$JID
cd $WT
source scripts/della/pls_env.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 JAX_PLATFORMS=cuda
mkdir -p "$LOGD"
IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
echo "job $JID on $(hostname): GPUs ${DEVS[*]}"
nvidia-smi -L || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv -l 10 > "$LOGD/gpu_util.csv" 2>&1 &
SMI=$!
P=muonh-qwen3-130m-della4xh100-pls1-sep
CUDA_VISIBLE_DEVICES=${DEVS[0]} $PY scripts/della/depthmed.py "$OUT" "$CFG" $P-aattnres-vs-d48-g4 --skip-layers 1,48 > "$LOGD/worker0.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=${DEVS[1]} $PY scripts/della/depthmed.py "$OUT" "$CFG" $P-adeepnorm-vs-d48-g4 --skip-layers 1,48 > "$LOGD/worker1.log" 2>&1 &
p1=$!
rc=0
wait $p0 || rc=$?
wait $p1 || rc=$?
kill $SMI 2>/dev/null || true
for f in "$LOGD"/worker*.log; do echo "== $f"; grep -v "Unknown leaf" "$f" | tail -25; done
exit $rc
