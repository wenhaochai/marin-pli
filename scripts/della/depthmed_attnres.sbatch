#!/bin/bash
#SBATCH --job-name=depthmed-attnres
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=48G
#SBATCH --time=01:01:00
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
#FETCHABLE: self-contained (every path below is literal; nothing comes from the submit shell)

# depthmed.py on the two full-AttnRes checkpoints (ordinary, separate heads), one process on one GPU. In depthmed.sbatch
# (15844436) both ran out of GPU memory at two processes per GPU (XLA_PYTHON_CLIENT_MEM_FRACTION 0.45): their 97-slot
# source buffers need the whole GPU; the smoke (15844368) ran attnres alone at 0.85.
set -uo pipefail
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
WT=/scratch/gpfs/GROUP/USER/project/marin-pls-auto10
OUT=/scratch/gpfs/GROUP/USER/tmp/pls_status/depthmed
LOGD=$OUT/logs/$JID
cd $WT
source scripts/della/pls_env.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 JAX_PLATFORMS=cuda
mkdir -p "$LOGD"
echo "job $JID on $(hostname): GPU ${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv -l 30 > "$LOGD/gpu_util.csv" 2>&1 &
SMI=$!
P=muonh-qwen3-130m-della4xh100-pls1-sep
$PY scripts/della/depthmed.py "$OUT" "$OUT/cfg" $P-bbfrozen-aattnres-vs-d48-g4 $P-aattnres-vs-d48-g4 > "$LOGD/worker0.log" 2>&1
rc=$?
kill $SMI 2>/dev/null || true
grep -E "wrote|WARN|FAILED|all done|Error" "$LOGD/worker0.log" | tail -8
exit $rc
