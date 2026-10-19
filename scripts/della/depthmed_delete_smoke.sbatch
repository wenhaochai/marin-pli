#!/bin/bash
#SBATCH --job-name=depthmed-del-smoke
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=48G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out

# Smoke test of depthmed_delete.sbatch: depthmed.py --delete on copies of hc-ord, lns-sep and attnres_block-ord (and the copy
# path on preln-ord) in tmp/pls_status/depthmed/smoke, layers 1 and 48, two workers on one GPU, utilization every 10 s.
set -uo pipefail
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
WT=/scratch/gpfs/GROUP/USER/project/marin-pls-auto10
D=/scratch/gpfs/GROUP/USER/tmp/pls_status/depthmed
OUT=$D/smoke
LOGD=$OUT/logs/$JID
cd $WT
source scripts/della/pls_env.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 JAX_PLATFORMS=cuda
mkdir -p "$LOGD"
for f in hc-ord lns-sep attnres_block-ord preln-ord; do cp $D/$f.json $OUT/$f.json; done
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv -l 10 > "$LOGD/gpu_util.csv" 2>&1 &
SMI=$!
P=muonh-qwen3-130m-della4xh100-pls1-sep
$PY scripts/della/depthmed.py "$OUT" "$D/cfg" $P-bbfrozen-apreln-vs-d48-g4 $P-bbfrozen-ahc-vs-d48-g4 $P-alns-vs-d48-g4 --delete --skip-layers 1,48 > "$LOGD/worker0.log" 2>&1 &
p0=$!
$PY scripts/della/depthmed.py "$OUT" "$D/cfg" $P-bbfrozen-aattnres_block-vs-d48-g4 --delete --skip-layers 1,48 > "$LOGD/worker1.log" 2>&1 &
p1=$!
rc=0
wait $p0 || rc=$?
wait $p1 || rc=$?
kill $SMI 2>/dev/null || true
for f in "$LOGD"/worker*.log; do echo "== $f"; grep -v "Unknown leaf" "$f" | tail -12; done
exit $rc
