#!/bin/bash
#SBATCH --job-name=depthmed
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=48G   # two worker processes per GPU, each loads a ~13 GB float32 model
#SBATCH --time=01:01:00
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
#FETCHABLE: self-contained (every path below is literal; nothing comes from the submit shell)

# Depth check-up (owner popup 2026-10-19, "按方案做"): scripts/della/depthmed.py on the 20 48-layer checkpoints (10
# DepthBench designs x ordinary / separate heads), two worker processes per visible GPU (the smoke, 15844368: a model
# takes ~2.5 min on one H100, mostly dispatch and compile, so one process alone leaves the GPU near 0% between bursts; 2
# GPUs, not 8, so no GPU waits idle); the models are split over the workers by longest-processing-time first on a cost
# estimate, so the workers finish close together. Writes
# tmp/pls_status/depthmed/<arch>-<ord|sep>.json. The smoke test is a copy of this script (depthmed_smoke.sbatch).
set -uo pipefail
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
WT=/scratch/gpfs/GROUP/USER/project/marin-pls-auto10
OUT=/scratch/gpfs/GROUP/USER/tmp/pls_status/depthmed
CFG=$OUT/cfg
LOGD=$OUT/logs/$JID
cd $WT
source scripts/della/pls_env.sh   # PY, PYTHONPATH, TMPDIR = tmp/job/$JID (removed at exit)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 JAX_PLATFORMS=cuda   # two processes per GPU
mkdir -p "$LOGD"
IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
G=${#DEVS[@]}
WK=$((2 * G))
echo "job $JID on $(hostname): $G GPUs (${DEVS[*]}), $WK workers, logs $LOGD"
nvidia-smi -L || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv -l 30 > "$LOGD/gpu_util.csv" 2>&1 &
SMI=$!
P=muonh-qwen3-130m-della4xh100-pls1-sep
NAMES=()
for a in "" preln lns deepnorm keel hc mhc attnres attnres_block moda; do
  suf=${a:+-a$a}
  NAMES+=("$P-bbfrozen$suf-vs-d48-g4" "$P$suf-vs-d48-g4")
done
# cost per model (relative): deepnorm/keel run each skip twice (zeroed writes and bypass); attnres/moda carry big buffers
mapfile -t GROUPS_ < <($PY - "$WK" "${NAMES[@]}" <<'EOF'
import sys
G, names = int(sys.argv[1]), sys.argv[2:]
cost = lambda n: 2.0 if ("deepnorm" in n or "keel" in n) else 1.5 if ("attnres" in n or "moda" in n) else 1.0
bins = [[0.0, []] for _ in range(G)]
for n in sorted(names, key=cost, reverse=True):
    b = min(bins, key=lambda x: x[0]); b[0] += cost(n); b[1].append(n)
for c, ns in bins:
    print(" ".join(ns))
EOF
)
pids=()
for r in $(seq 0 $((WK - 1))); do
  [ -z "${GROUPS_[$r]:-}" ] && continue
  echo "worker $r (GPU ${DEVS[$((r % G))]}): ${GROUPS_[$r]}"
  CUDA_VISIBLE_DEVICES=${DEVS[$((r % G))]} $PY scripts/della/depthmed.py "$OUT" "$CFG" ${GROUPS_[$r]} > "$LOGD/worker$r.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
kill $SMI 2>/dev/null || true
for f in "$LOGD"/worker*.log; do echo "== $f"; grep -E "wrote|WARN|FAILED|all done|Error" "$f" | tail -5; done
exit $rc
