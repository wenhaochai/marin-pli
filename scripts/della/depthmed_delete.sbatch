#!/bin/bash
#SBATCH --job-name=depthmed-delete
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=48G   # two worker processes, each loads a ~13 GB float32 model
#SBATCH --time=01:01:00
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
#FETCHABLE: self-contained (every path below is literal; nothing comes from the submit shell)

# depthmed.py --delete (coordinator follow-up 2026-10-19, after PracRemove.lean): adds skip_ce_delete to the 20 JSONs in
# tmp/pls_status/depthmed. Copies for preln, sandwich, moda, attnres (= skip_ce) and deepnorm, keel (= skip_ce_bypass);
# 48 deletion runs each for hc, mhc (plus the renumbered variant), lns and attnres_block, ord and sep: two workers on one
# GPU (one alone leaves it near idle between dispatches). Smoke: depthmed_delete_smoke.sbatch.
set -uo pipefail
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
WT=/scratch/gpfs/GROUP/USER/project/marin-pls-auto10
OUT=/scratch/gpfs/GROUP/USER/tmp/pls_status/depthmed
LOGD=$OUT/logs/$JID
cd $WT
source scripts/della/pls_env.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 JAX_PLATFORMS=cuda
mkdir -p "$LOGD"
echo "job $JID on $(hostname): GPU ${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv -l 30 > "$LOGD/gpu_util.csv" 2>&1 &
SMI=$!
P=muonh-qwen3-130m-della4xh100-pls1-sep
W0=() W1=()
for a in "" preln moda attnres deepnorm keel; do W0+=("$P-bbfrozen${a:+-a$a}-vs-d48-g4" "$P${a:+-a$a}-vs-d48-g4"); done
W0+=("$P-bbfrozen-aattnres_block-vs-d48-g4" "$P-bbfrozen-ahc-vs-d48-g4" "$P-bbfrozen-amhc-vs-d48-g4" "$P-bbfrozen-alns-vs-d48-g4")
W1+=("$P-aattnres_block-vs-d48-g4" "$P-ahc-vs-d48-g4" "$P-amhc-vs-d48-g4" "$P-alns-vs-d48-g4")
$PY scripts/della/depthmed.py "$OUT" "$OUT/cfg" "${W0[@]}" --delete > "$LOGD/worker0.log" 2>&1 &
p0=$!
$PY scripts/della/depthmed.py "$OUT" "$OUT/cfg" "${W1[@]}" --delete > "$LOGD/worker1.log" 2>&1 &
p1=$!
rc=0
wait $p0 || rc=$?
wait $p1 || rc=$?
kill $SMI 2>/dev/null || true
for f in "$LOGD"/worker*.log; do echo "== $f"; grep -E "wrote|copied|base CE|WARN|FAILED|all done|Error" "$f" | tail -30; done
exit $rc
