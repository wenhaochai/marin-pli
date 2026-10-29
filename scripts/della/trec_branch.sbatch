#!/bin/bash
#SBATCH --job-name=trec-br-a
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/trec/logs/slurm/%x_%j.out
#FETCHABLE self-contained: the arm comes from the job name (trec-br-a | trec-br-b | trec-br-c); needs nothing from the submit shell
# What sets how fast a batch is forgotten (owner popup 2026-10-29)? Figure 9: an early batch loses 81% of its imprint
# within ~500 steps, and of steps, sum of lr and sum of lr^2 since the read, sum of lr lines up d12 and rsqrtcd best.
# Here the d12-ck248 run is continued from its step-2480 checkpoint (full trainer state: step, optimizer, data position)
# at a constant lr for N steps with lr * N = 5 fixed: a 0.01 x 500, b 0.005 x 1000, c 0.0025 x 2000 (sum lr^2 0.05,
# 0.025, 0.0125). A checkpoint every 124 steps; each scores every batch read at steps 2232-2479 and 256 unread ones.
# Arm a also scores the d12-ck248 checkpoints 2232 (before those batches) and 2480 (the branch point) the same way.
set -uo pipefail
WT=/scratch/gpfs/GROUP/USER/project/marin-trec-rsqrt
SC=/scratch/gpfs/GROUP/USER/project/marin-trec   # scorer (branch trec)
PY=/scratch/gpfs/GROUP/USER/project/marin/.venv/bin/python
cd $WT
source scripts/della/job_tmpdir.sh
if [ -n "${FETCHED_JOB_ID:-}" ]; then NAME=${FETCHED_JOB_NAME:-$(sacct -X -n -j "$FETCHED_JOB_ID" -o JobName%80 | head -1 | tr -d ' ')}; else NAME=$SLURM_JOB_NAME; fi
case $NAME in trec-br-a) MUL=0.5; N=500;; trec-br-b) MUL=0.25; N=1000;; trec-br-c) MUL=0.125; N=2000;; *) echo "job name '$NAME' must be trec-br-a|b|c"; exit 2;; esac
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
module load proxy/default 2>/dev/null || true   # wandb only
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_PROJECT=marin-della XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
if [ "${WANDB_MODE:-}" != offline ] && ! curl -s -o /dev/null --max-time 20 https://api.wandb.ai/healthz; then
  export WANDB_MODE=offline; echo "W&B not reachable through the proxy: running offline (sync later)"
fi
unset VARIANT SMOKE_STEPS TOTAL_STEPS INIT_FROM CPT_TAG RUN_TAG SEED LR_MUL WARMUP PRECISION EMA_BETA DEVICE_TAG LR_SCHEDULE KEEP_EVERY
BASE=$MARIN_PREFIX/speedrun/muonh-qwen3-130m-della4xh100-d12-ck248/2026.09.13/checkpoints
[ -d $BASE/step-2480 ] || { echo "NO BRANCH CHECKPOINT $BASE/step-2480"; exit 4; }
export SIZE=130m LAYERS=12 NUM_GPUS=4 LR_SCHEDULE=constant LR_MUL=$MUL KEEP_EVERY=124 INIT_FROM=$BASE/step-2480 CPT_TAG=-br2480
src() { local p=$1; for d in "$1"/lib/*/src; do p=$p:$d; done; echo "$p"; }
rid=muonh-qwen3-130m-della4xh100-d12-constant-ck124-lrm${MUL}-br2480
out=$MARIN_PREFIX/speedrun/$rid/2026.09.13
O=/scratch/gpfs/GROUP/USER/project/trec/out/ck/branch
T1=$((2480 + N))
nvidia-smi -L
echo "arm $NAME lr $(python3 -c "print(0.02*$MUL)") x $N steps, run $rid $(date +%F_%T)"

# 1. smoke: 10 steps from the branch point into a throwaway run; it must start from the checkpoint (loss near 3.4, not 10)
if [ "$(cat "$out/.executor_status" 2>/dev/null || true)" != SUCCESS ] && [ ! -d "$out/checkpoints" ]; then
  slog=logs/trec_br_${JID}_smoke.log
  rc=0; TOTAL_STEPS=2490 KEEP_EVERY=0 RUN_TAG=-j${JID}s PYTHONPATH=$(src $WT) timeout 30m $PY -m experiments.references.della_muonh_qwen3_scaling > $slog 2>&1 || rc=$?
  for d in $MARIN_PREFIX/speedrun/muonh-qwen3-130m-della4xh100-d12-constant-lrm${MUL}-br2480-j${JID}s; do [ -d "$d" ] && rm -rf -- "$d" "$MARIN_PREFIX/tmp/checkpoints-temp$d"; done
  loss=$(grep -o "postfix:loss=[0-9.]*" $slog | tail -1 | cut -d= -f2)
  grep -q "Initializing from" $slog && [ -n "$loss" ] && python3 -c "import sys; sys.exit(0 if float('$loss') < 4 else 1)" || rc=${rc/#0/5}
  [ $rc -eq 0 ] || { echo "SMOKE FAILED rc=$rc loss=${loss:-none} ($slog)"; exit 3; }
  echo "smoke passed: started from step 2480, loss $loss $(date +%T)"
fi

# 2. training
if [ "$(cat "$out/.executor_status" 2>/dev/null || true)" != SUCCESS ]; then
  rc=0; TOTAL_STEPS=$T1 PYTHONPATH=$(src $WT) $PY -m experiments.references.della_muonh_qwen3_scaling > logs/trec_br_${JID}_train.log 2>&1 || rc=$?
  echo "training rc=$rc $(date +%T)"
  [ $rc -eq 0 ] || exit $rc
fi

# 3. scoring: batches 2232-2479 and 256 unread ones at every kept checkpoint (arm a: also d12-ck248 steps 2232 and 2480)
cd $SC && unset LR_SCHEDULE LAYERS SIZE NUM_GPUS KEEP_EVERY LR_MUL INIT_FROM CPT_TAG
cks=$(ls -d $out/checkpoints/step-* | sort -V)
[ $NAME = trec-br-a ] && cks="$BASE/step-2232 $BASE/step-2480 $cks"
fail=0
for c in $cks; do
  s=${c##*-}; dst=$O/$rid/step-$s; [[ $c == $BASE/* ]] && dst=$O/base/step-$s
  echo "score $c $(date +%T)"
  rc=0; TRAIN_RANGE=2232:2480 N_TAIL=256 CKPT=$c MODEL=130m-d12 STEPS=all OUT=$dst PYTHONPATH=$(src $SC) $PY -m experiments.references.trec_score > /dev/null 2>> $O/$rid.err || rc=$?
  [ $rc -eq 0 ] || { echo "scoring $s rc=$rc"; [ $rc -eq 143 ] && exit 143; fail=1; }
done
echo "DONE fail=$fail $(date +%T)"
exit $fail
