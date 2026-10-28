#!/bin/bash
#SBATCH --job-name=trec-rsqrt
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
#FETCHABLE self-contained: the arm comes from the job name (trec-rsqrt | trec-rsqrtcd); needs nothing from the submit shell
# The equal-weight learning rate of kexue.fm/archives/11459 on the 130m depth control (owner popup 2026-10-28): the
# d12 run (12 layers, 4959 steps, lr 0.02 held for a fifth then linear to 0) with LR_SCHEDULE=rsqrt
# (lr 0.02 / sqrt(1 + 23 * 0.02^2 t)) or rsqrtcd (the same up to 80%, then linear to 0); code, data, seed and batch as d12
# (branch trec-rsqrt from d12's commit 4530d3bb). Steps: 1. a 30-step smoke copy (throwaway run id, output removed);
# 2. the run; 3. its TREC: the final checkpoint scored on every training batch and 512 unread ones by the trec branch's
# experiments/references/trec_score.py (MODEL=130m-d12: same shape, stream and steps). A rerun skips what is done.
set -uo pipefail
WT=/scratch/gpfs/GROUP/USER/project/marin-trec-rsqrt
SC=/scratch/gpfs/GROUP/USER/project/marin-trec   # scorer (branch trec)
PY=/scratch/gpfs/GROUP/USER/project/marin/.venv/bin/python
cd $WT
source scripts/della/job_tmpdir.sh
if [ -n "${FETCHED_JOB_ID:-}" ]; then NAME=${FETCHED_JOB_NAME:-$(sacct -X -n -j "$FETCHED_JOB_ID" -o JobName%80 | head -1 | tr -d ' ')}; else NAME=$SLURM_JOB_NAME; fi
case $NAME in trec-rsqrt) SCHED=rsqrt;; trec-rsqrtcd) SCHED=rsqrtcd;; *) echo "job name '$NAME' must be trec-rsqrt or trec-rsqrtcd"; exit 2;; esac
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
module load proxy/default 2>/dev/null || true   # wandb only
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_PROJECT=marin-della XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
if [ "${WANDB_MODE:-}" != offline ] && ! curl -s -o /dev/null --max-time 20 https://api.wandb.ai/healthz; then
  export WANDB_MODE=offline; echo "W&B not reachable through the proxy: running offline (sync later)"
fi
unset VARIANT SMOKE_STEPS TOTAL_STEPS INIT_FROM CPT_TAG RUN_TAG SEED LR_MUL WARMUP PRECISION EMA_BETA DEVICE_TAG
export SIZE=130m LAYERS=12 NUM_GPUS=4 LR_SCHEDULE=$SCHED
src() { local p=$1; for d in "$1"/lib/*/src; do p=$p:$d; done; echo "$p"; }   # a worktree's sources, not project/marin's
rid=muonh-qwen3-130m-della4xh100-d12-$SCHED
out=$MARIN_PREFIX/speedrun/$rid/2026.09.13
nvidia-smi -L
echo "arm $SCHED run $rid $(date +%F_%T)"

# 1. smoke
if [ "$(cat "$out/.executor_status" 2>/dev/null || true)" != SUCCESS ] && [ ! -d "$out/checkpoints" ]; then
  sid=$rid-j${JID}s-smoke30
  rc=0; SMOKE_STEPS=30 RUN_TAG=-j${JID}s PYTHONPATH=$(src $WT) timeout 40m $PY -m experiments.references.della_muonh_qwen3_scaling > logs/trec_rsqrt_${JID}_smoke.log 2>&1 || rc=$?
  [ -d "$MARIN_PREFIX/speedrun/$sid" ] && rm -rf -- "$MARIN_PREFIX/speedrun/$sid" "$MARIN_PREFIX/tmp/checkpoints-temp$MARIN_PREFIX/speedrun/$sid"
  [ $rc -eq 0 ] || { echo "SMOKE FAILED rc=$rc (logs/trec_rsqrt_${JID}_smoke.log)"; exit 3; }
  echo "smoke passed $(date +%T)"
fi

# 2. training (levanter resumes from the run's last checkpoint)
if [ "$(cat "$out/.executor_status" 2>/dev/null || true)" != SUCCESS ]; then
  rc=0; PYTHONPATH=$(src $WT) $PY -m experiments.references.della_muonh_qwen3_scaling > logs/trec_rsqrt_${JID}_train.log 2>&1 || rc=$?
  echo "training rc=$rc $(date +%T)"
  [ $rc -eq 0 ] || exit $rc
fi

# 3. TREC
c=$out/checkpoints/step-4958
[ -d "$c" ] || { echo "NO CHECKPOINT $c"; exit 4; }
cd $SC && unset LR_SCHEDULE LAYERS SIZE NUM_GPUS
rc=0; CKPT=$c MODEL=130m-d12 STEPS=all OUT=/scratch/gpfs/GROUP/USER/project/trec/out/$rid PYTHONPATH=$(src $SC) $PY -m experiments.references.trec_score || rc=$?
echo "DONE trec rc=$rc $(date +%T)"
exit $rc
