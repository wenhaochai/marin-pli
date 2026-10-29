#!/bin/bash
#SBATCH --job-name=trec-ck-d12
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable-partial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=48G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/trec/logs/slurm/%x_%j.out
#FETCHABLE self-contained: the arm comes from the job name (trec-ck-d12 | trec-ck-rsqrtcd); needs nothing from the submit shell
# Imprint and decay of each batch (owner via the controller 2026-10-29: the experiment sessions add experiments). Figure 8
# showed that the weight c of an update in the final weights does not set how well the final model fits its batch. Here
# the 130m d12 run (trec-ck-d12: its own schedule; trec-ck-rsqrtcd: lr 0.02 / sqrt(1 + 0.0092 t), linear to 0 over the
# last fifth) is trained again keeping a checkpoint every 248 steps (KEEP_EVERY=248, 20 checkpoints), and each checkpoint
# scores 1 in 16 training batches (blocks of 8 steps every 128) and 256 unread ones; the final one scores every batch
# as in Figure 1. Steps: 1. 30-step smoke; 2. training (levanter resumes from its last checkpoint); 3. scoring (chunks
# present are skipped). Resumable: wq --partial (a stop by signal exits 143 and requeues).
set -uo pipefail
WT=/scratch/gpfs/GROUP/USER/project/marin-trec-rsqrt
SC=/scratch/gpfs/GROUP/USER/project/marin-trec   # scorer (branch trec)
PY=/scratch/gpfs/GROUP/USER/project/marin/.venv/bin/python
cd $WT
source scripts/della/job_tmpdir.sh
if [ -n "${FETCHED_JOB_ID:-}" ]; then NAME=${FETCHED_JOB_NAME:-$(sacct -X -n -j "$FETCHED_JOB_ID" -o JobName%80 | head -1 | tr -d ' ')}; else NAME=$SLURM_JOB_NAME; fi
case $NAME in trec-ck-d12) SCHED=; TAG=;; trec-ck-rsqrtcd) SCHED=rsqrtcd; TAG=-rsqrtcd;; *) echo "job name '$NAME' must be trec-ck-d12 or trec-ck-rsqrtcd"; exit 2;; esac
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
module load proxy/default 2>/dev/null || true   # wandb only
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_PROJECT=marin-della XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
if [ "${WANDB_MODE:-}" != offline ] && ! curl -s -o /dev/null --max-time 20 https://api.wandb.ai/healthz; then
  export WANDB_MODE=offline; echo "W&B not reachable through the proxy: running offline (sync later)"
fi
unset VARIANT SMOKE_STEPS TOTAL_STEPS INIT_FROM CPT_TAG RUN_TAG SEED LR_MUL WARMUP PRECISION EMA_BETA DEVICE_TAG LR_SCHEDULE
export SIZE=130m LAYERS=12 NUM_GPUS=4 KEEP_EVERY=248
[ -n "$SCHED" ] && export LR_SCHEDULE=$SCHED
src() { local p=$1; for d in "$1"/lib/*/src; do p=$p:$d; done; echo "$p"; }
rid=muonh-qwen3-130m-della4xh100-d12$TAG-ck248
out=$MARIN_PREFIX/speedrun/$rid/2026.09.13
O=/scratch/gpfs/GROUP/USER/project/trec/out/ck/$rid
nvidia-smi -L
echo "arm ${SCHED:-d12} run $rid $(date +%F_%T)"

# 1. smoke (only before the first training start)
if [ "$(cat "$out/.executor_status" 2>/dev/null || true)" != SUCCESS ] && [ ! -d "$out/checkpoints" ]; then
  sid=muonh-qwen3-130m-della4xh100-d12$TAG-ck10-j${JID}s-smoke30   # the smoke keeps a checkpoint every 10 steps: the keep path is tested
  rc=0; SMOKE_STEPS=30 RUN_TAG=-j${JID}s KEEP_EVERY=10 PYTHONPATH=$(src $WT) timeout 40m $PY -m experiments.references.della_muonh_qwen3_scaling > logs/trec_ck_${JID}_smoke.log 2>&1 || rc=$?
  kept=$(ls -d $MARIN_PREFIX/speedrun/$sid/2026.09.13/checkpoints/step-{10,20} 2>/dev/null | wc -l)
  [ -d "$MARIN_PREFIX/speedrun/$sid" ] && rm -rf -- "$MARIN_PREFIX/speedrun/$sid" "$MARIN_PREFIX/tmp/checkpoints-temp$MARIN_PREFIX/speedrun/$sid"
  [ $rc -eq 0 ] && [ "$kept" = 2 ] || rc=${rc/#0/5}
  [ $rc -eq 0 ] || { echo "SMOKE FAILED rc=$rc (logs/trec_ck_${JID}_smoke.log)"; exit 3; }
  echo "smoke passed $(date +%T)"
fi

# 2. training
if [ "$(cat "$out/.executor_status" 2>/dev/null || true)" != SUCCESS ]; then
  rc=0; PYTHONPATH=$(src $WT) $PY -m experiments.references.della_muonh_qwen3_scaling > logs/trec_ck_${JID}_train.log 2>&1 || rc=$?
  echo "training rc=$rc $(date +%T)"
  [ $rc -eq 0 ] || exit $rc
fi

# 3. scoring: every kept checkpoint on a 1-in-16 sample, the final one on every batch
cd $SC && unset LR_SCHEDULE LAYERS SIZE NUM_GPUS KEEP_EVERY
fail=0
for c in $(ls -d $out/checkpoints/step-* | sort -t- -k2 -n); do
  s=${c##*-}
  if [ "$s" = 4958 ]; then env_s="SAMPLE=1 N_TAIL=512"; else env_s="SAMPLE=16 BLOCK=8 N_TAIL=256"; fi
  echo "score step $s [$env_s] $(date +%T)"
  rc=0; env $env_s CKPT=$c MODEL=130m-d12 STEPS=all OUT=$O/step-$s PYTHONPATH=$(src $SC) $PY -m experiments.references.trec_score > /dev/null 2>> $O.err || rc=$?
  [ $rc -eq 0 ] || { echo "scoring step $s rc=$rc"; [ $rc -eq 143 ] && exit 143; fail=1; }
done
echo "DONE fail=$fail $(date +%T)"
exit $fail
