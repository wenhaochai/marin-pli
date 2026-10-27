#!/bin/bash
#SBATCH --job-name=pls-trace-arm
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
#FETCHABLE: self-contained (the knobs come from the job name, nothing from the submit shell)
# Early-training trace (2026-10-27): which part of the new stack moves the Pre-LN 48-layer probes model? w334 (fa4thd,
# lse kernel, gradient-in-forward) sits +0.02 to +0.05 above the JAX-flash seed-0 run at steps 20-300 and +0.005 at the
# end, while its separate-heads twin w333 tracks seed 0 within 0.008 all run. One arm = the probes model
# (PLS_DETACH_BB=1), seed 0, on the full 4959-step schedule (SMOKE_STEPS=4959: own run id, no eval), stopped by timeout
# after 70 min (about 300 steps); its W&B history (train/pls/L47) is compared offline with the seed-0 run's.
# Job name pls-trace-arm-<knobs>:
#   off   no speed knobs (JAX flash attention, XLA CE): the old stack on today's code
#   thd   ATTN_BACKEND=fa4thd only
#   ce    A1, gradient-in-forward and the lse kernel (the CE side of the new stack)
#   all   thd and ce together (w334's first segment)
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
module load proxy/default 2>/dev/null || true
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VARIANT=pls WANDB_PROJECT=marin-della
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 LOG_JAXPR=0
curl -s -o /dev/null --max-time 20 https://api.wandb.ai/healthz || export WANDB_MODE=offline
trap pls_tmp_cleanup EXIT
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
NAME=${FETCHED_JOB_NAME:-$SLURM_JOB_NAME}
knobs=${NAME#pls-trace-arm-}
CE="LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1"
case $knobs in
  off) ON="";;
  thd) ON="ATTN_BACKEND=fa4thd";;
  ce) ON="$CE";;
  all) ON="$CE ATTN_BACKEND=fa4thd";;
  *) echo "unknown knobs in '$NAME'"; exit 2;;
esac
BASE="SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=1 PLS_EVAL=1 PARAM_MAP=vocab NUM_GPUS=4 LAYERS=48 PDP=4 ARCH=preln SMOKE_STEPS=4959"
echo "trace $NAME: knobs [$ON], GPUs ${CUDA_VISIBLE_DEVICES:-all}, start $(date +%T)"; nvidia-smi -L || true
# shellcheck disable=SC2086
env $BASE $ON RUN_TAG=-j${JID}tr timeout 70m "$PY" -m experiments.references.della_muonh_qwen3_scaling \
  > logs/slurm/${NAME}_${JID}_tr.log 2>&1 || echo "trace rc=$? (124 = stopped by the timeout, as planned)"
grep -E "First train step completed|Traceback|Error" logs/slurm/${NAME}_${JID}_tr.log | tail -3 || true
echo "trace done $(date +%T)"
out=$MARIN_PREFIX/speedrun
for d in "$out"/*-j${JID}tr* "$MARIN_PREFIX"/tmp/checkpoints-temp"$out"/*-j${JID}tr*; do
  [[ -n $JID && $d == *-j${JID}tr* && -d $d ]] && rm -rf -- "$d"
done
echo done
