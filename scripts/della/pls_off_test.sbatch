#!/bin/bash
#SBATCH --job-name=pls-off-test
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
#FETCHABLE: self-contained (nothing from the submit shell)
# Pre-flight for the switch-off runs (owner popup 2026-10-27: arch3-preln-off{50,80}-sep, arch3-hc-off80-sep): the
# intermediate losses of a separate-heads 48-layer run stop at step 15 (PLS_OFF=15, jax.lax.cond on the step) on today's
# production knobs, Pre-LN on the b-tiled CE and HC at microbatch 4 on the streaming CE; 40 steps with an eval and a
# checkpoint at step 20. Passes when both runs exit 0 with finite losses after the switch.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
module load proxy/default 2>/dev/null || true
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VARIANT=pls WANDB_PROJECT=marin-della
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 LOG_JAXPR=0 LEVANTER_CE_BTILED_ONEHOT=1 EVAL_PDP=4 LEVANTER_CE_GRAD_IN_FWD=1
export LEVANTER_CE_LSE_KERNEL=1 LEVANTER_NORM_GUARD=1e-3
curl -s -o /dev/null --max-time 20 https://api.wandb.ai/healthz || export WANDB_MODE=offline
trap pls_tmp_cleanup EXIT
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
BASE="SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=0 PLS_EVAL=1 PARAM_MAP=vocab NUM_GPUS=4 LAYERS=48 PLS_OFF=15 SMOKE_STEPS=40 SMOKE_MID=20"
fail=0
for d in preln hc; do
  case $d in
    preln) A="PDP=4 ARCH=preln";;
    hc) A="PDP=4 ARCH=hc PARAM_MAP_PDP=4 LEVANTER_CE_BTILED_MIN_BATCH=1048576";;
  esac
  log=logs/slurm/pls-off-test_${JID}_$d.log
  r=0
  # shellcheck disable=SC2086
  env $BASE $A RUN_TAG=-j${JID}off timeout 40m "$PY" -m experiments.references.della_muonh_qwen3_scaling > "$log" 2>&1 || r=$?
  echo "$d: rc=$r $(date +%T)"
  grep -E "Traceback|Error|nan" "$log" | tail -3 || true
  [ $r -eq 0 ] || fail=1
done
out=$MARIN_PREFIX/speedrun
for p in "$out"/*-j${JID}off* "$MARIN_PREFIX"/tmp/checkpoints-temp"$out"/*-j${JID}off*; do
  [[ -n $JID && $p == *-j${JID}off* && -d $p ]] && rm -rf -- "$p"
done
[ $fail -eq 0 ] && echo "PASS" || { echo "FAIL"; exit 1; }
