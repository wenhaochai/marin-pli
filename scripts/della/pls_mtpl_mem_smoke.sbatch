#!/bin/bash
#SBATCH --job-name=mtpl-mem-smoke
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-gpu=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out

# Memory check for the MTP ladder after 520m's 8-GPU smoke 15591472 asked one 90.80 GiB allocation (VARIANT=pls at 32
# sequences per microbatch). With the pls microbatches (520m 16, 1_2b 8), run the two largest configurations, each NTP +
# MTP (one more layer and head), as smokes with the per-layer eval on, like pls_h100x8_single.sbatch's smoke.
# MEMSMOKE_FRAC (default 0.9, the real runs' XLA_PYTHON_CLIENT_MEM_FRACTION); MEMSMOKE_STEPS "<1_2b> <520m>" (default
# "30 30"). 1_2b at 32 per microbatch passed a 30-step smoke and ran out of memory at step ~90-100 (15591482, one
# 65.84 GiB allocation, no checkpoint saved yet), so the long setting runs past step 100 and one save (every 10 min).
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls-mtp
source scripts/della/pls_env.sh
LAUNCH=experiments.references.della_muonh_qwen3_scaling
unset SMOKE_STEPS TOTAL_STEPS INIT_FROM PDP PLS_MON RUN_TAG PROBE_FROM
module load proxy/default 2>/dev/null || true
export MARIN_PREFIX=${MARIN_PREFIX:-/scratch/gpfs/GROUP/USER/marin_store_big}
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della WANDB_MODE=offline
export VARIANT=pls XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMSMOKE_FRAC:-0.9}
read -r S12 S52 <<< "${MEMSMOKE_STEPS:-30 30}"
nvidia-smi -L
for spec in "SIZE=1_2b NUM_GPUS=8 PLS_W=0 PLS_EVAL=1 PLS_MTP_W=0.3 SMOKE_STEPS=$S12" "SIZE=520m NUM_GPUS=8 PLS_W=0 PLS_EVAL=1 PLS_MTP_W=0.3 SMOKE_STEPS=$S52"; do
  tag=-j${SLURM_JOB_ID}m; log=logs/slurm/mtpl-mem-smoke_${SLURM_JOB_ID}_${spec%% *}.log; log=${log/SIZE=/}
  t0=$SECONDS; r=0
  # shellcheck disable=SC2086
  env $spec RUN_TAG=$tag timeout 35m "$PY" -m $LAUNCH > "$log" 2>&1 || r=$?
  echo "[$spec] rc=$r in $(( (SECONDS - t0) / 60 )) min; $(grep -m1 'per-device microbatch' "$log"); OOM lines: $(grep -c RESOURCE_EXHAUSTED "$log")"
  grep -E 'Progress on:train' "$log" | tr '\r' '\n' | grep -oE 'train [^ ]+ rate:[^ ]+' | tail -1
  grep -oE 'eval loss: [0-9.]+' "$log" | tail -1
  echo "checkpoint saves: $(grep -c 'Saving temporary checkpoint' "$log"); last: $(grep -oE 'Saving temporary checkpoint at step [0-9]+' "$log" | tail -1)"
  rid=$(sed -n 's/^output: //p' "$log" | head -1); rid=$(basename "$(dirname "$rid")")
  if [[ -n $rid && $rid == *"$tag"-smoke[0-9]* ]]; then
    rm -rf -- "$MARIN_PREFIX/speedrun/$rid" "$MARIN_PREFIX/tmp/checkpoints-temp$MARIN_PREFIX/speedrun/$rid"
  fi
done
