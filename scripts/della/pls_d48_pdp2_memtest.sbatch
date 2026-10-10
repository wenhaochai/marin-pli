#!/bin/bash
#SBATCH --job-name=pls-d48-pdp2-memtest
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=32G
#SBATCH --time=01:01:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out

# Memory test (2026-10-09). hc's 48-layer train step needs a 32.91 GiB temp buffer at microbatch 4 and died at step 257
# under the 0.95 pool (15266153). The launcher now halves the microbatch for hc, mhc, attnres, attnres_block and moda
# under PARAM_MAP. Here hc (GPUs 0-3) and attnres (GPUs 4-7), separate heads, run a 30-step smoke plus the per-layer eval
# in the real 0.95 pool. XLA dumps jit__train_step's buffer assignment, which gives the temp buffer exactly (hc at
# microbatch 4: 35339801088 bytes = 32.91 GiB); the pool needs two temp buffers beside the ~11.3 GiB state, so
# temp <= ~31 GiB is required and <= ~26 GiB leaves room for a second stray buffer. The step times compare with
# 16.3 s/step for hc at microbatch 4.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
module load proxy/default 2>/dev/null || true  # wandb only
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della WANDB_MODE=offline
unset TOTAL_STEPS CPT_TAG TIE PRECISION EMA_BETA DEVICE_TAG SEED DRY_RUN ARCH PLS_LOCAL PLS_SHAPE PLS_PCGRAD PLS_OFF PLS_ON PLS_PROBE HEADS_FROM LR_MUL WARMUP INIT_FROM PARAM_MAP_PDP
export VARIANT=pls SIZE=130m LAYERS=48 PDP=4 PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=0 PARAM_MAP=vocab PARAM_MAP_MEM_FRACTION=0.95 SMOKE_STEPS=30
nvidia-smi -L
names=(hc attnres); gpus=(0,1,2,3 4,5,6,7); pids=(); ids=()
for i in 0 1; do
  a=${names[$i]}
  out=$(ARCH=$a RUN_TAG=-j${SLURM_JOB_ID}$a DRY_RUN=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $PY -m experiments.references.della_muonh_qwen3_scaling 2>/dev/null | sed -n 's/^output: //p' || true)
  id=$(basename "$(dirname "$out")")
  [[ $id == *-j${SLURM_JOB_ID}${a}*-smoke30 ]] || { echo "bad run id '$id'"; exit 2; }
  ids+=("$id")
  dump=$PLS_JOB_TMPDIR/xla_$a; mkdir -p "$dump"
  XLA_FLAGS="--xla_dump_to=$dump --xla_dump_hlo_module_re=.*train_step.*" \
  ARCH=$a RUN_TAG=-j${SLURM_JOB_ID}$a CUDA_VISIBLE_DEVICES=${gpus[$i]} timeout --signal=TERM --kill-after=60s 50m \
    $PY -m experiments.references.della_muonh_qwen3_scaling > logs/slurm/pls-d48-pdp2-memtest_${SLURM_JOB_ID}_$a.log 2>&1 &
  pids+=($!)
done
rc=(0 0); for i in 0 1; do wait "${pids[$i]}" || rc[$i]=$?; done
for i in 0 1; do
  f=logs/slurm/pls-d48-pdp2-memtest_${SLURM_JOB_ID}_${names[$i]}.log
  echo "== ${names[$i]} exit ${rc[$i]}"
  grep -m1 "per-device microbatch" "$f" || true
  grep -E "RESOURCE_EXHAUSTED" "$f" | head -1 | cut -c1-250 || true
  tr '\r' '\n' < "$f" | grep -E "Progress on:train" | tail -3 | cut -c1-170 || true
  for b in "$PLS_JOB_TMPDIR/xla_${names[$i]}"/*train_step*buffer-assignment.txt; do
    [ -e "$b" ] || { echo "no buffer assignment dump"; continue; }
    echo "$(basename "$b" | cut -c1-90): $(grep -m1 -E '^Total bytes used' "$b") | temp: $(grep -E 'preallocated-temp' "$b" | grep -oE 'size [0-9]+' | head -1)"
  done
  # the smoke's eval: main loss and the last readout (the pair script's gate quantity)
  grep -E "eval/loss|eval/L47/loss" "$f" | tail -2 | cut -c1-200 || true
done
# the test's own outputs only (their exact tagged run ids)
for id in "${ids[@]}"; do
  [[ $id == *-j${SLURM_JOB_ID}*-smoke30 ]] || continue
  rm -rf -- "$MARIN_PREFIX/speedrun/$id" "$MARIN_PREFIX/tmp/checkpoints-temp$MARIN_PREFIX/speedrun/$id"
done
[ "${rc[0]}" = 0 ] && [ "${rc[1]}" = 0 ]
