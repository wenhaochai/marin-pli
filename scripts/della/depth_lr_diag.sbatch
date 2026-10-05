#!/bin/bash
#SBATCH --job-name=depth-lr-diag
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=24G
#SBATCH --time=01:01:00
#SBATCH --array=1-7%16
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%A_%a.out

# Why do the 48-layer runs stall near 6.6 nats (2026-10-05, Sandwich-LN and Pre-LN, separate heads and probes alike)?
# Plain baseline (no readout heads) at the 130m width, batch and recipe on a short schedule of TOTAL_STEPS (default 600)
# steps; array task i picks arm i of ARMS = "LAYERS:LR_MUL" (LAYERS 0 = the size's own 6). Arms 1-4: depth 6, 12, 24, 48
# at the recipe's learning rate; arms 5-7: 48 layers at 1/2, 1/4, 1/8 of it. Run ids end in -diag<steps>, so no full
# run ever resumes from these. One process on all 4 GPUs: a crash ends the job.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
unset SMOKE_STEPS INIT_FROM CPT_TAG TIE PRECISION EMA_BETA DEVICE_TAG DRY_RUN PLS_W PLS_SEP PLS_DETACH PLS_OFF PLS_PROBE SEED ARCH PDP WARMUP HEADS_FROM

module load proxy/default  # wandb only
export MARIN_PREFIX=${MARIN_PREFIX:-/scratch/gpfs/GROUP/USER/marin_store_big}
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
ARMS=(0:1 12:1 24:1 48:1 48:0.5 48:0.25 48:0.125)
arm=${ARMS[$(( ${SLURM_ARRAY_TASK_ID:?} - 1 ))]}
export VARIANT=baseline SIZE=130m LAYERS=${arm%%:*} LR_MUL=${arm##*:} TOTAL_STEPS=${TOTAL_STEPS:-600}
export RUN_TAG=-diag$TOTAL_STEPS
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
echo "depth lr diag: arm $SLURM_ARRAY_TASK_ID LAYERS=$LAYERS LR_MUL=$LR_MUL TOTAL_STEPS=$TOTAL_STEPS RUN_TAG=$RUN_TAG PY=$PY"
nvidia-smi -L
$PY -m experiments.references.della_muonh_qwen3_scaling
