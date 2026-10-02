#!/bin/bash
#SBATCH --job-name=depth-ctrl
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=24G
#SBATCH --time=01:15:00
#SBATCH --array=1-5
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%A_%a.out

# Depth-matched controls for the per-layer readouts of VARIANT=pls: the plain muonh_qwen3 baseline at SIZE's width,
# batch, steps and schedule, with LAYERS = SLURM_ARRAY_TASK_ID (1..5). At every step the backbone compute of the
# d-layer control equals that of readout d-1 of the 6-layer pls run, so their curves compare step for step.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
unset SMOKE_STEPS TOTAL_STEPS INIT_FROM CPT_TAG TIE PRECISION EMA_BETA DEVICE_TAG DRY_RUN PLS_W PLS_SEP PLS_DETACH PLS_OFF PLS_PROBE RUN_TAG SEED

module load proxy/default  # wandb only
export MARIN_PREFIX=${MARIN_PREFIX:-/scratch/gpfs/GROUP/USER/marin_store_big}
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export VARIANT=baseline SIZE=${SIZE:-130m} LAYERS=${SLURM_ARRAY_TASK_ID:?}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
echo "depth control: SIZE=$SIZE LAYERS=$LAYERS PY=$PY"
nvidia-smi -L
$PY -m experiments.references.della_muonh_qwen3_scaling
