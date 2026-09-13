#!/bin/bash
#SBATCH --job-name=delphi-h100x4
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# Delphi rung (BUDGET=3e18|9e18|2e19) on 4xH100: single JAX process, Fray LocalClient.
# Data must already be tokenized under MARIN_PREFIX (scripts/della/prep_delphi_data.py).
set -euo pipefail
cd /scratch/gpfs/KARTHIKN/wc9403/project/marin

module load proxy/default  # wandb only
export MARIN_PREFIX=${MARIN_PREFIX:-/scratch/gpfs/KARTHIKN/wc9403/marin_store_big}
export HF_HOME=/scratch/gpfs/KARTHIKN/wc9403/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export BUDGET=${BUDGET:-3e18}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

nvidia-smi -L
.venv/bin/python -c 'import jax; print("jax", jax.__version__, "devices", jax.devices())'

.venv/bin/python -m experiments.references.della_delphi_ladder
