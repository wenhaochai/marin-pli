#!/bin/bash
#SBATCH --job-name=july-baseline-h100x4
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# July Baseline (#6882) reproduction on 4xH100: single JAX process, Fray LocalClient.
# Data must already be tokenized on a login node (--run_only tokenized/).
# DIM selects the cell (512/768/1024); ATTN selects the GPU attention kernel.
set -euo pipefail
cd /scratch/gpfs/KARTHIKN/wc9403/project/marin-july

module load proxy/default  # wandb only
export MARIN_PREFIX=/scratch/gpfs/KARTHIKN/wc9403/marin_store
export HF_HOME=/scratch/gpfs/KARTHIKN/wc9403/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export DIM=${DIM:-512} ATTN=${ATTN:-gpu_fa4_cute} NUM_GPUS=4
# If a tokenize dependency is not finished yet, this job takes it over offline; keep it multi-core.
export ZEPHYR_SUBPROCESS=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

nvidia-smi -L
.venv/bin/python -c 'import jax; print("jax", jax.__version__, "devices", jax.devices())'

.venv/bin/python -m experiments.grug.moe.della_july_baseline
