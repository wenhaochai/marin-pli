#!/bin/bash
#SBATCH --job-name=marin-tiny-h100
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=24G
#SBATCH --time=01:01:00
#SBATCH --output=logs/slurm/%x_%j.out

# Smoke test: Marin tutorial tiny Llama on N×H100 (DEVICE=h100x1 default; override --gres and DEVICE together), single JAX process, Fray LocalClient.
# Data must be tokenized on a login node first (scripts/della/prep_data.py): compute nodes
# cannot reach huggingface.co.
set -euo pipefail
cd /scratch/gpfs/KARTHIKN/wc9403/project/marin

module load proxy/default  # wandb only
export MARIN_PREFIX=/scratch/gpfs/KARTHIKN/wc9403/marin_store
export HF_HOME=/scratch/gpfs/KARTHIKN/wc9403/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export UV_CACHE_DIR=/scratch/gpfs/KARTHIKN/wc9403/.cache/uv
export UV_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

nvidia-smi -L
.venv/bin/python -c 'import jax; print("jax", jax.__version__, "devices", jax.devices())'

.venv/bin/python -m experiments.tutorials.train_tiny_model \
  --device ${DEVICE:-h100x1} --dataset ${DATASET:-wikitext} --version dev --run
