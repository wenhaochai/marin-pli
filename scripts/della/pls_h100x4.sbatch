#!/bin/bash
#SBATCH --job-name=pls
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=24G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out

# One muonh_qwen3 size with VARIANT=pls (SIZE, PLS_W, ... via --export) on 4xH100, from the pls worktree with its own
# sources first on PYTHONPATH (scripts/della/pls_env.sh). Resumable: rerunning with the same env continues the run.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh

module load proxy/default  # wandb only
export MARIN_PREFIX=${MARIN_PREFIX:-/scratch/gpfs/GROUP/USER/marin_store_big}
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export VARIANT=${VARIANT:-pls}
export SIZE=${SIZE:-130m}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

nvidia-smi -L
$PY -c 'import jax, levanter; print("jax", jax.__version__, "devices", jax.devices(), "levanter", levanter.__file__)'
$PY -m experiments.references.della_muonh_qwen3_scaling
