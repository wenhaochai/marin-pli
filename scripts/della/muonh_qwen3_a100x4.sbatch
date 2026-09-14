#!/bin/bash
#SBATCH --job-name=muonh-qwen3-a100
#SBATCH --account=karthikn
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=24G
#SBATCH --time=23:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# A100 lane copy of muonh_qwen3_h100x4.sbatch: one size (SIZE=130m|300m|520m|1_2b, VARIANT=baseline|fbt) on
# 4xA100-80GB. The 4xA100 queue takes days per submission, so these run as single long jobs (--qos/--time on the
# command line) rather than 2h chains; DEVICE_TAG=a100 gives them their own run ids and output paths.
set -euo pipefail
cd /scratch/gpfs/KARTHIKN/wc9403/project/marin

module load proxy/default  # wandb only
export MARIN_PREFIX=${MARIN_PREFIX:-/scratch/gpfs/KARTHIKN/wc9403/marin_store_big}
export HF_HOME=/scratch/gpfs/KARTHIKN/wc9403/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export SIZE=${SIZE:-130m}
export VARIANT=${VARIANT:-baseline}
export DEVICE_TAG=a100
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

nvidia-smi -L
.venv/bin/python -c 'import jax; print("jax", jax.__version__, "devices", jax.devices())'

.venv/bin/python -m experiments.references.della_muonh_qwen3_scaling
