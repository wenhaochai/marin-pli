#!/bin/bash
#SBATCH --job-name=pls-d48-paths
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=24G
#SBATCH --time=01:01:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out

# Diagnostic (2026-10-07): scripts/della/pls_d48_paths_cpu.py on GPU. The T2 48-layer checkpoint through the baseline
# path (fold) and the per-layer path (scan_via), plus a per-layer loop, with a real-length batch: on 1 GPU, then with
# the batch and parameters sharded over 4 GPUs as in training. On CPU the two paths agree exactly; at 48 layers on GPU
# eval/loss and eval/L47/loss of the same weights differ by up to 2.5 nats.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PATHS_GPU=1
nvidia-smi -L
echo "== 1 GPU, T=4096, B=2"
CUDA_VISIBLE_DEVICES=0 timeout 20m $PY scripts/della/pls_d48_paths_cpu.py 4096 2 2>&1 | grep -v -E "Unknown leaf|^W1|^I1" | tail -12
echo "== 4 GPUs, T=4096, B=8"
timeout 20m $PY scripts/della/pls_d48_paths_cpu.py 4096 8 2>&1 | grep -v -E "Unknown leaf|^W1|^I1" | tail -12
