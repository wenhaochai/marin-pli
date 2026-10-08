#!/bin/bash
#SBATCH --job-name=pls-d48-evalpath
#SBATCH --account=pli
#SBATCH --partition=pli
#SBATCH --qos=pli-low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=64G
#SBATCH --time=01:01:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out

# Diagnostic (2026-10-08). On 1 H100 (15207626) the d48 probes checkpoint scores 4.253 on paloma c4_en's first 65536
# tokens through the baseline path and through readout L47, and the run's main eval logged c4_en 4.231; the run's
# per-layer eval logged eval/L47 7.18 (all sets) on the same weights. Here the run's own eval functions and its
# ReadoutTaggedEvaluator accumulation, at the run's eval batch (4 sequences per GPU, 4 GPUs), then on 1 GPU.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PATHS_GPU=1
nvidia-smi -L
F="Unknown leaf|^W1|^I1"
echo "== 4 GPUs, T=4096, B=16"
timeout 20m $PY scripts/della/pls_d48_eval_path.py 4096 16 2>&1 | grep -v -E "$F" | tail -25
echo "== 1 GPU, T=4096, B=4"
CUDA_VISIBLE_DEVICES=0 timeout 20m $PY scripts/della/pls_d48_eval_path.py 4096 4 2>&1 | grep -v -E "$F" | tail -25
