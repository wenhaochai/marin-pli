#!/bin/bash
#SBATCH --job-name=pls-d48-evalpath8
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=64G
#SBATCH --time=01:01:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out

# Diagnostic (2026-10-08). On 1 H100 (15207626) the d48 probes checkpoint scores 4.253 on paloma c4_en's first 65536
# tokens through the baseline path and through readout L47, and the run's main eval logged c4_en 4.231; the run's
# per-layer eval logged eval/L47 7.18 (all sets) on the same weights. Here the run's own eval functions and its
# ReadoutTaggedEvaluator accumulation. Every d48 run trained on 8 GPUs (eval batch 4 per GPU = 32) and every 6/12-layer
# run, whose eval/L{last} equals eval/loss, on 4: so 8 GPUs at B=32 first, then 4 GPUs at B=16, then 1 GPU at B=4.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PATHS_GPU=1
nvidia-smi -L
F="Unknown leaf|^W1|^I1"
echo "== 8 GPUs, T=4096, B=32"
timeout 15m $PY scripts/della/pls_d48_eval_path.py 4096 32 2>&1 | grep -v -E "$F" | tail -25
echo "== 4 GPUs, T=4096, B=16"
CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 15m $PY scripts/della/pls_d48_eval_path.py 4096 16 2>&1 | grep -v -E "$F" | tail -25
echo "== 1 GPU, T=4096, B=4"
CUDA_VISIBLE_DEVICES=0 timeout 15m $PY scripts/della/pls_d48_eval_path.py 4096 4 2>&1 | grep -v -E "$F" | tail -25
