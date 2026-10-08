#!/bin/bash
#SBATCH --job-name=pls-d48-paths1
#SBATCH --account=pli
#SBATCH --partition=pli
#SBATCH --qos=pli-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=01:01:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out

# Diagnostic (2026-10-08), 1-GPU companion of pls_d48_paths_gpu.sbatch. On CPU, fold and scan_via agree exactly for the
# T2 baseline and the d48 probes checkpoints (T=4096, 1 or 8 devices); on GPU the d48 runs log eval/L47 7.18 vs
# eval/loss 5.31 on the same weights, and every readout from layer 0 sits at 7.2-8.4. Every d48 run used PDP=4
# (4 sequences per GPU), every 6/12-layer pls run 32, so the batch per GPU is varied here: 4, then 16.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PATHS_GPU=1
nvidia-smi -L
F="Unknown leaf|^W1|^I1"
echo "== probes ckpt step 4958, 1 GPU, T=4096, B=4"
timeout 15m $PY scripts/della/pls_d48_paths_probe_ckpt.py 4096 4 2>&1 | grep -v -E "$F" | tail -8
echo "== T2 baseline ckpt step 599, 1 GPU, T=4096, B=4 (VANILLA and JAX_FLASH)"
timeout 15m $PY scripts/della/pls_d48_paths_cpu.py 4096 4 2>&1 | grep -v -E "$F" | tail -12
echo "== probes ckpt step 4958, 1 GPU, T=4096, B=16"
timeout 15m $PY scripts/della/pls_d48_paths_probe_ckpt.py 4096 16 2>&1 | grep -v -E "$F" | tail -8
