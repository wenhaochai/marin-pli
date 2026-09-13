#!/bin/bash
#SBATCH --job-name=bench-d512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=logs/slurm/%x_%j.out

# Step-time micro-benchmark of exact-math kernel choices for the July d512 run.
# Submit with account/partition/qos and the SEQ/BATCH/ATTN env for the target GPU, e.g.
#   sbatch --account=karthikn -p mig -c 1 --mem=16G --export=ALL,ATTN=none,SEQ=2048,BATCH=1 scripts/della/bench_d512.sbatch
#   sbatch --account=pli -p pli-c --qos=pli-short -c 8 --mem=48G scripts/della/bench_d512.sbatch
set -euo pipefail
cd /scratch/gpfs/KARTHIKN/wc9403/project/marin-july
export MARIN_PREFIX=/scratch/gpfs/KARTHIKN/wc9403/marin_store
export HF_HOME=/scratch/gpfs/KARTHIKN/wc9403/cache/huggingface HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/scratch/gpfs/KARTHIKN/wc9403/.cache/triton
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
nvidia-smi -L
.venv/bin/python -m scripts.della.bench_d512
