#!/bin/bash
#SBATCH --job-name=pls-d48-evalpath-cpu8
#SBATCH --account=group
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=01:01:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out

# Diagnostic (2026-10-08): pls_d48_eval_path.py with 8 host devices on CPU (every d48 run trained on 8 GPUs, every
# 6/12-layer run on 4); della-vis1's 121 GiB user cgroup kills it. T=128, B=16 (2 per device).
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
EVAL_EOS=${EVAL_EOS:-} EVAL_NDEV=8 JAX_PLATFORMS=cpu timeout 55m $PY scripts/della/pls_d48_eval_path.py 128 16 2>&1 | grep -v -E "Unknown leaf|^W1|^I1" | tail -25
