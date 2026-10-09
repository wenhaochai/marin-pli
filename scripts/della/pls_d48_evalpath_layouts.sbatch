#!/bin/bash
#SBATCH --job-name=pls-smoke-d48-layouts
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=32G
#SBATCH --time=01:01:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out

# Diagnostic (2026-10-09). On 4 GPUs with the trainer's FSDP layout (embed over data), the d48 probes checkpoint's readout
# L47 is 4.2529 (= main eval) in one program, 4.2378 in the 48-readout program (the trainer's value) and 8.07 under
# PLS_SCAN_FIX=loop (15255032); unsharded parameters agree everywhere (15235521). Here the same script under three
# parameter layouts: FSDP (repeat), whole on every GPU (PARAM_MAP=rep), vocab-sized tensors sharded only (PARAM_MAP=vocab).
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PATHS_GPU=1
nvidia-smi -L
for layout in 1 rep vocab; do
  echo "== 4 GPUs, T=4096, B=16, PATHS_SHARD=$layout"
  PATHS_SHARD=$layout timeout 17m $PY scripts/della/pls_d48_eval_path.py 4096 16 2>&1 | grep -E "param mapping|grug eval|with_seg|Readout|readout loss|Error" | cut -c1-400 || true
done
