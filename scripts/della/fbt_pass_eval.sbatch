#!/bin/bash
#SBATCH --job-name=fbt-pass-eval
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/%x_%j.out

# Eval-only FBT diagnostic (experiments/references/fbt_pass_eval.py): Paloma under pass1 / pass2 / pass3 /
# shuffled-feedback / plain-prefix regimes for one checkpoint. CKPT (and the FBT_* flags of that run) come from the
# environment; SIZE defaults to 300m. A probe, hence pli-cp.
set -uo pipefail
cd /scratch/gpfs/KARTHIKN/wc9403/project/marin
module load proxy/default
export MARIN_PREFIX=${MARIN_PREFIX:-/scratch/gpfs/KARTHIKN/wc9403/marin_store_big}
export HF_HOME=/scratch/gpfs/KARTHIKN/wc9403/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0
nvidia-smi -L
echo "CKPT=$CKPT SIZE=${SIZE:-300m} MODES=${MODES:-default}"
.venv/bin/python -m experiments.references.fbt_pass_eval
echo "EVAL DONE $(date +%H:%M:%S) rc=$?"
