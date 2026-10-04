#!/bin/bash
#SBATCH --job-name=ov-mem-eval
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out

# Seen vs unseen training data for one checkpoint (experiments/references/ov_memorization_eval.py). Array task i runs
# line i of TABLE: "SIZE MODEL DATA_EPOCHS CKPT [KEY=VALUE ...]" (extra KEY=VALUE pairs are exported, e.g. OV_M=1.28e6).
# An eval: pli-short (pli-cp is for smoke tests only, user rule 2026-10-04).
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
module load proxy/default
export MARIN_PREFIX=${MARIN_PREFIX:-/scratch/gpfs/GROUP/USER/marin_store_big}
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TABLE")
read -r SIZE MODEL DATA_EPOCHS CKPT extra <<<"$line"
export SIZE MODEL DATA_EPOCHS CKPT
for kv in ${extra:-}; do export "$kv"; done
nvidia-smi -L
echo "TASK $SLURM_ARRAY_TASK_ID: SIZE=$SIZE MODEL=$MODEL DATA_EPOCHS=$DATA_EPOCHS CKPT=$CKPT extra=${extra:-}"
.venv/bin/python -m experiments.references.ov_memorization_eval
rc=$?
echo "EVAL DONE $(date +%H:%M:%S) rc=$rc"
exit $rc
