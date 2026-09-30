#!/bin/bash
#SBATCH --job-name=pls-readout-eval
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out

# Per-layer readout eval of an existing run's final checkpoint (e.g. a baseline, which has no eval/L{k} keys): the pls
# model (PLS_W=0, same parameter tree as the baseline) restores the checkpoint in full (INIT_FROM) with TOTAL_STEPS equal
# to the original schedule, so no training step runs and the trainer goes straight to its final hooks: the main eval
# (must reproduce the source run's final numbers) and eval/L{k}/... for every layer. SIZE, CKPT, STEPS, TAG from --export.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
unset SMOKE_STEPS TIE PRECISION EMA_BETA DEVICE_TAG DRY_RUN SEED RUN_TAG
module load proxy/default  # wandb only
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export VARIANT=pls PLS_W=0 SIZE=${SIZE:?} INIT_FROM=${CKPT:?} TOTAL_STEPS=${STEPS:?} CPT_TAG=-${TAG:?}
echo "readout eval: SIZE=$SIZE INIT_FROM=$INIT_FROM TOTAL_STEPS=$TOTAL_STEPS CPT_TAG=$CPT_TAG"
$PY -m experiments.references.della_muonh_qwen3_scaling
