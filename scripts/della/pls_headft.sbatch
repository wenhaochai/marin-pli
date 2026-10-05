#!/bin/bash
#SBATCH --job-name=pls-headft
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=24G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
# Refit the readout heads of a finished VARIANT=pls run (launcher HEADS_FROM): load its weights, freeze the backbone,
# train the heads for TOTAL_STEPS steps. Knobs from the submission environment: SIZE, HEADS_FROM, TOTAL_STEPS (all
# required), HEADS_WARMUP, the finished run's own PLS_W, PLS_SEP, PLS_DETACH_BB, PLS_MON, PLS_EVAL, SEED, and
# SMOKE_STEPS for a smoke test. Every other launcher knob that would change the run is cleared.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
unset INIT_FROM CPT_TAG TIE PRECISION EMA_BETA DEVICE_TAG DRY_RUN LAYERS ARCH PDP PLS_LOCAL PLS_PROBE PLS_OFF PLS_DETACH RUN_TAG

module load proxy/default  # wandb only
export MARIN_PREFIX=${MARIN_PREFIX:-/scratch/gpfs/GROUP/USER/marin_store_big}
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export VARIANT=pls
export SIZE=${SIZE:?set SIZE} HEADS_FROM=${HEADS_FROM:?set HEADS_FROM} TOTAL_STEPS=${TOTAL_STEPS:?set TOTAL_STEPS}
[ -n "${SMOKE_STEPS:-}" ] && export RUN_TAG=-j${SLURM_JOB_ID}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
echo "headft knobs: SIZE=$SIZE HEADS_FROM=$HEADS_FROM TOTAL_STEPS=$TOTAL_STEPS HEADS_WARMUP=${HEADS_WARMUP:-100} SMOKE_STEPS=${SMOKE_STEPS:-} PLS_W=${PLS_W:-1} PLS_SEP=${PLS_SEP:-0} PLS_DETACH_BB=${PLS_DETACH_BB:-0} PLS_MON=${PLS_MON:-16} PLS_EVAL=${PLS_EVAL:-1} SEED=${SEED:-0} PY=$PY"
ls -d "$HEADS_FROM"/step-* | tail -1

nvidia-smi -L
$PY -m experiments.references.della_muonh_qwen3_scaling
