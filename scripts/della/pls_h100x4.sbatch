#!/bin/bash
#SBATCH --job-name=pls
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=24G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out

# One muonh_qwen3 size with VARIANT=pls on 4xH100, from the pls worktree with its own sources first on PYTHONPATH
# (scripts/della/pls_env.sh). Knobs read from the submission environment: SIZE (required), PLS_W, PLS_DETACH, PLS_SEP, PLS_OFF, PLS_PROBE, PLS_MON, PLS_EVAL, SEED,
# RUN_TAG. Every other launcher knob that would change the run (schedule length, init, precision, tying, EMA, smoke) is
# cleared, so a stray export in the submitting shell cannot turn this into a different run. Resumable: rerunning with
# the same knobs continues the run.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
unset SMOKE_STEPS TOTAL_STEPS INIT_FROM CPT_TAG TIE PRECISION EMA_BETA DEVICE_TAG DRY_RUN

module load proxy/default  # wandb only
export MARIN_PREFIX=${MARIN_PREFIX:-/scratch/gpfs/GROUP/USER/marin_store_big}
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export VARIANT=pls
export SIZE=${SIZE:?set SIZE (130m|300m|520m|1_2b)}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
echo "pls knobs: SIZE=$SIZE PLS_W=${PLS_W:-1} PLS_DETACH=${PLS_DETACH:-0} PLS_SEP=${PLS_SEP:-0} PLS_DETACH_BB=${PLS_DETACH_BB:-0} PLS_OFF=${PLS_OFF:-} PLS_PROBE=${PLS_PROBE:-0} PLS_MON=${PLS_MON:-16} PLS_EVAL=${PLS_EVAL:-1} SEED=${SEED:-0} RUN_TAG=${RUN_TAG:-} PY=$PY"

nvidia-smi -L
$PY -c 'import jax, levanter; print("jax", jax.__version__, "devices", jax.devices(), "levanter", levanter.__file__)'
$PY -m experiments.references.della_muonh_qwen3_scaling
