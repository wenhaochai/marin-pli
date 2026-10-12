#!/bin/bash
#SBATCH --job-name=vo-passes
#SBATCH --account=group
#SBATCH --qos=short
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=1-3
#SBATCH --output=logs/slurm/%x_%A_%a.out

# Builds the 8 passes of the 300m repeated subsets for Q10 (experiments/references/vocab_materialize_passes.py; owner,
# 2026-10-11): task 1 the Marin tokenizer reshuffled each pass, task 2 the 8K truncation reshuffled, task 3 the Marin
# tokenizer reshuffled and re-segmented with BPE-dropout 0.1. CPU only.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
source scripts/della/job_tmpdir.sh   # TMPDIR under scratch, per task, removed at exit (never the shared /tmp)
export JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=16
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
unset VOCAB_K MODE P PASS_DATA VARIANT
case $SLURM_ARRAY_TASK_ID in
  1) export MODE=rs ;;
  2) export MODE=rs VOCAB_K=8000 ;;
  3) export MODE=bpd P=0.1 ;;
esac
echo "task $SLURM_ARRAY_TASK_ID: MODE=$MODE VOCAB_K=${VOCAB_K:-} P=${P:-}"
SIZE=300m .venv/bin/python -m experiments.references.vocab_materialize_passes
