#!/bin/bash
#SBATCH --job-name=sv-convert
#SBATCH --account=group
#SBATCH --qos=short
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=1-4
#SBATCH --output=logs/slurm/%x_%A_%a.out

# Converts fineweb-edu-10B and the 16 Paloma validation caches to the truncated Marin tokenizer
# (experiments/references/convert_small_vocab_cache.py); array task i takes K from the list below. CPU only.
# Bytes (task 5, ~4.7x the tokens) ran with --array=5 --time=03:00:00 (job 15211747).
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
export JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
KS=(0 64000 32000 16000 8000 256)   # 256: bytes, the tokenizer with no merges (array task 5)
export K=${KS[$SLURM_ARRAY_TASK_ID]}
echo "K=$K"
.venv/bin/python -m experiments.references.convert_small_vocab_cache
