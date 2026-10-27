#!/bin/bash
#SBATCH --job-name=fv-convert
#SBATCH --account=group
#SBATCH --qos=short
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=logs/slurm/%x_%j.out

# Converts fineweb-edu-10B and the 16 Paloma validation caches to the frequency set f1000 (owner popup 2026-10-27,
# "B 按频率选词表"; small_vocab_tokenizer.py build-freq 1000, 38,638 tokens) with convert_small_vocab_cache.py. CPU only.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
source scripts/della/job_tmpdir.sh
export JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export VOCAB_SET=f1000
.venv/bin/python -m experiments.references.convert_small_vocab_cache
