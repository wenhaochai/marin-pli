#!/bin/bash
#SBATCH --job-name=vo-toklosses
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=01:01:00
#SBATCH --comment=fetchable
#SBATCH --output=logs/slurm/%x_%j.out
#FETCHABLE self-contained: SPECS is set here; needs nothing from the submitting shell
# Per-position Paloma losses of the cvd seed-1 pair (full data, 8 passes) at their last step, for the B1 grouped costs.
# The previous pair took 7 minutes on one GPU (vo-toklosses 15470524).
export SPECS="muonh-qwen3-300m-della4xh100-cvd8k-p0.5-s1 11443;muonh-qwen3-300m-della4xh100-cvd8k-p0.5-rep8-s1 11443"
exec bash /scratch/gpfs/GROUP/USER/project/marin/scripts/della/vocab_token_losses.sbatch
