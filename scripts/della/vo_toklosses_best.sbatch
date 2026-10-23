#!/bin/bash
#SBATCH --job-name=vo-toklosses-best
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
# Owner popups 2026-10-23: per-position Paloma losses at the last step of the lowest setups (8K LR x2, the 128K best
# setup at seeds 0 and 1) and of random input tokens at 128K and 8K (the setup that helps 128K and hurts 8K), plus the
# plain 128K seed-0 rep8 run, for costs by token frequency (vocab_cost_groups.py). About 4 min per checkpoint on one GPU.
export SPECS="muonh-qwen3-300m-della4xh100-lrx2-fp2-v8k-rep8 15463 VOCAB_K=8000;muonh-qwen3-300m-della4xh100-cvd8k-p0.5-mdetach-rep8-rs 11443;muonh-qwen3-300m-della4xh100-cvd8k-p0.5-mdetach-rep8-rs-s1 11443;muonh-qwen3-300m-della4xh100-inrand0.1-fp2-rep8 11443;muonh-qwen3-300m-della4xh100-inrand0.1-fp2-v8k-rep8 15463 VOCAB_K=8000;muonh-qwen3-300m-della4xh100-rep8 11443"
exec bash /scratch/gpfs/GROUP/USER/project/marin/scripts/della/vocab_token_losses.sbatch
