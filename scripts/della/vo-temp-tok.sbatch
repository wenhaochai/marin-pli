#!/bin/bash
#SBATCH --job-name=vo-temp-tok
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=02:30:00
#SBATCH --comment=fetchable
#SBATCH --output=logs/slurm/%x_%j.out
#FETCHABLE self-contained: SPECS is set here; needs nothing from the submitting shell
# Owner popup 2026-10-23 (温度检验): Paloma loss at softmax temperatures 0.95-1.2 (logits / T) of the final checkpoints,
# to test whether 8 passes make the model overconfident everywhere and more so at 128K (why noise helps 128K, hurts 8K).
export SPECS="muonh-qwen3-300m-della4xh100-s1-rerun 11443;muonh-qwen3-300m-della4xh100-rep8-s1 11443;muonh-qwen3-300m-della4xh100-rep8 11443;muonh-qwen3-300m-della4xh100-inrand0.1-fp2-rep8 11443;muonh-qwen3-300m-della4xh100-v8k 15463 VOCAB_K=8000;muonh-qwen3-300m-della4xh100-v8k-rep8 15463 VOCAB_K=8000;muonh-qwen3-300m-della4xh100-v8k-s1 15463 VOCAB_K=8000;muonh-qwen3-300m-della4xh100-v8k-rep8-s1 15463 VOCAB_K=8000;muonh-qwen3-300m-della4xh100-inrand0.1-fp2-v8k-rep8 15463 VOCAB_K=8000;muonh-qwen3-300m-della4xh100-lrx2-fp2-v8k-rep8 15463 VOCAB_K=8000;muonh-qwen3-300m-della4xh100-lrx2-fp2-v8k 15463 VOCAB_K=8000"
exec bash /scratch/gpfs/GROUP/USER/project/marin/scripts/della/vocab_temp_losses.sbatch
