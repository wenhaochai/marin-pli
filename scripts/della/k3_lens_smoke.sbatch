#!/bin/bash
#SBATCH --job-name=k3-lens-smoke
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
# Smoke test of the Kimi K3 logit lens (pli-cp): k3_lens.sbatch on one GPU through layers 0-3 (KDA with the dense MLP,
# KDA with MoE, MLA with MoE), on 1/1 of the token file's sequences... capped by the GPU: 64 sequences of 1,024 tokens.
export K3_GPUS=1 K3_MAX_LAYERS=4
exec bash /scratch/gpfs/GROUP/USER/project/marin-pls-auto10/scripts/della/k3_lens.sbatch
