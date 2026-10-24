#!/bin/bash
#SBATCH --job-name=fa4-ab-sl4224
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-gpu=24G
#SBATCH --time=02:20:00
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell (FA4 attention A/B, two 4-GPU arms side by side)
export PLAN=/scratch/gpfs/GROUP/USER/tmp/speed_wt/fa4-segmented-attention-backend/scripts/della/fa4_ab/fa4-ab-sl4224.txt PACKED_NAME=fa4-ab-sl4224
exec bash /scratch/gpfs/GROUP/USER/project/marin/scripts/della/packed_job.sbatch
