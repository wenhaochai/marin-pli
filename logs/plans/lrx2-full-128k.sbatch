#!/bin/bash
#SBATCH --job-name=lrx2-full-128k
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=24G
#SBATCH --time=07:30:00
#SBATCH --exclude=della-k14g3,della-k17g3
#SBATCH --comment=fetchable-partial
#SBATCH --output=logs/slurm/%x_%j.out
#FETCHABLE self-contained: written by scripts/della/submit_packed.sh; needs nothing from the submitting shell
export PLAN=/scratch/gpfs/GROUP/USER/project/marin/logs/plans/lrx2-full-128k.txt PACKED_NAME=lrx2-full-128k
exec bash /scratch/gpfs/GROUP/USER/project/marin/scripts/della/packed_job.sbatch
