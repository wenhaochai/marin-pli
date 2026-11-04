#!/bin/bash
#SBATCH --job-name=vs-vs128k16ku
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=24G
#SBATCH --time=08:30:00
#SBATCH --exclude=della-k14g3,della-k17g3
#SBATCH --comment=fetchable-partial
#SBATCH --output=logs/slurm/%x_%j.out
#FETCHABLE self-contained: written by tmp_plots/vocab/hnet_diag/write_plans_shrink4.py; needs nothing from the submitting shell
export PLAN=/scratch/gpfs/GROUP/USER/project/marin/logs/plans/vs-vs128k16ku.txt PACKED_NAME=vs-vs128k16ku
exec bash /scratch/gpfs/GROUP/USER/project/marin/scripts/della/packed_job.sbatch
