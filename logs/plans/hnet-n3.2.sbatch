#!/bin/bash
#SBATCH --job-name=hnet-n3.2
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-gpu=24G
#SBATCH --time=10:45:00
#SBATCH --exclude=della-k14g3,della-k17g3
#SBATCH --comment=fetchable-partial
#SBATCH --output=logs/slurm/%x_%j.out
#FETCHABLE self-contained: written by tmp_plots/vocab/hnet_diag/write_plans.py; needs nothing from the submitting shell
export PLAN=/scratch/gpfs/GROUP/USER/project/marin/logs/plans/hnet-n3.2.txt PACKED_NAME=hnet-n3.2
exec bash /scratch/gpfs/GROUP/USER/project/marin/scripts/della/packed_job.sbatch
