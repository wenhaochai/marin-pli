#!/bin/bash
#SBATCH --job-name=bytes-300m-rep2.4
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=24G
#SBATCH --time=24:00:00
#SBATCH --exclude=della-k14g3,della-k17g3

#SBATCH --output=logs/slurm/%x_%j.out

export PLAN=/scratch/gpfs/GROUP/USER/project/marin/logs/plans/bytes-300m-rep2.4.txt PACKED_NAME=bytes-300m-rep2.4
bash /scratch/gpfs/GROUP/USER/project/marin/scripts/della/packed_job.sbatch; rc=$?
bash /scratch/gpfs/GROUP/USER/tools/fetch_pending.sh   # the time left runs pending fetchable jobs
exit $rc
