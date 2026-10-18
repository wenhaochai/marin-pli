#!/bin/bash
#SBATCH --job-name=why-cvd520-mdetach-rs
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-gpu=24G
#SBATCH --time=24:00:00
#SBATCH --exclude=della-k14g3,della-k17g3
#SBATCH --comment=fetchable
#SBATCH --output=logs/slurm/%x_%j.out
#FETCHABLE self-contained: written by scripts/della/submit_packed.sh; needs nothing from the submitting shell
export PLAN=/scratch/gpfs/GROUP/USER/project/marin/logs/plans/why-cvd520-mdetach-rs.txt PACKED_NAME=why-cvd520-mdetach-rs
bash /scratch/gpfs/GROUP/USER/project/marin/scripts/della/packed_job.sbatch; rc=$?
bash /scratch/gpfs/GROUP/USER/tools/fetch_pending.sh   # the time left runs pending fetchable jobs
exit $rc
