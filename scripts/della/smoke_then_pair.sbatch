#!/bin/bash
#SBATCH --job-name=smoke-then-pair
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# Smoke and training in one allocation (A/B on queue time, owner via the terminal controller, 2026-10-07): run
# over_vocab_smoke.sbatch on GPUs 0-3 with SIZES / VARIANTS / SMOKE_ENVS from the environment; if it fails, exit
# without training; if it passes, run muonh_qwen3_h100x8_pair.sbatch (PAIR_A on GPUs 0-3, PAIR_B on 4-7) in the same
# allocation. Both scripts run with bash, so their #SBATCH lines are ignored and this job's resources apply.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
echo "SMOKE PHASE START $(date +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/della/over_vocab_smoke.sbatch
rc=$?
echo "SMOKE PHASE END rc=$rc $(date +%H:%M:%S)"
[ "$rc" -eq 0 ] || { echo "smoke failed: no training"; exit $rc; }
echo "TRAIN PHASE START $(date +%H:%M:%S)"
bash scripts/della/muonh_qwen3_h100x8_pair.sbatch
rc=$?
echo "TRAIN PHASE END rc=$rc $(date +%H:%M:%S)"
exit $rc
