#!/bin/bash
#SBATCH --job-name=trec-runs-smoke
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/trec/logs/slurm/%x_%j.out
#FETCHABLE self-contained: every setting is in this file and trec_runs_common.sh; needs nothing from the submitting shell
# TREC of the other MuonH checkpoints (trec_runs_common.sh), smoke: the first 4 and last 12 training batches and 4 unread
# batches of each; experiments/trec/smoke_check.py then compares the last ones with the W&B train loss.
set -uo pipefail
source /scratch/gpfs/GROUP/USER/project/marin-trec/scripts/della/trec_runs_common.sh
O=/scratch/gpfs/GROUP/USER/project/trec/out/smoke3
score_runs smoke $O <<'RUNS'
muonh-qwen3-1_2b-della8xh100-pls0-noev 22887 model
muonh-qwen3-520m-della8xh100-pls0-noev 9917 model
muonh-qwen3-300m-della8xh100-pls0-noev 11443 model
muonh-qwen3-130m-della4xh100-d1 4958 model
muonh-qwen3-130m-della4xh100-d2 4958 model
muonh-qwen3-130m-della4xh100-d3 4958 model
muonh-qwen3-130m-della4xh100-d4 4958 model
muonh-qwen3-130m-della4xh100-d5 4958 model
muonh-qwen3-130m-della4xh100-d24 4958 model
muonh-qwen3-130m-della4xh100-d36 4958 model
muonh-qwen3-300m-della4xh100-v8k 15463 model
muonh-qwen3-300m-della4xh100-lrx2-fp2-v8k 15463 model
muonh-qwen3-300m-della4xh100-lrx3-fp2-v8k 15463 model
muonh-qwen3-300m-della4xh100-lrx4-fp2-v8k 15463 model
muonh-qwen3-300m-della4xh100-ema0.999 11443 model
muonh-qwen3-300m-della4xh100-ema0.999 11443 model_averaging/model
muonh-qwen3-300m-della4xh100-ema0.9998 11443 model
muonh-qwen3-300m-della4xh100-ema0.9998 11443 model_averaging/model
muonh-qwen3-130m-della4xh100-tpp200 49589 model
muonh-qwen3-130m-della4xh100-tpp200-s1 49589 model
muonh-qwen3-300m-della4xh100-b96 15258 model
muonh-qwen3-300m-della4xh100-v16k 13781 model
muonh-qwen3-300m-della4xh100-v32k 12616 model
muonh-qwen3-300m-della4xh100-v64k 11801 model
muonh-qwen3-300m-della4xh100-bytes 54247 model
muonh-qwen3-300m-della4xh100-v8k-s1 15463 model
RUNS
/scratch/gpfs/GROUP/USER/project/marin/.venv/bin/python experiments/trec/smoke_check.py $O || fail=1
echo "DONE fail=$fail $(date +%H:%M:%S)"
exit 0   # the scoring jobs read smoke_check's verdict per run ($O/check.json), so one bad run does not hold back the rest
