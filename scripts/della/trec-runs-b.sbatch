#!/bin/bash
#SBATCH --job-name=trec-runs-b
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable-partial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/trec/logs/slurm/%x_%j.out
#FETCHABLE self-contained: every setting is in this file and trec_runs_common.sh; needs nothing from the submitting shell
# TREC of the other MuonH checkpoints (trec_runs_common.sh), part b: 1 in 8 training batches (blocks of 8 steps every 64)
# and 512 unread batches of each checkpoint whose smoke passed (out/smoke3/check.json).
# Each output chunk is skipped when present, so a rerun continues (resumable: wq --partial; a stop by signal exits 143 and requeues).
set -uo pipefail
source /scratch/gpfs/GROUP/USER/project/marin-trec/scripts/della/trec_runs_common.sh
export SAMPLE=8 BLOCK=8
/scratch/gpfs/GROUP/USER/project/marin/.venv/bin/python experiments/trec/smoke_check.py --passed /scratch/gpfs/GROUP/USER/project/trec/out/smoke3 > $TMPDIR/runs.txt <<'RUNS' || fail=1
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
score_runs all /scratch/gpfs/GROUP/USER/project/trec/out/runs < $TMPDIR/runs.txt
echo "DONE fail=$fail $(date +%H:%M:%S)"
exit $fail
