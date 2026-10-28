#!/bin/bash
#SBATCH --job-name=trec-runs-a
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
# TREC of the other MuonH checkpoints (trec_runs_common.sh), part a: 1 in 8 training batches (blocks of 8 steps every 64)
# and 512 unread batches of each checkpoint whose smoke passed (out/smoke3/check.json).
# Each output chunk is skipped when present, so a rerun continues (resumable: wq --partial; a stop by signal exits 143 and requeues).
set -uo pipefail
source /scratch/gpfs/GROUP/USER/project/marin-trec/scripts/della/trec_runs_common.sh
export SAMPLE=8 BLOCK=8
/scratch/gpfs/GROUP/USER/project/marin/.venv/bin/python experiments/trec/smoke_check.py --passed /scratch/gpfs/GROUP/USER/project/trec/out/smoke3 > $TMPDIR/runs.txt <<'RUNS' || fail=1
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
RUNS
score_runs all /scratch/gpfs/GROUP/USER/project/trec/out/runs < $TMPDIR/runs.txt
echo "DONE fail=$fail $(date +%H:%M:%S)"
exit $fail
