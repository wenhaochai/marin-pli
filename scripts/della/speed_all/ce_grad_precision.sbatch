#!/bin/bash
#SBATCH --job-name=ce-grad-precision
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=64G
#SBATCH --time=00:40:00
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell; run through wq --prio 10
# One H100: scripts/della/speed_all/ce_grad_precision.py (branch async-hooks), CE gradient precision and speed by path
# (default, GIF with the lse kernel, GIF bf16, GIF f32 logits) at V 8256, 32256 and 128256 against an f32 reference.
# JSON lines go to $OUT/bench.jsonl and to this log.
set -uo pipefail
MAIN=/scratch/gpfs/GROUP/USER/project/marin
WT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/async
PY=$MAIN/.venv/bin/python
cd "$WT"
source scripts/della/job_tmpdir.sh
export PYTHONPATH=$WT/lib/levanter/src:$WT/lib/haliax/src:$WT XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
OUT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/profiles/ce-precision-$JID
mkdir -p $OUT
nvidia-smi -L
echo "BENCH START $(date +%H:%M:%S) $(git -C $WT log --oneline -1)"
timeout 35m $PY scripts/della/speed_all/ce_grad_precision.py 2> $OUT/bench.err | tee $OUT/bench.jsonl
rc=${PIPESTATUS[0]}
tail -5 $OUT/bench.err
echo "BENCH END rc=$rc $(date +%H:%M:%S) out=$OUT"
exit $rc
