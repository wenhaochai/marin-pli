#!/bin/bash
#SBATCH --job-name=fa4thd-debug
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=64G
#SBATCH --time=00:25:00
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell; run through wq --prio 10
# One H100: scripts/della/speed_all/fa4thd_debug.py, one step per process (fwd128 bwd128 fwd64 bwd64 local64), with
# PYTHONFAULTHANDLER so a C++ abort prints the Python stack of the call that raised it.
set -uo pipefail
MAIN=/scratch/gpfs/GROUP/USER/project/marin
WT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/all
PY=$MAIN/.venv/bin/python
cd "$WT"
source scripts/della/job_tmpdir.sh
export PYTHONPATH=$WT/lib/levanter/src:$WT/lib/haliax/src:$WT XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
OUT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/profiles/fa4thd-debug-$JID
mkdir -p $OUT
nvidia-smi -L
fail=0
for s in fwd128 bwd128 fwd64 bwd64 local64; do
  echo "DEBUG $s START $(date +%H:%M:%S)"
  PYTHONFAULTHANDLER=1 timeout 8m $PY scripts/della/speed_all/fa4thd_debug.py $s > $OUT/$s.log 2>&1; echo "DEBUG $s rc=$?"
  grep -E "^STEP|^config|^forward" $OUT/$s.log; grep -B2 -A14 -E "Fatal Python|Traceback|Error" $OUT/$s.log | grep -vE "^\s*$" | tail -30
done
echo "BENCH END rc=0 $(date +%H:%M:%S) out=$OUT"
exit 0
