#!/bin/bash
#SBATCH --job-name=attn-bench-dense
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=64G
#SBATCH --time=00:20:00
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell; run through wq --prio 10
# One H100: scripts/della/speed_all/attn_bench.py, segmented FA4 (fa4) against upstream FA4 varlen (fa4thd) on the
# layouts dense (row 0 holds 24 documents; errors against fp32 on rows 0-1) and overflow (12 documents in every row, past
# the fa4thd bound of 10 per row: its output must hold a NaN), 32 x 4096 x 12 heads x 64, document-masked
# causal attention. Results go to $OUT/attn_bench_<T>_<block>.jsonl.
set -uo pipefail
MAIN=/scratch/gpfs/GROUP/USER/project/marin
WT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/all
PY=$MAIN/.venv/bin/python
cd "$WT"
source scripts/della/job_tmpdir.sh
export PYTHONPATH=$WT/lib/levanter/src:$WT/lib/haliax/src:$WT XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
OUT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/profiles/attn-$JID
mkdir -p $OUT
nvidia-smi -L
fail=0
run() {  # T FLASH BACKENDS
  echo "BENCH T=$1 FLASH=$2 backends=$3 START $(date +%H:%M:%S)"
  T=$1 FLASH=$2 BACKENDS="$3" LAYOUTS="$4" timeout 15m $PY scripts/della/speed_all/attn_bench.py > $OUT/attn_bench_$1_$2.jsonl 2> $OUT/attn_bench_$1_$2.err || fail=1
  cat $OUT/attn_bench_$1_$2.jsonl
}
run 4096 1024 "fa4 fa4thd" "dense overflow"
echo "BENCH DONE $(date +%H:%M:%S) fail=$fail out=$OUT"
exit $fail
