#!/bin/bash
#SBATCH --job-name=attn-bench-thd
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=64G
#SBATCH --time=00:30:00
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell; run through wq --prio 10
# One H100: scripts/della/speed_all/attn_bench.py, segmented FA4 (fa4) against upstream FA4 varlen (fa4thd, branch
# fa4-sm90) at the 300m per-device shape (32 x T x 12 heads x 64, T 4096 and 5504), document-masked
# causal attention. Results go to $OUT/attn_bench_<T>_<block>.jsonl.
set -uo pipefail
MAIN=/scratch/gpfs/GROUP/USER/project/marin
WT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/fa4sm90   # branch fa4-sm90
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
  T=$1 FLASH=$2 BACKENDS="$3" timeout 15m $PY scripts/della/speed_all/attn_bench.py > $OUT/attn_bench_$1_$2.jsonl 2> $OUT/attn_bench_$1_$2.err || fail=1
  cat $OUT/attn_bench_$1_$2.jsonl
}
run 4096 1024 "fa4 fa4thd"
run 5504 688 "fa4 fa4thd"
echo "BENCH DONE $(date +%H:%M:%S) fail=$fail out=$OUT"
exit $fail
