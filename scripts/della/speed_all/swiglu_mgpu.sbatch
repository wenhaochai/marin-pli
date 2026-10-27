#!/bin/bash
#SBATCH --job-name=swiglu-mgpu-bench
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell; run through wq --prio 10
# The SwiGLU forward as one Mosaic GPU GEMM (scripts/della/speed_all/swiglu_mgpu.py, prototype of 2026-10-26 night) against
# two cuBLAS GEMMs plus XLA's product at the 300m shape, four tile configs side by side, one per GPU.
set -uo pipefail
MAIN=/scratch/gpfs/GROUP/USER/project/marin
WT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/async
PY=$MAIN/.venv/bin/python
cd "$WT/scripts/della/speed_all"
source $WT/scripts/della/job_tmpdir.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
OUT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/profiles/swiglu-$JID
mkdir -p $OUT
nvidia-smi -L
echo "BENCH START $(date +%H:%M:%S) $(git -C $WT log --oneline -1)"
# this job's own GPUs as the host set them (a guest on GPUs 4-7 must not touch 0-3)
IFS=, read -r -a GPUS <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CFGS=(m128wgm4 m128wgm3 m128wgn4 m128wgn3)
pids=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} timeout 15m $PY swiglu_mgpu.py ${CFGS[$i]} > $OUT/${CFGS[$i]}.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
for c in "${CFGS[@]}"; do
  echo "== $c"
  grep -E "^SWIGLU" $OUT/$c.log || tail -8 $OUT/$c.log
done
echo "BENCH DONE fail=$fail $(date +%H:%M:%S) out=$OUT"
# a bench exits 0 when it ran: an arm that fails (a tile config past shared memory, say) is a result, shown in the
# DONE line; a nonzero exit counts as a quick failure on the host, and three of them stop its fetching (w472)
exit 0
