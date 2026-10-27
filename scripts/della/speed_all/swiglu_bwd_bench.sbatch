#!/bin/bash
#SBATCH --job-name=swiglu-bwd-bench
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
# The fused SwiGLU backward kernel and whole-MLP timings (scripts/della/speed_all/swiglu_bwd_bench.py, 2026-10-27):
# round 2 (2026-10-27 ~00:00): GPU 0 the gradients' error against f32 (production path and both kernels), GPU 1-3 the
# forward kernel at wgmma's native 64 x 256 per warpgroup (4 and 3 stages) and at the current default m128wgn4.
set -uo pipefail
MAIN=/scratch/gpfs/GROUP/USER/project/marin
WT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/async
PY=$MAIN/.venv/bin/python
cd "$WT/scripts/della/speed_all"
source $WT/scripts/della/job_tmpdir.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
OUT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/profiles/swiglubwd-$JID
mkdir -p $OUT
nvidia-smi -L
echo "BENCH START $(date +%H:%M:%S) $(git -C $WT log --oneline -1)"
# this job's own GPUs as the host set them (a guest on GPUs 4-7 must not touch 0-3)
IFS=, read -r -a GPUS <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
ARGS=("precision m128wgn4 m128wgm3" "fwd m64n256wgm4" "fwd m64n256wgm3" "fwd m128wgn4")
NAMES=(a0 a1 a2 a3)
export PYTHONPATH=$WT/lib/levanter/src:$WT/lib/haliax/src
pids=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} timeout 15m $PY swiglu_bwd_bench.py ${ARGS[$i]} > $OUT/${NAMES[$i]}.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
for c in "${NAMES[@]}"; do
  echo "== $c"
  grep -E "^SWIGLU" $OUT/$c.log || tail -8 $OUT/$c.log
done
echo "BENCH DONE fail=$fail $(date +%H:%M:%S) out=$OUT"
exit $fail
