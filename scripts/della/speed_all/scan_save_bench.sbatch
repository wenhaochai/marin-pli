#!/bin/bash
#SBATCH --job-name=scan-save-bench
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=96G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell; run through wq --prio 10
# Four one-GPU benches side by side (speed work 2026-10-26 night), the production XLA flags of speed_env.sh in each:
#   GPU 0-2  scripts/della/speed_all/scan_save_bench.py: saved MLP activations through the layer scan, autodiff (A)
#            against explicit buffers (B); GPU 0 the production flags, GPU 1 plus dynamic-slice fusion, GPU 2 plus
#            dynamic-slice fusion with the legacy cuBLAS call
#   GPU 3    scripts/della/speed_all/fp8_head_bench.py: the output head's GEMMs in bf16 and fp8 at 128K, 64K and 32K
# The optimized HLO of each scan-save version goes to $OUT/hlo_<arm>_<A|B>/step.txt.
set -uo pipefail
MAIN=/scratch/gpfs/GROUP/USER/project/marin
WT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/async
PY=$MAIN/.venv/bin/python
cd "$WT"
source scripts/della/job_tmpdir.sh
source $MAIN/scripts/della/speed_env.sh > /dev/null
export PYTHONPATH=$WT XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
OUT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/profiles/scansave-$JID
mkdir -p $OUT
nvidia-smi -L
echo "BENCH START $(date +%H:%M:%S) $(git -C $WT log --oneline -1) XLA_FLAGS=$XLA_FLAGS"
B=scripts/della/speed_all
# this job's own GPUs as the host set them (a guest on GPUs 4-7 must not touch 0-3)
IFS=, read -r -a GPUS <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
DSF="--xla_gpu_enable_dynamic_slice_fusion=true"
CUDA_VISIBLE_DEVICES=${GPUS[0]} HLO_DIR=$OUT timeout 20m $PY $B/scan_save_bench.py prod > $OUT/prod.log 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=${GPUS[1]} HLO_DIR=$OUT XLA_FLAGS="$XLA_FLAGS $DSF" timeout 20m $PY $B/scan_save_bench.py dsf > $OUT/dsf.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=${GPUS[2]} HLO_DIR=$OUT XLA_FLAGS="$XLA_FLAGS $DSF --xla_gpu_enable_cublaslt=false" timeout 20m $PY $B/scan_save_bench.py dsflegacy > $OUT/dsflegacy.log 2>&1 &
p2=$!
CUDA_VISIBLE_DEVICES=${GPUS[3]} timeout 20m $PY $B/fp8_head_bench.py > $OUT/fp8.log 2>&1 &
p3=$!
fail=0
for p in $p0 $p1 $p2 $p3; do wait $p || fail=$((fail + 1)); done
for f in prod dsf dsflegacy fp8; do
  echo "== $f"
  grep -E "^(BENCH|GRAD|FP8HEAD)" $OUT/$f.log || tail -5 $OUT/$f.log
done
echo "BENCH DONE fail=$fail $(date +%H:%M:%S) out=$OUT"
# a bench exits 0 when it ran: an arm that fails (a tile config past shared memory, say) is a result, shown in the
# DONE line; a nonzero exit counts as a quick failure on the host, and three of them stop its fetching (w472)
exit 0
