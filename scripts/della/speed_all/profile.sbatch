#!/bin/bash
#SBATCH --job-name=speedall-prof
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=24G
#SBATCH --time=00:50:00
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell; run through wq --prio 10
# H100 profile of branch speed-all (owner 2026-10-25: dig into infra and kernels): one 300m config, SMOKE_STEPS=30 steps,
# a JAX profiler trace of steps 20-22 (PROFILE=20:3), three arms in turn on the same 4 GPUs:
#   base = today's path (JAX flash, full remat, no command buffers; at 4096 the block rule is unchanged)
#   jax  = REMAT=flash + CUDA-graph while loops
#   fa4  = FA4/CuTe attention + CUDA-graph while loops
# The traces are copied to $OUT/<arm>/ for scripts/della/speed_all/trace_breakdown.py (run on a vis node).
set -uo pipefail
CFG=${1:?usage: profile.sbatch <4096|4480>}
MAIN=/scratch/gpfs/GROUP/USER/project/marin
WT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/all
PY=$MAIN/.venv/bin/python
cd "$WT"
source scripts/della/job_tmpdir.sh
module load proxy/default  # wandb only
source scripts/della/wandb_guard.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONPATH=$WT/lib/levanter/src:$WT/lib/haliax/src:$WT
unset VARIANT DATA_EPOCHS PASS_DATA TOTAL_STEPS INIT_FROM SEED BATCH PDP NUM_GPUS TIE PRECISION XLA_FLAGS ATTN_BACKEND REMAT
case $CFG in
  4096) export SIZE=300m SEQ_LEN=4096; unset VOCAB_K ;;
  4480) export SIZE=300m SEQ_LEN=4480 VOCAB_K=32000 ;;
  *) echo "unknown config $CFG"; exit 2 ;;
esac
export SMOKE_STEPS=30 PROFILE=20:3
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
OUT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/profiles/${CFG}-$JID
mkdir -p $OUT
nvidia-smi -L
fail=0
for ARM in base jax fa4; do
  unset XLA_FLAGS ATTN_BACKEND REMAT
  CB="--xla_gpu_nccl_termination_timeout_seconds=600 --xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL"
  case $ARM in
    jax) export REMAT=flash XLA_FLAGS="$CB" ;;
    fa4) export ATTN_BACKEND=fa4 XLA_FLAGS="$CB" ;;
  esac
  export RUN_TAG=-abprof${ARM}-j$JID
  run_id=$(DRY_RUN=1 $PY -m experiments.references.della_muonh_qwen3_scaling 2>/dev/null | sed -n 's#^output: speedrun/\([^/]*\)/.*#\1#p' | head -1)
  case "$run_id" in *-abprof${ARM}-j$JID*) ;; *) echo "ARM $ARM: dry run gave '$run_id'"; fail=1; continue ;; esac
  log=$OUT/$ARM.log
  echo "PROF $run_id START $(date +%H:%M:%S) XLA_FLAGS=${XLA_FLAGS:-} REMAT=${REMAT:-} ATTN_BACKEND=${ATTN_BACKEND:-}"
  timeout --signal=TERM 12m $PY -m experiments.references.della_muonh_qwen3_scaling > $log 2>&1
  rc=$?; [ $rc -ne 0 ] && fail=1
  mkdir -p $OUT/$ARM
  if [ -d "logs/$run_id/profiler" ]; then cp -r "logs/$run_id/profiler" $OUT/$ARM/ && rm -rf "logs/$run_id"; fi
  echo "PROF $run_id END rc=$rc files=$(find $OUT/$ARM -type f | wc -l) $(date +%H:%M:%S)"
  case "$run_id" in *-abprof*-j$JID*) rm -rf "$MARIN_PREFIX/speedrun/$run_id" ;; esac
done
echo "PROF DONE $(date +%H:%M:%S) fail=$fail out=$OUT"
exit $fail
