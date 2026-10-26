#!/bin/bash
#SBATCH --job-name=speedall-prof2
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=48G
#SBATCH --time=00:50:00
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell; run through wq --prio 10
# H100 profile of branch speed-all (owner 2026-10-25: dig into infra and kernels): one 300m config, SMOKE_STEPS=30 steps,
# a JAX profiler trace of steps 20-22 (PROFILE=20:3), three arms in turn on the same 4 GPUs:
#   base = today's path (JAX flash, full remat, no command buffers; at 4096 the block rule is unchanged)
#   jax  = REMAT=flash
#   fa4  = FA4/CuTe attention        none = REMAT=none (arms after the config on the command line; default base jax fa4)
#   now  = the defaults since 2026-10-26 04:30 (FA4, REMAT=flash, CE A1 + GIF); nowdots = now with REMAT=fdots
# Every arm runs with no command buffers and dumps its optimized HLO, so trace_families.py can name the part of the step
# each kernel belongs to; step times come from the A/B jobs, not from here.
# The traces and HLO go to $OUT/<arm>/; FAM lines: the trace_families.py summary per arm.
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
export NUM_GPUS=2 PDP=32   # the 2-GPU copy: 32 sequences per device per microbatch as on 4 GPUs (2 microbatches); per-GPU throughput matches 4 GPUs within 2% (w202)
case $CFG in
  4096) export SIZE=300m SEQ_LEN=4096; unset VOCAB_K ;;
  4480) export SIZE=300m SEQ_LEN=4480 VOCAB_K=32000 ;;
  4096v8k) export SIZE=300m SEQ_LEN=4096 VOCAB_K=8000 ;;
  5504) export SIZE=300m SEQ_LEN=5504 VOCAB_K=8000 ;;
  *) echo "unknown config $CFG"; exit 2 ;;
esac
export SMOKE_STEPS=30 PROFILE=20:3
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
OUT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/profiles/${CFG}-$JID
mkdir -p $OUT
nvidia-smi -L
fail=0
ARMS=("${@:2}"); [ ${#ARMS[@]} -ge 1 ] || ARMS=(base jax fa4)
for ARM in "${ARMS[@]}"; do
  unset XLA_FLAGS ATTN_BACKEND REMAT LEVANTER_CE_BTILED_ONEHOT LEVANTER_CE_GRAD_IN_FWD
  export REMAT=full   # an arm that names no REMAT keeps full remat (the code default is flash since 2026-10-26)
  # attribution: no command buffers in any arm (each kernel then names its own HLO instruction; kernel times are the
  # same), and the step's optimized HLO dumped for scripts/della/speed_all/trace_families.py
  mkdir -p $OUT/$ARM/hlo
  export XLA_FLAGS="--xla_gpu_nccl_termination_timeout_seconds=600 --xla_gpu_enable_command_buffer= --xla_dump_to=$OUT/$ARM/hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train_step.*"
  case $ARM in   # explicit backends: the code defaults to fa4 since 2026-10-26
    base) export ATTN_BACKEND=jax_flash ;;
    jax) export ATTN_BACKEND=jax_flash REMAT=flash ;;
    none) export ATTN_BACKEND=jax_flash REMAT=none ;;
    fa4) export ATTN_BACKEND=fa4 ;;
    fa4dots) export ATTN_BACKEND=fa4 REMAT=dots ;;
    now) export ATTN_BACKEND=fa4 REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 ;;   # the defaults since 2026-10-26 04:30
    nowdots) export ATTN_BACKEND=fa4 REMAT=fdots LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 ;;
    *) echo "unknown arm $ARM"; fail=1; continue ;;
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
  [ $rc -eq 0 ] && $PY scripts/della/speed_all/trace_families.py $OUT/$ARM/profiler $OUT/$ARM/hlo 3 2>/dev/null > $OUT/$ARM/families.txt && sed "s/^/FAM $ARM /" $OUT/$ARM/families.txt | head -18
  case "$run_id" in *-abprof*-j$JID*) rm -rf "$MARIN_PREFIX/speedrun/$run_id" ;; esac
done
echo "PROF DONE $(date +%H:%M:%S) fail=$fail out=$OUT"
exit $fail
