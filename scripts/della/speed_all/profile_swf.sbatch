#!/bin/bash
#SBATCH --job-name=speedall-profswf
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=48G
#SBATCH --time=00:50:00
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell; run through wq --prio 10
# H100 profile of the fused SwiGLU path against the production state (2026-10-27): this branch's code with
# scripts/della/speed_env.sh, 300m, 4 GPUs, SMOKE_STEPS=30 and a JAX profiler trace of steps 20-22 (PROFILE=20:3), per
# config two arms, both with no command buffers and the optimized HLO dumped so trace_families.py names each kernel's part:
#   attr    = the production flags
#   attrswf = the same with LEVANTER_SWIGLU_FUSED=1 (both SwiGLU kernels)
# Configs (arguments, default 4096 4096v8k): 4096 = 128K, 4096v8k = 8K. Traces and HLO go to $OUT/<config>/<arm>/.
set -uo pipefail
MAIN=/scratch/gpfs/GROUP/USER/project/marin
WT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/async   # the trace scripts
PY=$MAIN/.venv/bin/python
cd "$WT"   # this branch: LEVANTER_SWIGLU_FUSED lives here
source scripts/della/job_tmpdir.sh
module load proxy/default  # wandb only
source scripts/della/wandb_guard.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONPATH=$WT/lib/levanter/src:$WT/lib/haliax/src:$WT
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
OUT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/profiles/swf-$JID
mkdir -p $OUT
nvidia-smi -L
fail=0
CFGS=("$@"); [ ${#CFGS[@]} -ge 1 ] || CFGS=(4096 4096v8k)
for CFG in "${CFGS[@]}"; do
  for ARM in attr attrswf; do
    # a clean environment per arm: the knobs speed_env.sh sets, and the run's own
    unset VARIANT DATA_EPOCHS PASS_DATA TOTAL_STEPS INIT_FROM SEED BATCH PDP NUM_GPUS TIE PRECISION XLA_FLAGS ATTN_BACKEND REMAT US_EPS US_FROM_PASS VOCAB_K SEQ_LEN
    unset XLA_CMD_BUFFER_WHILE XLA_ZERO_INIT_ELIM LEVANTER_CE_BTILED_ONEHOT LEVANTER_CE_GRAD_IN_FWD LEVANTER_CE_LSE_KERNEL LEVANTER_ASYNC_STEP LEVANTER_EMBED_GRAD_F32 LEVANTER_NORM_GUARD
    unset LEVANTER_CE_UNNORM LEVANTER_CE_BTILED_MIN_VOCAB LEVANTER_CE_LSE_MGPU LEVANTER_CE_LSE_KERNEL_EVAL LEVANTER_SWIGLU_VJP LEVANTER_SWIGLU_FUSED
    export NUM_GPUS=4 PDP=32 SIZE=300m SEQ_LEN=4096
    case $CFG in
      4096) ;;
      4096v8k) export VOCAB_K=8000 ;;
      *) echo "unknown config $CFG"; fail=1; continue ;;
    esac
    D=$OUT/$CFG/$ARM; mkdir -p $D/hlo
    [ $ARM = attrswf ] && export LEVANTER_SWIGLU_FUSED=1
    if [ $ARM = attr ] || [ $ARM = attrswf ]; then
      export XLA_CMD_BUFFER_WHILE=0
      export XLA_FLAGS="--xla_gpu_nccl_termination_timeout_seconds=600 --xla_gpu_enable_command_buffer= --xla_dump_to=$D/hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train_step.*"
    fi
    source scripts/della/speed_env.sh > $D/speed_env.txt
    export SMOKE_STEPS=30 PROFILE=20:3 RUN_TAG=-abprofswf${ARM}-j$JID
    run_id=$(DRY_RUN=1 $PY -m experiments.references.della_muonh_qwen3_scaling 2>/dev/null | sed -n 's#^output: speedrun/\([^/]*\)/.*#\1#p' | head -1)
    case "$run_id" in *-abprofswf${ARM}-j$JID*) ;; *) echo "$CFG $ARM: dry run gave '$run_id'"; fail=1; continue ;; esac
    echo "PROF $CFG $ARM $run_id START $(date +%H:%M:%S) XLA_FLAGS=${XLA_FLAGS:-}"
    timeout --signal=TERM 12m $PY -m experiments.references.della_muonh_qwen3_scaling > $D/run.log 2>&1
    rc=$?; [ $rc -ne 0 ] && fail=1
    if [ -d "logs/$run_id/profiler" ]; then cp -r "logs/$run_id/profiler" $D/ && rm -rf "logs/$run_id"; fi
    echo "PROF $CFG $ARM END rc=$rc files=$(find $D -type f | wc -l) $(date +%H:%M:%S)"
    if [ $rc -eq 0 ]; then
      [ $ARM != prod ] && $PY $WT/scripts/della/speed_all/trace_families.py $D/profiler $D/hlo 3 2>/dev/null > $D/families.txt && sed "s/^/FAM $CFG $ARM /" $D/families.txt | head -18
      $PY $WT/scripts/della/speed_all/trace_gaps.py $D/profiler 300 2>/dev/null > $D/gaps.txt && sed "s/^/GAPS $CFG $ARM /" $D/gaps.txt | head -16
    fi
    case "$run_id" in *-abprofswf*-j$JID*) rm -rf "$MARIN_PREFIX/speedrun/$run_id" ;; esac
  done
done
echo "PROF DONE $(date +%H:%M:%S) fail=$fail out=$OUT"
exit $fail
