#!/bin/bash
#SBATCH --job-name=speedall-arms
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=24G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell; run through wq --prio 10 (owner 2026-10-25: tests go to wq, not pli-cp)

# H100 check of branch speed-all (owner 2026-10-25 "全部都上"): one 300m config, SMOKE_STEPS steps at the real batch, the
# arms named on the command line in turn on the same 4 GPUs:
#   old     = the main checkout (vocab-overfitting, every run so far): JAX flash, full remat, power-of-two flash block
#   base    = speed-all with no switch on (new flash block rule only)
#   cb      = CUDA-graph while loops
#   jax     = REMAT=flash + cb        none    = REMAT=none (no per-layer remat) + cb
#   fa4     = FA4/CuTe attention + cb  fa4none = FA4 + REMAT=none + cb
#   a1      = CE label term as a select (pls 5ab04d3e45)   gif = a1 + CE gradient in the forward
#   dbase, dbase2, da1, dgif = base, base again, a1, gif with --xla_gpu_exclude_nondeterministic_ops (per-step loss bitwise?)
# Per arm a SUMMARY line (exit code, peak memory, mean and median step time over steps >= 3, final loss); per later arm a
# PAIR line, its per-step train/loss minus the first arm's (same seed, data, schedule; bf16 noise is ~1e-3).
set -uo pipefail
CFG=${1:?usage: ab_arms.sbatch <4096|4480> <arm>...}
shift
ARMS=("$@")
[ ${#ARMS[@]} -ge 1 ] || { echo "no arms"; exit 2; }
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
unset VARIANT DATA_EPOCHS PASS_DATA TOTAL_STEPS INIT_FROM SEED BATCH PDP NUM_GPUS TIE PRECISION XLA_FLAGS ATTN_BACKEND REMAT
case $CFG in
  4096) export SIZE=300m SEQ_LEN=4096; unset VOCAB_K ;;
  4480) export SIZE=300m SEQ_LEN=4480 VOCAB_K=32000 ;;
  5504) export SIZE=300m SEQ_LEN=5504 VOCAB_K=8000 ;;   # charwin v8k (w138)
  4096v8k) export SIZE=300m SEQ_LEN=4096 VOCAB_K=8000 ;;
  *) echo "unknown config $CFG"; exit 2 ;;
esac
export SMOKE_STEPS=${SMOKE_STEPS:-60}
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}   # wq guest: SLURM_JOB_ID is the host's
LOGDIR=$MAIN/logs/slurm
nvidia-smi -L
fail=0
declare -A RID
for ARM in "${ARMS[@]}"; do
  if [ $ARM = old ]; then DIR=$MAIN; PP=""; else DIR=$WT; PP="$WT/lib/levanter/src:$WT/lib/haliax/src:$WT"; fi   # old = the main checkout; every other arm = speed-all
  unset XLA_FLAGS ATTN_BACKEND REMAT LEVANTER_CE_BTILED_ONEHOT LEVANTER_CE_GRAD_IN_FWD LEVANTER_CE_XLA_ONEHOT
  export REMAT=full   # an arm that names no REMAT keeps full remat (the code default is flash since 2026-10-26)
  DET="--xla_gpu_nccl_termination_timeout_seconds=600 --xla_gpu_exclude_nondeterministic_ops=true"
  CB="--xla_gpu_nccl_termination_timeout_seconds=600 --xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL"
  case $ARM in
    old) export ATTN_BACKEND=jax_flash ;;   # the old path: the main checkout defaults to fa4 since 2026-10-26
    base) ;;
    cb) export XLA_FLAGS="$CB" ;;
    jax) export REMAT=flash XLA_FLAGS="$CB" ;;
    none) export REMAT=none XLA_FLAGS="$CB" ;;
    fa4) export ATTN_BACKEND=fa4 XLA_FLAGS="$CB" ;;
    jaxflash) export ATTN_BACKEND=jax_flash XLA_FLAGS="$CB" ;;
    fa4none) export ATTN_BACKEND=fa4 REMAT=none XLA_FLAGS="$CB" ;;
    dots) export REMAT=dots XLA_FLAGS="$CB" ;;
    fa4dots) export ATTN_BACKEND=fa4 REMAT=dots XLA_FLAGS="$CB" ;;
    fa4flash) export ATTN_BACKEND=fa4 REMAT=flash XLA_FLAGS="$CB" ;;
    a1) export LEVANTER_CE_BTILED_ONEHOT=1 ;;
    gif) export LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 ;;
    dbase|dbase2) export XLA_FLAGS="$DET" ATTN_BACKEND=jax_flash ;;   # det arms on JAX flash (FA4 backward is not run-to-run deterministic)
    da1) export XLA_FLAGS="$DET" ATTN_BACKEND=jax_flash LEVANTER_CE_BTILED_ONEHOT=1 ;;
    dgif) export XLA_FLAGS="$DET" ATTN_BACKEND=jax_flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 ;;
    *) echo "unknown arm $ARM"; fail=1; continue ;;
  esac
  export RUN_TAG=-abspd${ARM}-j${JID}
  cd "$DIR"
  got=$(PYTHONPATH=$PP $PY -c 'import levanter; print(levanter.__file__)')
  echo "ARM $ARM dir=$DIR levanter=$got XLA_FLAGS=${XLA_FLAGS:-} ATTN_BACKEND=${ATTN_BACKEND:-} REMAT=${REMAT:-} CE: onehot=${LEVANTER_CE_BTILED_ONEHOT:-0} gif=${LEVANTER_CE_GRAD_IN_FWD:-0}"
  run_id=$(PYTHONPATH=$PP DRY_RUN=1 $PY -m experiments.references.della_muonh_qwen3_scaling 2>/dev/null | sed -n 's#^output: speedrun/\([^/]*\)/.*#\1#p' | head -1)
  case "$run_id" in *-abspd${ARM}-j${JID}*) ;; *) echo "ARM $ARM: dry run gave '$run_id'"; fail=1; continue ;; esac
  RID[$ARM]=$run_id
  log=$LOGDIR/speedall-ab_${JID}_${run_id}.log
  echo "SMOKE $run_id START $(date +%H:%M:%S)"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > $log.mem &
  mpid=$!
  PYTHONPATH=$PP timeout --signal=TERM ${RUN_TIMEOUT:-10m} $PY -m experiments.references.della_muonh_qwen3_scaling > $log 2>&1
  rc=$?
  kill $mpid
  peak=$(sort -n $log.mem | tail -1)
  err=$(grep -a -m1 -E "RESOURCE_EXHAUSTED|out of memory|Traceback|NotImplementedError|CUDA_ERROR" $log | cut -c1-200)
  { [ "$rc" -ne 0 ] || [ -n "$err" ]; } && fail=1
  echo "SMOKE $run_id END rc=$rc $(date +%H:%M:%S)"
  $PY - "$run_id" "$rc" "$err" "$peak" <<'PY'
import sys
import numpy as np
import wandb
run_id, rc, err, peak = sys.argv[1:5]
try:
    run = wandb.Api(timeout=60).run(f"marin-della/{run_id}")
    s = run.summary
    d = [r["throughput/duration"] for r in run.scan_history(keys=["_step", "throughput/duration"]) if r["_step"] >= 3]
    print(f"SUMMARY {run_id} rc={rc} peak_mem_MiB={peak} step_ms={np.mean(d) * 1e3:.0f}(n={len(d)}) median_step_ms={np.median(d) * 1e3:.0f} "
          f"loss={s.get('train/loss')} eval_loss={s.get('eval/loss')} err='{err}'", flush=True)
except Exception as e:
    print(f"SUMMARY {run_id} rc={rc} peak_mem_MiB={peak} wandb_error={str(e)[:160]} err='{err}'", flush=True)
PY
  case "$run_id" in *-abspd*-j${JID}*) rm -rf "$MARIN_PREFIX/speedrun/$run_id" ;; esac
done
cd "$WT"
REF=${ARMS[0]}
for ARM in "${ARMS[@]:1}"; do
  [ -n "${RID[$REF]:-}" ] && [ -n "${RID[$ARM]:-}" ] || continue
  $PY - "${RID[$REF]}" "${RID[$ARM]}" <<'PY'
import sys
import numpy as np
import wandb
a, b = sys.argv[1:3]
try:
    api = wandb.Api(timeout=60)
    h = {r: {x["_step"]: x["train/loss"] for x in api.run(f"marin-della/{r}").scan_history(keys=["_step", "train/loss"])} for r in (a, b)}
    steps = sorted(set(h[a]) & set(h[b]))
    d = np.array([h[b][t] - h[a][t] for t in steps])
    print(f"PAIR {b} - {a}: steps={len(steps)} max_abs_loss_diff={np.max(np.abs(d)):.3e} bitwise={bool(np.all(d == 0))} mean_diff={d.mean():+.5f} "
          f"last10_mean_diff={d[-10:].mean():+.5f} first5={np.round(d[:5], 5).tolist()}", flush=True)
except Exception as e:
    print(f"PAIR {a} {b}: wandb_error={str(e)[:160]}", flush=True)
PY
done
echo "SMOKE DONE $(date +%H:%M:%S) fail=$fail"
exit $fail
