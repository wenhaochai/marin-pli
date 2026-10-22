#!/bin/bash
#SBATCH --job-name=pls-single
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96   # 8-GPU jobs ask for all 96 CPUs: priority JobSize counts CPUs
#SBATCH --mem-per-gpu=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out

# One VARIANT=pls run on all 8 GPUs of a node (NUM_GPUS=8), with its own smoke copy (packing rules: an independent run
# is its own job carrying its own smoke). SINGLE holds the run's knobs as space-separated VAR=value assignments, e.g.
#   SINGLE="SIZE=300m NUM_GPUS=8 PLS_W=0 PLS_EVAL=0 PLS_MTP_W=0.3" sbatch --time=... pls_h100x8_single.sbatch;
# every other pls knob is cleared first. Steps, each through tools/packed_task.sh (ledger logs/packed/<jobid>.tsv):
#   0. a run whose output already says SUCCESS (an earlier job of an afterany chain finished it) exits 0 at once;
#   1. smoke copy: SMOKE_STEPS=SINGLE_SMOKE_STEPS (30) as a throwaway run (-j<jobid>s-smoke30, kind other), with the
#      per-layer eval on (PLS_EVAL=1), whose last readout must equal the main eval (as in pls_h100x8_pair.sbatch);
#   2. training (kind main), resumable: it stops 5 min before the job's end and the next job of the chain resumes it.
#      A run with no first step after SINGLE_FIRST_STEP_MIN (25) min or a silent log for SINGLE_STALL_MIN (30) min is
#      stopped (Della cancels jobs whose GPUs sit at 0% for 90 min).
#   3. a PROBE_FROM run (id *-probes<n>at<step>) that finished deletes its own checkpoints: they hold every probe head
#      with its optimizer state (~100 GB at 1_2b) and nothing reads them; the metrics live in W&B.
set -euo pipefail   # every command that may fail below is written as `x=0; cmd || x=$?` or `|| true`
cd /scratch/gpfs/GROUP/USER/project/marin-pls-mtp   # the pls-mtp worktree (its code; the venv stays marin-pls's, pls_env.sh)
source scripts/della/pls_env.sh
PT=/scratch/gpfs/GROUP/USER/tools/packed_task.sh
LAUNCH=experiments.references.della_muonh_qwen3_scaling
unset SMOKE_STEPS TOTAL_STEPS INIT_FROM CPT_TAG TIE PRECISION EMA_BETA DEVICE_TAG DRY_RUN
unset SIZE LAYERS ARCH PDP PLS_W PLS_DETACH PLS_SEP PLS_DETACH_BB PLS_LOCAL PLS_SHAPE PLS_PCGRAD PLS_OFF PLS_ON PLS_PROBE PLS_MON PLS_EVAL SEED RUN_TAG LR_MUL WARMUP HEADS_FROM PARAM_MAP PLS_SCAN_FIX PLS_OPT_SPLIT HEADS_WARMUP PATHS_SHARD PARAM_MAP_PDP PARAM_MAP_MEM_FRACTION PLS_MTP_W PLS_FUT NUM_GPUS PROBE_FROM

module load proxy/default 2>/dev/null || true  # wandb only
export MARIN_PREFIX=${MARIN_PREFIX:-/scratch/gpfs/GROUP/USER/marin_store_big}
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della

# W&B through the proxy: offline when it does not answer within 20 s (2026-10-16 evening: the compute-node proxy dropped
# W&B for hours; wandb.init timed out after 90 s and the process hung until the smoke limit). An offline run is synced
# later from a vis node with `wandb sync`; logs/wandb_offline.txt lists the jobs that ran offline.
if [ "${WANDB_MODE:-}" != offline ] && ! curl -s -o /dev/null --max-time 20 https://api.wandb.ai/healthz; then
  export WANDB_MODE=offline
  echo "W&B not reachable through the proxy: running offline (sync later)"
  echo "$(date +%F_%T) job ${FETCHED_JOB_ID:-$SLURM_JOB_ID} ($PWD) offline" >> /scratch/gpfs/GROUP/USER/project/marin-pls/logs/wandb_offline.txt
fi
export VARIANT=pls
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
SMOKE=${SINGLE_SMOKE_STEPS:-30}
SMOKE_LIMIT=${SINGLE_SMOKE_MIN:-25}m
# Run by a host job through tools/fetch_pending.sh, FETCHED_JOB_ID is this job's id and SLURM_JOB_ID the host's: names
# (logs, smoke run tags, fail markers) and the time limit come from this job's own id, name and --time (fetch_pending.sh
# stops it at that --time); the ledger stays the host's.
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
if [ -n "${FETCHED_JOB_ID:-}" ]; then
  # wq jobs (w<n>, owner 2026-10-21) are not in sacct: the host passes the name and the end time, Slurm guests too
  JNAME=${FETCHED_JOB_NAME:-$(sacct -X -n -j "$FETCHED_JOB_ID" -o JobName%80 | head -1 | tr -d ' ')}
  if [[ ${SLURM_JOB_END_TIME:-} =~ ^[0-9]+$ ]]; then lim=$(( (SLURM_JOB_END_TIME - $(date +%s)) / 60 ))
  else lim=$(sacct -X -n -j "$FETCHED_JOB_ID" -o Timelimit | head -1 | tr -d ' '); fi
else
  lim=$(squeue -h -j "$SLURM_JOB_ID" -o %l); JNAME=$SLURM_JOB_NAME   # the job's own --time: [D-]HH:MM:SS
fi
JOB_MIN=$(awk -F'[-:]' '{ if (NF == 4) print $1 * 1440 + $2 * 60 + $3; else if (NF == 3) print $1 * 60 + $2; else print $1 }' <<< "$lim")
[[ $JOB_MIN =~ ^[0-9]+$ ]] && [ "$JOB_MIN" -gt 30 ] || { echo "cannot read the job's time limit from '$lim'"; exit 2; }
FIRST_MIN=${SINGLE_FIRST_STEP_MIN:-25}
STALL_MIN=${SINGLE_STALL_MIN:-30}
export PACKED_LEDGER_DIR=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/packed

[ -n "${SINGLE:-}" ] || { echo "set SINGLE"; exit 2; }
for kv in $SINGLE; do [[ $kv =~ ^[A-Z_][A-Z0-9_]*=[^[:space:]]*$ ]] || { echo "bad assignment '$kv'"; exit 2; }; done
[[ " $SINGLE " == *" SIZE="* && " $SINGLE " == *" NUM_GPUS=8 "* ]] || { echo "SINGLE needs SIZE and NUM_GPUS=8: '$SINGLE'"; exit 2; }
nvidia-smi -L || true
mkdir -p "$PACKED_LEDGER_DIR"; ledger=$PACKED_LEDGER_DIR/$SLURM_JOB_ID.tsv
[[ -s $ledger ]] || printf 'task\trun\tkind\tgpus\tstart\tend\texit\n' > "$ledger"

pid=; dog=
stop() {   # TERM the task's timeout (it passes it on; packed_task.sh writes its ledger line), KILL after 90 s
  local t; t=$(pgrep -P "$1" -x timeout 2>/dev/null | head -1 || true)
  if [ -n "$t" ]; then kill -TERM "$t" 2>/dev/null || true; fi
  for _ in $(seq 1 18); do kill -0 "$1" 2>/dev/null || return 0; sleep 5; done
  if [ -n "$t" ]; then kill -KILL -- "-$t" 2>/dev/null || true; fi
  sleep 5; kill -KILL "$1" 2>/dev/null || true
}
trap 'if [ -n "$dog" ]; then kill "$dog" 2>/dev/null || true; fi; if [ -n "$pid" ]; then stop "$pid"; fi; pls_tmp_cleanup' EXIT
run_id() {   # the launcher's run id for $SINGLE plus extra assignments; a CPU dry run
  # shellcheck disable=SC2086
  basename "$(dirname "$(env $SINGLE ${1:-} DRY_RUN=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu "$PY" -m $LAUNCH 2>>"logs/slurm/${JNAME}_${JID}_dryrun.log" | sed -n 's/^output: //p' || true)")"
}
watchdog() {
  local t0=$SECONDS
  while kill -0 "$1" 2>/dev/null; do
    sleep 60
    if ! grep -q "First train step completed" "$2" 2>/dev/null && (( SECONDS - t0 > FIRST_MIN * 60 )); then
      echo "watchdog: no first step in $2 after $FIRST_MIN min, stopping it $(date +%H:%M:%S)"; stop "$1"; return 0
    fi
    if (( $(date +%s) - $(stat -c %Y "$2" 2>/dev/null || date +%s) > STALL_MIN * 60 )); then
      echo "watchdog: $2 silent for $STALL_MIN min, stopping it $(date +%H:%M:%S)"; stop "$1"; return 0
    fi
  done
}

smoke_id=$(run_id "PLS_EVAL=1 SMOKE_STEPS=$SMOKE RUN_TAG=-j${JID}s")
[[ $smoke_id == *-j${JID}s-smoke$SMOKE ]] || { echo "bad smoke run id '$smoke_id'"; exit 2; }
rid=$(run_id)
[[ $rid == muonh-qwen3-*-della8xh100-* ]] || { echo "bad run id '$rid'"; exit 2; }
out=$MARIN_PREFIX/speedrun/$rid/2026.09.13
FAILDIR=logs/pair_failed; mkdir -p "$FAILDIR"
[ ! -e "$FAILDIR/$rid.fail" ] || { echo "an earlier job of this run failed its smoke: $(cat "$FAILDIR/$rid.fail"); not running"; exit 6; }

# 0. already finished
if [ "$(cat "$out/.executor_status" 2>/dev/null || true)" = SUCCESS ]; then echo "run $rid already finished: nothing to do"; exit 0; fi

# 1. smoke copy
slog=logs/slurm/${JNAME}_${JID}_smoke.log
r=0
# shellcheck disable=SC2086
env $SINGLE PLS_EVAL=1 SMOKE_STEPS=$SMOKE RUN_TAG=-j${JID}s bash $PT smoke - other 8 $SMOKE_LIMIT -- "$PY" -m $LAUNCH > "$slog" 2>&1 || r=$?
echo "smoke $smoke_id exited rc=$r $(date +%H:%M:%S)"
if [[ $smoke_id == *-j${JID}s-smoke$SMOKE ]]; then
  rm -rf -- "$MARIN_PREFIX/speedrun/$smoke_id" "$MARIN_PREFIX/tmp/checkpoints-temp$MARIN_PREFIX/speedrun/$smoke_id"
fi
[ $r -eq 0 ] || { echo "job $JID: smoke failed rc=$r" > "$FAILDIR/$rid.fail"; echo "smoke failed: no training"; exit 4; }
main=$(sed -n 's/.*levanter.eval eval loss: \([0-9.]*\).*/\1/p' "$slog" | tail -1)
last=$(sed -n 's/.*per_layer_qwen3 eval\/L\([0-9]*\) loss \([0-9.]*\).*/\1 \2/p' "$slog" | sort -n | tail -1 | cut -d' ' -f2)
gap=$(awk -v a="${main:-nan}" -v b="${last:-nan}" 'BEGIN { d = a - b; if (d < 0) d = -d; print (a == "nan" || b == "nan") ? "nan" : d }')
echo "smoke: eval loss ${main:-?}, last readout ${last:-?}, gap $gap"
if ! awk -v g="$gap" 'BEGIN { exit !(g != "nan" && g <= 0.005) }'; then
  echo "job $JID: smoke gap $gap" > "$FAILDIR/$rid.fail"; echo "last readout differs from the main eval (or is missing): no training"; exit 5
fi
# Probe runs: the gap above compares two eval programs. The train step is a third program, and at 520m it read the frozen
# head at 9.7 nats against eval 3.55 (15738658, 2026-10-21). So the train path's last readout (the frozen own head, on
# training data, which sits below the eval loss: 1_2b 2.40 vs 3.34) must not exceed the eval loss by 0.5 nats or more.
if [[ $SINGLE == *PROBE_FROM=* ]]; then
  wsum=$(ls -td /scratch/gpfs/GROUP/USER/project/marin-pls-mtp/wandb/*-"$smoke_id" /scratch/gpfs/GROUP/USER/wandb-runs/wandb/*-"$smoke_id" 2>/dev/null | head -1 || true)   # ls exits 1 when one of the two globs matches nothing (pls w15 died here silently)
  tr_ev=$(python3 -c 'import json, sys; s = json.load(open(sys.argv[1] + "/files/wandb-summary.json")); k = max((k for k in s if k.startswith("train/pls/L")), key=lambda k: int(k[11:])); print(s[k], s["eval/loss"])' "$wsum" 2>/dev/null || true)
  echo "smoke: train-path last readout and eval loss: ${tr_ev:-missing} ($wsum)"
  if ! awk -v t="${tr_ev% *}" -v e="${tr_ev#* }" 'BEGIN { exit !(t != "" && e != "" && t - e < 0.5) }'; then
    echo "job $JID: smoke train-path readout ${tr_ev:-missing}" > "$FAILDIR/$rid.fail"; echo "train-path last readout is off (or missing): no training"; exit 5
  fi
fi

# 2. training
TRAIN_LIMIT=$(( JOB_MIN * 60 - SECONDS - 300 ))s
echo "training limit $TRAIN_LIMIT"
log=logs/slurm/${JNAME}_${JID}_train.log
# shellcheck disable=SC2086
setsid env $SINGLE bash $PT train "${rid#muonh-qwen3-}" main 8 "$TRAIN_LIMIT" -- "$PY" -m $LAUNCH > "$log" 2>&1 &
pid=$!
watchdog "$pid" "$log" & dog=$!
echo "run $rid [$SINGLE]: pid $pid $(date +%H:%M:%S)"
rc=0; wait "$pid" || rc=$?
pid=
echo "run $rid exited rc=$rc $(date +%H:%M:%S)"
# a crash (not the time limit: timeout's 124, or 143 after TERM) stops the later jobs of the chain
if [ $rc -ne 0 ] && [ $rc -ne 124 ] && [ $rc -ne 143 ]; then echo "job $JID: training rc=$rc" > "$FAILDIR/$rid.fail"; fi

# 3. a finished probe run drops its checkpoints (only this run's own dir, only for a probe run id)
if [ $rc -eq 0 ] && [[ $rid == *-probes[0-9]*at[0-9]* ]] && [ "${KEEP_PROBE_CKPT:-0}" != 1 ] && [ "$(cat "$out/.executor_status" 2>/dev/null || true)" = SUCCESS ]; then
  rm -rf -- "$out"/checkpoints/step-* "$MARIN_PREFIX/tmp/checkpoints-temp$MARIN_PREFIX/speedrun/$rid"
  echo "probe run finished: its checkpoints removed"
fi
exit $rc
