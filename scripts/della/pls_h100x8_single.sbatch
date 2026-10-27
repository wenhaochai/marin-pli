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
cd /scratch/gpfs/GROUP/USER/project/marin-pls   # the pls branch (ported from pls-mtp 1d96c98bee for the 48-layer L1 / L2 runs, 2026-10-21)
source scripts/della/pls_env.sh
PT=/scratch/gpfs/GROUP/USER/tools/packed_task.sh
LAUNCH=experiments.references.della_muonh_qwen3_scaling
unset SMOKE_STEPS TOTAL_STEPS INIT_FROM CPT_TAG TIE PRECISION EMA_BETA DEVICE_TAG DRY_RUN
unset SIZE LAYERS ARCH PDP PLS_W PLS_DETACH PLS_SEP PLS_DETACH_BB PLS_LOCAL PLS_SHAPE PLS_PCGRAD PLS_OFF PLS_ON PLS_PROBE PLS_MON PLS_EVAL SEED RUN_TAG LR_MUL WARMUP HEADS_FROM PARAM_MAP PLS_SCAN_FIX PLS_OPT_SPLIT HEADS_WARMUP PATHS_SHARD PARAM_MAP_PDP PARAM_MAP_MEM_FRACTION PLS_MTP_W PLS_FUT NUM_GPUS PROBE_FROM ROUTE GAIN_INIT GAIN_STEP MHC_RES AR_QCLIP PLS_VIS AR_VNORM HC_STREAMS AR_BLOCKS ATTN_BACKEND PLS_FOLD_NORM PLS_SERIAL HC_RW

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
# Speed items (owner 2026-10-25, via the marin.cpp session: ship every speed item that can go in). None changes the math:
#   LOG_JAXPR=0  no jaxpr / StableHLO text dumps (the same compiled programs; 35-213 s of idle GPUs per process start)
#   CKPT_MIN=30  temporary checkpoints every 30 min instead of 10 (each save blocks the loop ~20 s); the training task
#                also saves once at LEVANTER_CKPT_DEADLINE, 150 s before its time limit, so a segment end redoes ~no work
#   LEVANTER_CE_BTILED_ONEHOT=1  the label term of the b-tiled CE backward as a select, not a scatter: one fused pass
#                instead of three over each 8192 x 128256 block (bitwise equal on H100: det A/B wq w205 vs w206, every
#                logged metric over 40 steps, 2026-10-25); the microbatch-1 designs take another CE path and are unchanged
#   EVAL_PDP=4   eval at 4 sequences per device: the microbatch-1 designs (hc, mhc, attnres) evaluate in a quarter of
#                the batches (HC: 7.5 -> 5.1 min per eval, no OOM, wq w225 vs w226); training is untouched and the eval
#                values move at summation-order level (b-tiled instead of the streaming CE); PDP=4 designs are unchanged
#   LEVANTER_CE_GRAD_IN_FWD=1  the b-tiled CE computes its gradients in the forward from one logits GEMM per block (3
#                GEMMs instead of 4; preln d48 13.10 -> 12.13 s/step with A1). Two identical base runs differ already
#                (GEMM autotuning picks per run: w205 vs w229 up to 0.076 nats over 40 det steps); against the base run
#                of its own autotune class (w229) it moves the logged losses by at most 1e-6 and does not grow (w217,
#                w218), inside the twins' envelope (plan rule G4)
#   (2026-10-26 02:25-02:50 its fallback for microbatch-1 shards built a sharded NamedArray inside the shard_map and failed
#   four HC smokes; fixed in 867c735932; on GPU the fallback is bitwise equal to the old path: hc w255 = w219, 40 det steps)
#   LEVANTER_CE_LSE_KERNEL=1  inside the gradient-in-forward block one Triton kernel (the Tokenizer Size session's) runs the
#                logits GEMM and writes the bf16 logits, the label logit and the logsumexp (preln d48 12.13 -> 10.06 s/step,
#                w299). Gradients differ from the XLA path by rel 1.5e-6 to 1.6e-5, against 2e-3 to 1.3e-2 for both paths to
#                an f32 reference, bias ~1e-10 (w306); trains on after an eval and a checkpoint (w299, w301). Only the
#                b-tiled path (PDP=4 designs) reaches it. Owner popup 2026-10-26 ~06:40: on for all pls jobs.
#   PLS_FOLD_NORM=1  the probes model (PLS_DETACH_BB=1) folds each readout's RMSNorm scale into its head (f32 product), so
#                its CE input is not differentiated and XLA drops the input-gradient GEMM: preln probes 7.31 -> 6.13 s/step
#                (w377), HC probes 20.30 -> 17.44 (w349); 40 det steps inside the twins' envelope (w360-w362); the scale's
#                gradient against f32: 0.98% at init (today 0.41%), 0.10% trained (today 0.30%) (w367). Separate-heads
#                runs are unchanged (bitwise on CPU). Owner popup 2026-10-26 ~16:40: on for all queued probes runs.
#   LEVANTER_NORM_GUARD=1e-3  stop when a weight that MuonH / AdamH keeps at a constant norm (projections, lm_head, the 47
#                aux_lm_heads) moves by more than 0.1% between watch steps (a silent corruption: the marin vocab run w158);
#                the stop exits 99 below, so wq resumes from the last checkpoint. No false stop over 40 steps with a mid
#                eval and checkpoint (w354); stops at 1e-9 (w355). Owner popup 2026-10-26 ~16:40: on for all pls jobs.
export LOG_JAXPR=0 CKPT_MIN=30 LEVANTER_CE_BTILED_ONEHOT=1 EVAL_PDP=4 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1
export PLS_FOLD_NORM=1 LEVANTER_NORM_GUARD=1e-3
# HCMB: HC (ARCH=hc) and mHC (ARCH=mhc) at 4 sequences per microbatch, the CE kept on the xla streaming path by a b-tiled threshold above any
# shard (only the gradient summation order changes; eval also streams). 24.78 -> 19.37 s/step (w442, w444); train-step temp
# 21.67 GiB against 17.27 at microbatch 1 (w461, w462), under half the ~64 GiB pool. Runs whose SINGLE sets their own CE
# threshold or microbatch keep theirs. A smoke that fails at microbatch 4 is run again at microbatch 1, which then trains.
HCMB=
# HC with 8 streams: temp 24.86 GiB at microbatch 4 (w473).
# mHC since 2026-10-27 (owner popup): 25.35 -> 19.67 s/step (w487, w488), temp 21.71 GiB (w489).
if [[ ( " $SINGLE " == *" ARCH=hc "* || " $SINGLE " == *" ARCH=mhc "* ) && " $SINGLE " != *" LEVANTER_CE_BTILED_MIN_BATCH="* && " $SINGLE " != *" PARAM_MAP_PDP="* ]]; then
  HCMB="PARAM_MAP_PDP=4 LEVANTER_CE_BTILED_MIN_BATCH=1048576"
fi
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
# A host may run a resumable (fetchable-partial) job for less than its --time: it then ends at SLURM_JOB_END_TIME, so
# training stops 5 min before that and resumes in the next job of its chain.
if [[ ${SLURM_JOB_END_TIME:-} =~ ^[0-9]+$ ]] && (( (SLURM_JOB_END_TIME - $(date +%s)) / 60 < JOB_MIN )); then
  JOB_MIN=$(( (SLURM_JOB_END_TIME - $(date +%s)) / 60 ))
fi
[[ $JOB_MIN =~ ^[0-9]+$ ]] && [ "$JOB_MIN" -gt 30 ] || { echo "cannot read the job's time limit from '$lim'"; exit 2; }
FIRST_MIN=${SINGLE_FIRST_STEP_MIN:-25}
STALL_MIN=${SINGLE_STALL_MIN:-30}
export PACKED_LEDGER_DIR=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/packed

[ -n "${SINGLE:-}" ] || { echo "set SINGLE"; exit 2; }
for kv in $SINGLE; do [[ $kv =~ ^[A-Z_][A-Z0-9_]*=[^[:space:]]*$ ]] || { echo "bad assignment '$kv'"; exit 2; }; done
# NUM_GPUS=4 for designs whose 8-GPU program stalls (HC and Full AttnRes under PARAM_MAP=vocab: a device thread misses
# the first AllToAll rendezvous for 40 s and XLA aborts, wq w16/w17 2026-10-22); a 4-GPU job runs beside another on a host.
[[ " $SINGLE " == *" SIZE="* && " $SINGLE " =~ \ NUM_GPUS=([48])\  ]] || { echo "SINGLE needs SIZE and NUM_GPUS=4 or 8: '$SINGLE'"; exit 2; }
NG=${BASH_REMATCH[1]}
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

# A wq job keeps its id (w<n>) across resume segments, so the smoke tag also carries this segment's host job id: with the
# same smoke run id, W&B resumed the first segment's smoke run and dropped steps 0-29 as already logged, so the train-path
# gate below read the first segment's summary on every resume (w28: 7.4239 in all three segments, found 2026-10-24).
SEG=${FETCHED_JOB_ID:+h$SLURM_JOB_ID}
smoke_id=$(run_id "PLS_EVAL=1 SMOKE_STEPS=$SMOKE RUN_TAG=-j${JID}s${SEG}")
[[ $smoke_id == *-j${JID}s${SEG}-smoke$SMOKE ]] || { echo "bad smoke run id '$smoke_id'"; exit 2; }
rid=$(run_id)
[[ $rid == muonh-qwen3-*-della${NG}xh100-* ]] || { echo "bad run id '$rid'"; exit 2; }
out=$MARIN_PREFIX/speedrun/$rid/2026.09.13
FAILDIR=logs/pair_failed; mkdir -p "$FAILDIR"
[ ! -e "$FAILDIR/$rid.fail" ] || { echo "an earlier job of this run failed its smoke: $(cat "$FAILDIR/$rid.fail"); not running"; exit 6; }

# 0. already finished
if [ "$(cat "$out/.executor_status" 2>/dev/null || true)" = SUCCESS ]; then echo "run $rid already finished: nothing to do"; exit 0; fi

# SINGLE_SMOKE_REUSE=1 (the default since 2026-10-27, owner popup; 0 turns it off): a resumed segment skips its smoke
# when the code and the knobs equal those of this run's last
# passed smoke: the key hashes the committed trees of lib/ and experiments/ (the training program; a docs or sbatch
# commit leaves them unchanged), their uncommitted diff, the run's knobs (SINGLE, HCMB, GPU count) and the exported
# LEVANTER_/PLS_/XLA_/... settings. A run that never passed a smoke here, or any change, smokes as before (each segment's
# smoke cost 11-19 min of 4 GPUs, 2026-10-27).
SMOKEDIR=logs/smoke_pass; mkdir -p "$SMOKEDIR"
smoke_key() {
  { git rev-parse HEAD:lib HEAD:experiments || true; git diff HEAD -- lib experiments || true; echo "$SINGLE|$HCMB|$NG"
    env | grep -E '^(LEVANTER_|PLS_|XLA_|LOG_JAXPR=|CKPT_MIN=|EVAL_PDP=|ATTN_BACKEND=|DEPTH_|PARAM_MAP)' | sort || true; } 2>/dev/null | sha256sum | cut -c1-16
}
skey=$(smoke_key)
if [ "${SINGLE_SMOKE_REUSE:-1}" = 1 ] && [ "$(cat "$SMOKEDIR/$rid.key" 2>/dev/null || true)" = "$skey" ]; then
  echo "smoke skipped: same code and knobs as this run's last passed smoke (key $skey)"
else
  # 1. smoke copy
  slog=logs/slurm/${JNAME}_${JID}_smoke.log
  r=0
  # shellcheck disable=SC2086
  env $SINGLE $HCMB PLS_EVAL=1 SMOKE_STEPS=$SMOKE RUN_TAG=-j${JID}s${SEG} bash $PT smoke - other $NG $SMOKE_LIMIT -- "$PY" -m $LAUNCH > "$slog" 2>&1 || r=$?
  if [ $r -ne 0 ] && [ -n "$HCMB" ]; then
    echo "smoke at microbatch 4 failed rc=$r $(date +%H:%M:%S): smoke and training at microbatch 1"
    if [[ $smoke_id == *-j${JID}s${SEG}-smoke$SMOKE ]]; then
      rm -rf -- "$MARIN_PREFIX/speedrun/$smoke_id" "$MARIN_PREFIX/tmp/checkpoints-temp$MARIN_PREFIX/speedrun/$smoke_id"
    fi
    HCMB=; r=0
    # shellcheck disable=SC2086
    env $SINGLE PLS_EVAL=1 SMOKE_STEPS=$SMOKE RUN_TAG=-j${JID}s${SEG} bash $PT smoke - other $NG $SMOKE_LIMIT -- "$PY" -m $LAUNCH > "$slog" 2>&1 || r=$?
  fi
  echo "smoke $smoke_id exited rc=$r $(date +%H:%M:%S)"
  if [[ $smoke_id == *-j${JID}s${SEG}-smoke$SMOKE ]]; then
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
  # 48-layer runs: the 8-GPU readouts went wrong before (pls_d48_g4_submit.sh), so they take the same check; trained from
  # scratch, the train loss at step 30 may sit a little above the eval, hence a 1.0-nat margin there.
  GATE=; [[ $SINGLE == *PROBE_FROM=* ]] && GATE=0.5; [[ " $SINGLE " == *" LAYERS=48 "* ]] && GATE=${GATE:-1.0}
  if [ -n "$GATE" ]; then
    wsum=$(ls -td "$PWD"/wandb/*-"$smoke_id" /scratch/gpfs/GROUP/USER/wandb-runs/wandb/*-"$smoke_id" 2>/dev/null | head -1 || true)   # ls exits 1 when one of the two globs matches nothing (w15 died here silently)
    tr_ev=$(python3 -c 'import json, sys; s = json.load(open(sys.argv[1] + "/files/wandb-summary.json")); k = max((k for k in s if k.startswith("train/pls/L")), key=lambda k: int(k[11:])); print(s[k], s["eval/loss"])' "$wsum" 2>/dev/null || true)
    echo "smoke: train-path last readout and eval loss: ${tr_ev:-missing} ($wsum)"
    if ! awk -v t="${tr_ev% *}" -v e="${tr_ev#* }" -v g="$GATE" 'BEGIN { exit !(t != "" && e != "" && t - e < g) }'; then
      echo "job $JID: smoke train-path readout ${tr_ev:-missing}" > "$FAILDIR/$rid.fail"; echo "train-path last readout is off (or missing): no training"; exit 5
    fi
  fi
  skey=$(smoke_key)   # HCMB may have fallen back to microbatch 1
  echo "$skey" > "$SMOKEDIR/$rid.key"
fi

# 2. training
TRAIN_LIMIT=$(( JOB_MIN * 60 - SECONDS - 300 ))s
DEADLINE=$(( $(date +%s) + ${TRAIN_LIMIT%s} - 150 ))
echo "training limit $TRAIN_LIMIT, checkpoint deadline $(date -d @$DEADLINE +%F_%T)"
log=logs/slurm/${JNAME}_${JID}_train.log
# shellcheck disable=SC2086
setsid env $SINGLE $HCMB LEVANTER_CKPT_DEADLINE=$DEADLINE bash $PT train "${rid#muonh-qwen3-}" main $NG "$TRAIN_LIMIT" -- "$PY" -m $LAUNCH > "$log" 2>&1 &
pid=$!
watchdog "$pid" "$log" & dog=$!
echo "run $rid [$SINGLE${HCMB:+ $HCMB}]: pid $pid $(date +%H:%M:%S)"
rc=0; wait "$pid" || rc=$?
pid=
echo "run $rid exited rc=$rc $(date +%H:%M:%S)"
# LEVANTER_NORM_GUARD stopped it (a weight its optimizer keeps at a constant norm moved: an update read a wrong value,
# the marin vocab run w158): exit 99, so wq runs it again from its last checkpoint, and no fail marker
if [ $rc -ne 0 ] && grep -q "norm guard: " "$log"; then echo "run $rid: stopped by the norm guard; wq resumes it from its last checkpoint"; rc=99; fi
# a crash (not the time limit: timeout's 124, or 143 after TERM; not the norm guard's 99) stops the later jobs of the chain
if [ $rc -ne 0 ] && [ $rc -ne 124 ] && [ $rc -ne 143 ] && [ $rc -ne 99 ]; then echo "job $JID: training rc=$rc" > "$FAILDIR/$rid.fail"; fi

# 3. a finished probe run drops its checkpoints (only this run's own dir, only for a probe run id)
if [ $rc -eq 0 ] && [[ $rid == *-probes[0-9]*at[0-9]* ]] && [ "$(cat "$out/.executor_status" 2>/dev/null || true)" = SUCCESS ]; then
  rm -rf -- "$out"/checkpoints/step-* "$MARIN_PREFIX/tmp/checkpoints-temp$MARIN_PREFIX/speedrun/$rid"
  echo "probe run finished: its checkpoints removed"
fi
exit $rc
