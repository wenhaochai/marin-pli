#!/bin/bash
#SBATCH --job-name=pls-pair
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96   # 8-GPU jobs ask for all 96 CPUs: priority JobSize counts CPUs
#SBATCH --mem-per-gpu=24G
#SBATCH --time=03:20:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out

# Two pls_h100x4.sbatch runs packed on one 8xH100 node, GPUs 0-3 and 4-7 (packing rules, owner via the terminal
# session 2026-10-07: two 4-GPU runs within 1 h of each other share an 8-GPU job; every job carries its own smoke copy).
# PAIR_A and PAIR_B hold each run's knobs as space-separated VAR=value assignments, e.g.
#   PAIR_A="SIZE=130m PLS_W=1 PLS_SEP=1 PLS_ON=300:400" PAIR_B="SIZE=130m PLS_W=1 PLS_SEP=1 PLS_ON=1500:1600";
# every other pls knob is cleared first, so the submitting shell cannot change either run.
# Steps, each through tools/packed_task.sh (task ledger logs/packed/<jobid>.tsv, which the record uses to split the
# job's H100-hours):
#   1. smoke copies: each half runs its own config for PAIR_SMOKE_STEPS (30) steps as a throwaway run
#      (-j<jobid><a|b>-smoke30, kind other); both must pass, or the job exits before training;
#   2. training: both runs (kind main), resumable as in the 4-GPU script. Logs logs/slurm/<job-name>_<jobid>_{a,b}.log.
# Della cancels a job whose GPUs sit at 0% for 90 min (memory reference_della_idle_gpu_kill):
#   - a run that fails stops the other at once, and the job exits with its code;
#   - after one run ends cleanly the other gets PAIR_GRACE_MIN (75) minutes; still running then, it is stopped
#     (it resumes from its last checkpoint, saved every 10 min) and the job exits 3.
# A task is stopped by TERM to its timeout, which passes it on, so packed_task.sh still writes its ledger line.
# A training task with no first step after PAIR_FIRST_STEP_MIN (25) min or a silent log for PAIR_STALL_MIN (30) min is
# stopped (a hung half must not hold 4 GPUs at 0%). Run ids come from CPU dry runs before any GPU task starts.
set -euo pipefail   # every command that may fail below is written as `x=0; cmd || x=$?` or `|| true`
cd /scratch/gpfs/GROUP/USER/project/marin-pls-mtp   # the pls-mtp worktree (its code; the venv stays marin-pls's, pls_env.sh)
source scripts/della/pls_env.sh
PY=${PAIR_TEST_PY:-$PY}   # tests only: a stub in place of the launcher
PT=/scratch/gpfs/GROUP/USER/tools/packed_task.sh
LAUNCH=experiments.references.della_muonh_qwen3_scaling
unset SMOKE_STEPS TOTAL_STEPS INIT_FROM CPT_TAG TIE PRECISION EMA_BETA DEVICE_TAG DRY_RUN
unset SIZE LAYERS ARCH PDP PLS_W PLS_DETACH PLS_SEP PLS_DETACH_BB PLS_LOCAL PLS_SHAPE PLS_PCGRAD PLS_OFF PLS_ON PLS_PROBE PLS_MON PLS_EVAL SEED RUN_TAG LR_MUL WARMUP HEADS_FROM PARAM_MAP PLS_SCAN_FIX PLS_OPT_SPLIT HEADS_WARMUP PATHS_SHARD PARAM_MAP_PDP PARAM_MAP_MEM_FRACTION PLS_MTP_W PLS_FUT

module load proxy/default 2>/dev/null || true  # wandb only
export MARIN_PREFIX=${MARIN_PREFIX:-/scratch/gpfs/GROUP/USER/marin_store_big}
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export VARIANT=pls
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
GRACE=$(( ${PAIR_GRACE_MIN:-75} * 60 ))
SMOKE=${PAIR_SMOKE_STEPS:-30}
SMOKE_LIMIT=${PAIR_SMOKE_MIN:-15}m   # each smoke copy's limit; 48-layer designs on 4 GPUs compile and step for 15-35 min
JOB_MIN=${PAIR_JOB_MIN:-200}           # the job's --time in minutes (03:20:00); training stops 5 min before it, so
                                      # timeout, not Slurm, ends it and packed_task.sh writes its ledger line
FIRST_MIN=${PAIR_FIRST_STEP_MIN:-25}   # a training task with no first step by then is stopped (hang watchdog)
STALL_MIN=${PAIR_STALL_MIN:-30}        # ... and one whose log stops growing for this long
export PACKED_LEDGER_DIR=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/packed   # never the submit dir

[ -n "${PAIR_A:-}" ] && [ -n "${PAIR_B:-}" ] || { echo "set PAIR_A and PAIR_B"; exit 2; }
for s in "$PAIR_A" "$PAIR_B"; do
  for kv in $s; do [[ $kv =~ ^[A-Z_][A-Z0-9_]*=[^[:space:]]*$ ]] || { echo "bad assignment '$kv' in '$s'"; exit 2; }; done
  [[ " $s " == *" SIZE="* ]] || { echo "SIZE missing in '$s'"; exit 2; }
done
nvidia-smi -L || true

# the ledger header first: packed_task.sh writes it when the file is empty, and two tasks starting in the same second
# could both do so (audit 2026-10-07)
mkdir -p "$PACKED_LEDGER_DIR"; ledger=$PACKED_LEDGER_DIR/$SLURM_JOB_ID.tsv
[[ -s $ledger ]] || printf 'task\trun\tkind\tgpus\tstart\tend\texit\n' > "$ledger"

specs=("$PAIR_A" "$PAIR_B"); names=(a b); gpus=(0,1,2,3 4,5,6,7); pids=(); dogs=()
stop() {   # $1 = a task's packed_task.sh pid. TERM its timeout (which passes it to the training process and then lets
  # packed_task.sh write the ledger line); after 90 s KILL what is left. timeout runs in its own process group.
  local t; t=$(pgrep -P "$1" -x timeout 2>/dev/null | head -1 || true)
  if [ -n "$t" ]; then kill -TERM "$t" 2>/dev/null || true; fi
  for _ in $(seq 1 18); do kill -0 "$1" 2>/dev/null || return 0; sleep 5; done
  if [ -n "$t" ]; then kill -KILL -- "-$t" 2>/dev/null || true; fi
  sleep 5; kill -KILL "$1" 2>/dev/null || true   # packed_task.sh first gets to write its ledger line
}
trap 'for d in ${dogs[*]:-}; do kill "$d" 2>/dev/null || true; done; for p in ${pids[*]:-}; do stop "$p"; done; pls_tmp_cleanup' EXIT
run_id() {   # the run id the launcher builds for spec $1 plus extra assignments; a dry run on the CPU, no GPU touched
  # shellcheck disable=SC2086
  basename "$(dirname "$(env $1 ${2:-} DRY_RUN=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu "$PY" -m $LAUNCH 2>>"logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_dryrun.log" | sed -n 's/^output: //p' || true)")"
}
watchdog() {   # $1 pid, $2 log: stop a training task that hangs (no first step after FIRST_MIN, or a silent log for STALL_MIN),
  # so a hung half cannot hold 4 GPUs at 0% until Della cancels the whole job
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

# every run id first (dry runs on the CPU), before any task holds a GPU
for i in 0 1; do
  smoke_ids[$i]=$(run_id "${specs[$i]}" "SMOKE_STEPS=$SMOKE RUN_TAG=-j${SLURM_JOB_ID}${names[$i]}")
  [[ ${smoke_ids[$i]} == *-j${SLURM_JOB_ID}${names[$i]}-smoke$SMOKE ]] || { echo "bad smoke run id '${smoke_ids[$i]}' for ${specs[$i]}"; exit 2; }
  run_ids[$i]=$(run_id "${specs[$i]}")
  [[ ${run_ids[$i]} == muonh-qwen3-* ]] || { echo "bad run id '${run_ids[$i]}' for ${specs[$i]}"; exit 2; }
done
[ "${run_ids[0]}" != "${run_ids[1]}" ] || { echo "both halves would write run ${run_ids[0]}"; exit 2; }
# a failed smoke earlier in this pair's chain (marker keyed by run a's id): stop before touching the GPUs; delete the
# marker by hand once the cause is fixed
FAILDIR=logs/pair_failed; mkdir -p "$FAILDIR"
[ ! -e "$FAILDIR/${run_ids[0]}.fail" ] || { echo "an earlier job of this pair failed its smoke: $(cat "$FAILDIR/${run_ids[0]}.fail"); not running"; exit 6; }

# 1. smoke copies, both halves at once
for i in 0 1; do
  # shellcheck disable=SC2086
  setsid env ${specs[$i]} SMOKE_STEPS=$SMOKE RUN_TAG=-j${SLURM_JOB_ID}${names[$i]} CUDA_VISIBLE_DEVICES=${gpus[$i]} \
    bash $PT smoke-${names[$i]} - other 4 $SMOKE_LIMIT -- "$PY" -m $LAUNCH > logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${names[$i]}-smoke.log 2>&1 &
  pids[$i]=$!
  echo "smoke ${names[$i]} ${smoke_ids[$i]} on GPUs ${gpus[$i]}: pid ${pids[$i]} $(date +%H:%M:%S)"
done
smoke_ok=1
for i in 0 1; do
  r=0; wait "${pids[$i]}" || r=$?
  echo "smoke ${names[$i]} exited rc=$r $(date +%H:%M:%S)"; [ $r -eq 0 ] || smoke_ok=0
  # throwaway checkpoints, final and temporary: only this job's tagged smoke directory (never an empty or foreign id)
  if [[ ${smoke_ids[$i]} == *-j${SLURM_JOB_ID}${names[$i]}-smoke$SMOKE ]]; then
    rm -rf -- "$MARIN_PREFIX/speedrun/${smoke_ids[$i]}" "$MARIN_PREFIX/tmp/checkpoints-temp$MARIN_PREFIX/speedrun/${smoke_ids[$i]}"
  fi
done
pids=()
[ $smoke_ok = 1 ] || { echo "job $SLURM_JOB_ID: a smoke copy failed" > "$FAILDIR/${run_ids[0]}.fail"; echo "a smoke copy failed: no training"; exit 4; }
# The smoke's last-layer readout must equal the main eval (both decode the final state with lm_head): at 48 layers under
# FSDP the readout path computed a different function on GPU (2026-10-09, 15255032/15257881) and training followed it.
# Enforced for every design: the DepthBench designs decode outputs(...)[-1] in both paths too (depth_arch_qwen3.py), and
# their FSDP smokes showed gaps of 0.015-0.146 (14969857, 15024096). A failure leaves a marker so that the later jobs
# of an afterany chain stop at once instead of repeating the smoke.
for i in 0 1; do
  slog=logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${names[$i]}-smoke.log
  main=$(sed -n 's/.*levanter.eval eval loss: \([0-9.]*\).*/\1/p' "$slog" | tail -1)
  last=$(sed -n 's/.*per_layer_qwen3 eval\/L\([0-9]*\) loss \([0-9.]*\).*/\1 \2/p' "$slog" | sort -n | tail -1 | cut -d' ' -f2)
  gap=$(awk -v a="${main:-nan}" -v b="${last:-nan}" 'BEGIN { d = a - b; if (d < 0) d = -d; print (a == "nan" || b == "nan") ? "nan" : d }')
  echo "smoke ${names[$i]}: eval loss ${main:-?}, last readout ${last:-?}, gap $gap"
  if ! awk -v g="$gap" 'BEGIN { exit !(g != "nan" && g <= 0.005) }'; then
    echo "smoke ${names[$i]}: last readout differs from the main eval (or is missing): no training"
    echo "job $SLURM_JOB_ID: smoke ${names[$i]} gap $gap" > "$FAILDIR/${run_ids[0]}.fail"; exit 5
  fi
done

# 2. training, limited to the job's remaining time minus 5 min
TRAIN_LIMIT=${PAIR_TRAIN_LIMIT:-$(( JOB_MIN * 60 - SECONDS - 300 ))s}
echo "training limit $TRAIN_LIMIT"
for i in 0 1; do
  log=logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${names[$i]}.log
  # shellcheck disable=SC2086  # word-split the VAR=value assignments on purpose
  setsid env ${specs[$i]} CUDA_VISIBLE_DEVICES=${gpus[$i]} \
    bash $PT train-${names[$i]} "${run_ids[$i]#muonh-qwen3-}" main 4 "$TRAIN_LIMIT" -- "$PY" -m $LAUNCH > "$log" 2>&1 &
  pids[$i]=$!
  watchdog "${pids[$i]}" "$log" & dogs[$i]=$!
  echo "run ${names[$i]} ${run_ids[$i]} [${specs[$i]}] on GPUs ${gpus[$i]}: pid ${pids[$i]} $(date +%H:%M:%S)"
done

rc=0; done_pid=; wait -n -p done_pid "${pids[@]}" || rc=$?
first=0; [ "$done_pid" = "${pids[1]}" ] && first=1
other=$((1 - first))
echo "run ${names[$first]} exited first, rc=$rc $(date +%H:%M:%S)"
if [ "$rc" -ne 0 ]; then
  echo "stopping run ${names[$other]}: its partner failed"; stop "${pids[$other]}"
  wait "${pids[$other]}" 2>/dev/null || true; pids=(); exit "$rc"
fi
waited=0
while kill -0 "${pids[$other]}" 2>/dev/null && [ $waited -lt $GRACE ]; do sleep 15; waited=$((waited + 15)); done
if kill -0 "${pids[$other]}" 2>/dev/null; then
  echo "run ${names[$other]} still running after ${PAIR_GRACE_MIN:-75} min alone: stopping it so the idle GPUs do not cancel the job; resume it"
  stop "${pids[$other]}"; wait "${pids[$other]}" 2>/dev/null || true; pids=(); exit 3
fi
rc2=0; wait "${pids[$other]}" || rc2=$?
echo "run ${names[$other]} exited rc=$rc2 $(date +%H:%M:%S)"
pids=(); exit "$rc2"
