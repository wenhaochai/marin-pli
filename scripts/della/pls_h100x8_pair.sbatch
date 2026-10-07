#!/bin/bash
#SBATCH --job-name=pls-pair
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
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
# A task is stopped by TERM to its training process only, so packed_task.sh still writes its ledger line.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
PY=${PAIR_TEST_PY:-$PY}   # tests only: a stub in place of the launcher
PT=/scratch/gpfs/GROUP/USER/tools/packed_task.sh
LAUNCH=experiments.references.della_muonh_qwen3_scaling
unset SMOKE_STEPS TOTAL_STEPS INIT_FROM CPT_TAG TIE PRECISION EMA_BETA DEVICE_TAG DRY_RUN
unset SIZE LAYERS ARCH PDP PLS_W PLS_DETACH PLS_SEP PLS_DETACH_BB PLS_LOCAL PLS_SHAPE PLS_PCGRAD PLS_OFF PLS_ON PLS_PROBE PLS_MON PLS_EVAL SEED RUN_TAG LR_MUL WARMUP HEADS_FROM

module load proxy/default 2>/dev/null || true  # wandb only
export MARIN_PREFIX=${MARIN_PREFIX:-/scratch/gpfs/GROUP/USER/marin_store_big}
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export VARIANT=pls
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
GRACE=$(( ${PAIR_GRACE_MIN:-75} * 60 ))
SMOKE=${PAIR_SMOKE_STEPS:-30}
TRAIN_LIMIT=${PAIR_TRAIN_LIMIT:-3h}

[ -n "${PAIR_A:-}" ] && [ -n "${PAIR_B:-}" ] || { echo "set PAIR_A and PAIR_B"; exit 2; }
for s in "$PAIR_A" "$PAIR_B"; do
  for kv in $s; do [[ $kv =~ ^[A-Z_][A-Z0-9_]*=[^[:space:]]*$ ]] || { echo "bad assignment '$kv' in '$s'"; exit 2; }; done
  [[ " $s " == *" SIZE="* ]] || { echo "SIZE missing in '$s'"; exit 2; }
done
nvidia-smi -L || true

specs=("$PAIR_A" "$PAIR_B"); names=(a b); gpus=(0,1,2,3 4,5,6,7); pids=()
stop() {   # $1 = a task's packed_task.sh pid. TERM its timeout (which passes it to the training process and then lets
  # packed_task.sh write the ledger line); after 90 s KILL what is left. timeout runs in its own process group.
  local t; t=$(pgrep -P "$1" -x timeout 2>/dev/null | head -1)
  [ -n "$t" ] && kill -TERM "$t" 2>/dev/null
  for _ in $(seq 1 18); do kill -0 "$1" 2>/dev/null || return 0; sleep 5; done
  [ -n "$t" ] && kill -KILL -- "-$t" 2>/dev/null; kill -KILL "$1" 2>/dev/null || true
}
trap 'for p in ${pids[*]:-}; do stop "$p"; done' EXIT
run_id() {   # the run id the launcher builds for spec $1 plus extra assignments
  # shellcheck disable=SC2086
  basename "$(dirname "$(env $1 ${2:-} DRY_RUN=1 "$PY" -m $LAUNCH 2>/dev/null | sed -n 's/^output: //p')")"
}

# 1. smoke copies, both halves at once
for i in 0 1; do
  sid=$(run_id "${specs[$i]}" "SMOKE_STEPS=$SMOKE RUN_TAG=-j${SLURM_JOB_ID}${names[$i]}")
  [[ $sid == *-j${SLURM_JOB_ID}${names[$i]}-smoke$SMOKE ]] || { echo "bad smoke run id '$sid' for ${specs[$i]}"; exit 2; }
  smoke_ids[$i]=$sid
  # shellcheck disable=SC2086
  setsid env ${specs[$i]} SMOKE_STEPS=$SMOKE RUN_TAG=-j${SLURM_JOB_ID}${names[$i]} CUDA_VISIBLE_DEVICES=${gpus[$i]} \
    bash $PT smoke-${names[$i]} - other 4 15m -- "$PY" -m $LAUNCH > logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${names[$i]}-smoke.log 2>&1 &
  pids[$i]=$!
  echo "smoke ${names[$i]} $sid on GPUs ${gpus[$i]}: pid ${pids[$i]} $(date +%H:%M:%S)"
done
smoke_ok=1
for i in 0 1; do
  wait "${pids[$i]}"; r=$?
  echo "smoke ${names[$i]} exited rc=$r $(date +%H:%M:%S)"; [ $r -eq 0 ] || smoke_ok=0
  # throwaway checkpoints: only this job's tagged smoke directory (never an empty or foreign id)
  [[ ${smoke_ids[$i]} == *-j${SLURM_JOB_ID}${names[$i]}-smoke$SMOKE ]] && rm -rf -- "$MARIN_PREFIX/speedrun/${smoke_ids[$i]}"
done
pids=()
[ $smoke_ok = 1 ] || { echo "a smoke copy failed: no training"; exit 4; }

# 2. training
for i in 0 1; do
  rid=$(run_id "${specs[$i]}")
  [ -n "$rid" ] || { echo "no run id for ${specs[$i]}"; exit 2; }
  # shellcheck disable=SC2086  # word-split the VAR=value assignments on purpose
  setsid env ${specs[$i]} CUDA_VISIBLE_DEVICES=${gpus[$i]} \
    bash $PT train-${names[$i]} "${rid#muonh-qwen3-}" main 4 "$TRAIN_LIMIT" -- "$PY" -m $LAUNCH \
    > logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${names[$i]}.log 2>&1 &
  pids[$i]=$!
  echo "run ${names[$i]} $rid [${specs[$i]}] on GPUs ${gpus[$i]}: pid ${pids[$i]} $(date +%H:%M:%S)"
done

wait -n "${pids[@]}"; rc=$?
first=-1; for i in 0 1; do kill -0 "${pids[$i]}" 2>/dev/null || first=$i; done
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
wait "${pids[$other]}"; rc2=$?
echo "run ${names[$other]} exited rc=$rc2 $(date +%H:%M:%S)"
pids=(); exit "$rc2"
