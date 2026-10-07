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

# Two pls_h100x4.sbatch runs packed on one 8xH100 node, GPUs 0-3 and 4-7 (8-GPU jobs start sooner than 4-GPU ones).
# PAIR_A and PAIR_B hold each run's knobs as space-separated VAR=value assignments, e.g.
#   PAIR_A="SIZE=130m PLS_W=1 PLS_SEP=1 PLS_ON=300:400" PAIR_B="SIZE=130m PLS_W=1 PLS_SEP=1 PLS_ON=1500:1600";
# every other pls knob is cleared first, so the submitting shell cannot change either run. Each run is the 4-GPU
# script's command, resumable as there. Logs: logs/slurm/<job-name>_<jobid>_{a,b}.log. Levanter starts no
# jax.distributed here, so the two single-process runs share only the host.
# Della cancels a job whose GPUs sit at 0% for 90 min (memory reference_della_idle_gpu_kill):
#   - a run that fails stops the other at once, and the job exits with its code;
#   - after one run ends cleanly the other gets PAIR_GRACE_MIN (default 75) minutes; still running then, it is stopped
#     (it resumes from its last checkpoint, saved every 10 min) and the job exits 3.
# Pair runs of the same length, so the grace is rarely used.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
PY=${PAIR_TEST_PY:-$PY}   # tests only: a stub in place of the launcher
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

[ -n "${PAIR_A:-}" ] && [ -n "${PAIR_B:-}" ] || { echo "set PAIR_A and PAIR_B"; exit 2; }
for s in "$PAIR_A" "$PAIR_B"; do
  for kv in $s; do [[ $kv =~ ^[A-Z_][A-Z0-9_]*=[^[:space:]]*$ ]] || { echo "bad assignment '$kv' in '$s'"; exit 2; }; done
  [[ " $s " == *" SIZE="* ]] || { echo "SIZE missing in '$s'"; exit 2; }
done
nvidia-smi -L || true

specs=("$PAIR_A" "$PAIR_B"); names=(a b); pids=()
trap 'kill ${pids[*]:-} 2>/dev/null || true' EXIT
for i in 0 1; do
  gpus=$([ $i = 0 ] && echo 0,1,2,3 || echo 4,5,6,7)
  # shellcheck disable=SC2086  # word-split the VAR=value assignments on purpose
  env ${specs[$i]} CUDA_VISIBLE_DEVICES=$gpus "$PY" -m experiments.references.della_muonh_qwen3_scaling \
    > logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${names[$i]}.log 2>&1 &
  pids+=($!)
  echo "run ${names[$i]} [${specs[$i]}] on GPUs $gpus: pid ${pids[$i]} $(date +%H:%M:%S)"
done

wait -n "${pids[@]}"; rc=$?
first=-1; for i in 0 1; do kill -0 "${pids[$i]}" 2>/dev/null || first=$i; done
other=$((1 - first))
echo "run ${names[$first]} exited first, rc=$rc $(date +%H:%M:%S)"
if [ "$rc" -ne 0 ]; then
  echo "stopping run ${names[$other]}: its partner failed"; kill "${pids[$other]}" 2>/dev/null || true
  wait "${pids[$other]}" 2>/dev/null || true; exit "$rc"
fi
waited=0
while kill -0 "${pids[$other]}" 2>/dev/null && [ $waited -lt $GRACE ]; do sleep 15; waited=$((waited + 15)); done
if kill -0 "${pids[$other]}" 2>/dev/null; then
  echo "run ${names[$other]} still running after ${PAIR_GRACE_MIN:-75} min alone: stopping it so the idle GPUs do not cancel the job; resume it"
  kill -TERM "${pids[$other]}" 2>/dev/null || true; sleep 60; kill -KILL "${pids[$other]}" 2>/dev/null || true
  wait "${pids[$other]}" 2>/dev/null || true; exit 3
fi
wait "${pids[$other]}"; rc2=$?
echo "run ${names[$other]} exited rc=$rc2 $(date +%H:%M:%S)"
exit "$rc2"
