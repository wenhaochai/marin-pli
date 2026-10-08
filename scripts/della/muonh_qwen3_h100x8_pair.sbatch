#!/bin/bash
#SBATCH --job-name=muonh-qwen3-pair
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# Two muonh_qwen3_h100x4.sbatch runs packed on one 8xH100 node (GPUs 0-3 and 4-7), for when the per-user pli-short
# node cap binds. PAIR_A and PAIR_B hold each run's own knobs as space-separated VAR=value assignments (e.g.
# PAIR_A="VARIANT=baseline SEED=0 RUN_TAG=-restore" PAIR_B="VARIANT=ss SEED=0"); knobs common to both come from
# --export as for the 4-GPU script, and the training command is the same. Logs: logs/slurm/<job-name>_<jobid>[PAIR_LOG_TAG]_{a,b}.log.
# Levanter does not initialise jax.distributed here (no cluster detected, as in the 4-GPU runs), so the two
# single-process runs share nothing but the host.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
source scripts/della/job_tmpdir.sh   # TMPDIR under scratch, per task, removed at exit (never the shared /tmp)

module load proxy/default  # wandb only
export MARIN_PREFIX=${MARIN_PREFIX:-/scratch/gpfs/GROUP/USER/marin_store_big}
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export SIZE=${SIZE:-130m}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

nvidia-smi -L
[ -n "${PAIR_A:-}" ] && [ -n "${PAIR_B:-}" ] || { echo "set PAIR_A and PAIR_B"; exit 2; }
specs=("$PAIR_A" "$PAIR_B"); names=(a b); pids=()
for i in 0 1; do
  gpus=$([ $i = 0 ] && echo 0,1,2,3 || echo 4,5,6,7)
  # shellcheck disable=SC2086  # word-split the VAR=value assignments on purpose
  env ${specs[$i]} CUDA_VISIBLE_DEVICES=$gpus .venv/bin/python -m experiments.references.della_muonh_qwen3_scaling \
    > logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}${PAIR_LOG_TAG:-}_${names[$i]}.log 2>&1 &
  pids+=($!)
  echo "run ${names[$i]} [${specs[$i]}] on GPUs $gpus: pid ${pids[$i]} $(date +%H:%M:%S)"
done
rc=0
for i in 0 1; do
  wait "${pids[$i]}"; r=$?
  echo "run ${names[$i]} [${specs[$i]}] exited rc=$r $(date +%H:%M:%S)"
  [ $r -ne 0 ] && rc=$r
done
exit $rc
