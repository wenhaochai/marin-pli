#!/bin/bash
#SBATCH --job-name=pls-probe
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=24G
#SBATCH --time=01:01:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out

# Diagnostic of VARIANT=pls with PLS_PROBE=1: the full run's schedule (no SMOKE_STEPS / TOTAL_STEPS, so the LR and data
# order match the real runs step for step) with per-layer gradient alignment and in-context scores logged every step,
# stopped after PROBE_MIN minutes (the first few hundred steps). Knobs: SIZE (default 130m), PLS_W, PLS_SEP, PLS_DETACH,
# PLS_OFF, PROBE_MIN. The run id carries -j<jobid>, so it never resumes another run; its output and checkpoints are
# deleted afterwards (the metrics live in W&B).
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
unset SMOKE_STEPS TOTAL_STEPS INIT_FROM CPT_TAG TIE PRECISION EMA_BETA DEVICE_TAG DRY_RUN

module load proxy/default  # wandb only
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export VARIANT=pls PLS_PROBE=1 RUN_TAG=-j${SLURM_JOB_ID} SIZE=${SIZE:-130m}
PROBE_MIN=${PROBE_MIN:-40}
echo "pls probe knobs: SIZE=$SIZE PLS_W=${PLS_W:-1} PLS_SEP=${PLS_SEP:-0} PLS_DETACH=${PLS_DETACH:-0} PLS_OFF=${PLS_OFF:-} PROBE_MIN=$PROBE_MIN"
nvidia-smi -L

out=$(DRY_RUN=1 $PY -m experiments.references.della_muonh_qwen3_scaling 2>/dev/null | sed -n 's/^output: //p')
run_id=$(basename "$(dirname "$out")")
case "$run_id" in
  *-probe-j${SLURM_JOB_ID}) ;;
  *) echo "bad run id '$run_id' from dry run output '$out'"; exit 1 ;;
esac
echo "PROBE $run_id START $(date +%H:%M:%S)"
timeout --signal=TERM ${PROBE_MIN}m $PY -m experiments.references.della_muonh_qwen3_scaling
rc=$?
echo "PROBE $run_id END rc=$rc $(date +%H:%M:%S) (124 = stopped by the time budget, expected)"
rm -rf "$MARIN_PREFIX/speedrun/$run_id"
find "$MARIN_PREFIX/tmp/checkpoints-temp" -maxdepth 4 -type d -name "*${run_id}*" -prune -exec rm -rf {} + 2>/dev/null
echo "PROBE DONE $(date +%H:%M:%S)"
