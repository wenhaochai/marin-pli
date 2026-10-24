#!/bin/bash
#SBATCH --job-name=pls-speed-probe
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls-speed/logs/slurm/%x_%j.out
# Speed work (owner 2026-10-24): one short run of a 48-layer design in the pls-speed worktree with the jax profiler on, to
# measure where a train step's time goes. Smoke-length run with its own run id (RUN_TAG); nothing else reads it.
# PROBE_ARCH (default attnres) and PROBE_EXTRA (extra VAR=value knobs) come from the job name: pls-speed-probe[-<arch>].
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls-speed
source scripts/della/pls_env.sh
module load proxy/default 2>/dev/null || true
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline VARIANT=pls
trap pls_tmp_cleanup EXIT
case ${SLURM_JOB_NAME#pls-speed-probe} in
  ""|-attnres) A="LAYERS=48 PDP=2 ARCH=attnres";;
  -preln) A="LAYERS=48 PDP=4 ARCH=preln";;
  -hc) A="LAYERS=48 PDP=4 ARCH=hc";;
  *) echo "unknown probe '$SLURM_JOB_NAME'"; exit 2;;
esac
S="SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=0 PARAM_MAP=vocab NUM_GPUS=4 $A SMOKE_STEPS=40 PROFILE=20:3 RUN_TAG=-j${SLURM_JOB_ID}speed"
echo "knobs: $S"; nvidia-smi -L
# shellcheck disable=SC2086
env $S timeout 80m "$PY" -m experiments.references.della_muonh_qwen3_scaling > logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_train.log 2>&1 || echo "rc=$?"
rid=$(ls -td logs/muonh-qwen3-*-j${SLURM_JOB_ID}speed* 2>/dev/null | head -1 || true); echo "log dir: ${rid:-none}"
out=$MARIN_PREFIX/speedrun; for d in "$out"/*-j${SLURM_JOB_ID}speed*; do [[ $d == *-j${SLURM_JOB_ID}speed* && -d $d ]] && rm -rf -- "$d"; done
for d in "$MARIN_PREFIX"/tmp/checkpoints-temp"$out"/*-j${SLURM_JOB_ID}speed*; do [[ $d == *-j${SLURM_JOB_ID}speed* && -d $d ]] && rm -rf -- "$d"; done
echo done
