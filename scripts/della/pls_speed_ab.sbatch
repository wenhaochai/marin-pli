#!/bin/bash
#SBATCH --job-name=pls-speed-ab
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-gpu=32G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
# Speed work (owner 2026-10-24): matched A/B of speed knobs on a real 48-layer design, two 4-GPU arms side by side on one
# node (GPUs 0-3 and 4-7), so both arms see the same node. Each arm is a smoke-length run (SMOKE_STEPS=40, its own RUN_TAG,
# W&B group muonh-qwen3-smoke). Rounds (AB_ROUNDS, from the job name pls-speed-ab-<design>):
#   det: XLA_FLAGS --xla_gpu_exclude_nondeterministic_ops=true on both arms; off vs on, for a bitwise check
#   twin: the same with knobs off on both arms (the determinism floor, used only if det is not bitwise)
#   time: default flags, off vs on, both with PROFILE=20:3, for step time, kernel times and memory
#   eval: 20 steps, then the same 640 eval sequences at the train microbatch (160 batches) vs EVAL_PDP=4 (40 batches)
# AB_ON holds the knobs of the on arm.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
module load proxy/default 2>/dev/null || true
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VARIANT=pls WANDB_PROJECT=marin-della
curl -s -o /dev/null --max-time 20 https://api.wandb.ai/healthz || export WANDB_MODE=offline
trap pls_tmp_cleanup EXIT
case ${SLURM_JOB_NAME#pls-speed-ab-} in
  preln) A="LAYERS=48 PDP=4 ARCH=preln"; ON="LEVANTER_CE_BTILED_ONEHOT=1 LOG_JAXPR=0";;
  attnres-eval) A="LAYERS=48 PDP=2 ARCH=attnres"; ON="EVAL_PDP=4"; AB_ROUNDS=${AB_ROUNDS:-eval};;
  *) echo "unknown design '$SLURM_JOB_NAME'"; exit 2;;
esac
ON=${AB_ON:-$ON}
BASE="SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=0 PARAM_MAP=vocab NUM_GPUS=4 $A SMOKE_STEPS=40"
nvidia-smi -L
arm() {  # arm <gpus> <tag> <extra knobs...>: one run, log logs/slurm/<job>_<tag>.log
  local g=$1 t=$2; shift 2
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$g env $BASE "$@" RUN_TAG=-j${SLURM_JOB_ID}$t timeout 40m "$PY" -m experiments.references.della_muonh_qwen3_scaling \
    > logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_$t.log 2>&1 || echo "arm $t rc=$?"
}
DET="XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true"
for r in ${AB_ROUNDS:-det time twin}; do
  echo "round $r $(date +%T)"
  case $r in
    det) arm 0,1,2,3 detoff $DET & arm 4,5,6,7 deton $DET $ON & wait;;
    twin) arm 0,1,2,3 detoffa $DET & arm 4,5,6,7 detoffb $DET & wait;;
    time) arm 0,1,2,3 timeoff PROFILE=20:3 & arm 4,5,6,7 timeon PROFILE=20:3 $ON & wait;;
    eval) arm 0,1,2,3 evaloff SMOKE_STEPS=20 SMOKE_EVAL_BATCHES=160 & arm 4,5,6,7 evalon SMOKE_STEPS=20 SMOKE_EVAL_BATCHES=40 $ON & wait;;
  esac
done
echo "rounds done $(date +%T)"
out=$MARIN_PREFIX/speedrun
for d in "$out"/*-j${SLURM_JOB_ID}* "$MARIN_PREFIX"/tmp/checkpoints-temp"$out"/*-j${SLURM_JOB_ID}*; do
  [[ $d == *-j${SLURM_JOB_ID}* && -d $d ]] && rm -rf -- "$d"
done
echo done
