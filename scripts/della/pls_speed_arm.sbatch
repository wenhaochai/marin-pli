#!/bin/bash
#SBATCH --job-name=pls-speed-arm
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
#FETCHABLE: self-contained (design, knobs and rounds come from the job name, nothing from the submit shell)
# Speed work (owner 2026-10-25: ship every speed item, test jobs go through wq with --prio 10, not pli-cp). One arm of
# an A/B on 4 GPUs: a smoke-length run of a 48-layer design per round, each with its own run id (RUN_TAG -j<id><round>,
# W&B group muonh-qwen3-smoke). Two arms with the same design and different knobs are compared offline from W&B.
# Job name: pls-speed-arm-<design>-<knobs>, design preln | hc | attnres, knobs:
#   base   today's programs
#   a1     LEVANTER_CE_BTILED_ONEHOT=1 (label term as a select in the b-tiled CE backward)
#   gif    LEVANTER_CE_GRAD_IN_FWD=1 (b-tiled CE gradients computed in the forward, one logits GEMM per block)
#   a1gif  both
# Rounds: det (XLA_FLAGS --xla_gpu_exclude_nondeterministic_ops=true, 40 steps: bitwise comparison between arms) for
# preln; time (default flags, 40 steps, PROFILE=20:3: step time, kernel times, memory) for every design.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
module load proxy/default 2>/dev/null || true
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VARIANT=pls WANDB_PROJECT=marin-della
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 LOG_JAXPR=0
curl -s -o /dev/null --max-time 20 https://api.wandb.ai/healthz || export WANDB_MODE=offline
trap pls_tmp_cleanup EXIT
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
NAME=${FETCHED_JOB_NAME:-$SLURM_JOB_NAME}
spec=${NAME#pls-speed-arm-}; design=${spec%%-*}; knobs=${spec#*-}
case $design in
  preln) A="LAYERS=48 PDP=4 ARCH=preln"; ROUNDS="det time";;
  hc) A="LAYERS=48 PDP=4 ARCH=hc"; ROUNDS="time";;
  attnres) A="LAYERS=48 PDP=2 ARCH=attnres"; ROUNDS="time";;
  *) echo "unknown design in '$NAME'"; exit 2;;
esac
case $knobs in
  base) ON="";;
  a1) ON="LEVANTER_CE_BTILED_ONEHOT=1";;
  gif) ON="LEVANTER_CE_GRAD_IN_FWD=1";;
  a1gif) ON="LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1";;
  *) echo "unknown knobs in '$NAME'"; exit 2;;
esac
BASE="SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=0 PARAM_MAP=vocab NUM_GPUS=4 $A SMOKE_STEPS=40"
echo "arm $NAME: design $design, knobs [$ON], rounds $ROUNDS, GPUs ${CUDA_VISIBLE_DEVICES:-all}"; nvidia-smi -L || true
for r in $ROUNDS; do
  echo "round $r $(date +%T)"
  extra=; [ "$r" = det ] && extra="XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true"; [ "$r" = time ] && extra="PROFILE=20:3"
  # shellcheck disable=SC2086
  env $BASE $ON $extra RUN_TAG=-j${JID}$r timeout 40m "$PY" -m experiments.references.della_muonh_qwen3_scaling \
    > logs/slurm/${NAME}_${JID}_$r.log 2>&1 || echo "round $r rc=$?"
  grep -E "First train step completed|Saving|peak|Traceback|Error" logs/slurm/${NAME}_${JID}_$r.log | tail -5 || true
done
echo "rounds done $(date +%T)"
out=$MARIN_PREFIX/speedrun
for d in "$out"/*-j${JID}det* "$out"/*-j${JID}time* "$MARIN_PREFIX"/tmp/checkpoints-temp"$out"/*-j${JID}det* "$MARIN_PREFIX"/tmp/checkpoints-temp"$out"/*-j${JID}time*; do
  [[ -n $JID && ( $d == *-j${JID}det* || $d == *-j${JID}time* ) && -d $d ]] && rm -rf -- "$d"
done
echo done
