#!/bin/bash
#SBATCH --job-name=vs-token-losses
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=24G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell
# A vocabulary that shrinks as data repeats (owner 2026-11-04): the mix at no shift lowers the 128K regret from 0.080 to 0.024 when a window redraws its view
# each pass, and not when it keeps its view (w1086). Per-position Paloma losses at 128K of the mixed runs, to read the regret by the target token's count
# against the plain 128K pair (vs_token_costs.py). One run per GPU. Reads checkpoints only; writes logs/vocab_token_losses/<run>.npz.
set -uo pipefail
CODE=/scratch/gpfs/GROUP/USER/project/marin
cd $CODE
source scripts/della/job_tmpdir.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=$CODE:$CODE/lib/levanter/src:$CODE/lib/haliax/src
S=$MARIN_PREFIX/speedrun
P=muonh-qwen3-300m-della4xh100
nvidia-smi -L
echo "START $(date +%H:%M:%S) code $(git -C $CODE log --oneline -1 | cut -c1-60)"
IFS=, read -r -a GPUS <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
unset VOCAB_K VARIANT CV_SIDE CV_K CV_OUT_REDUCE SEQ_LEN SIZE
fail=0
pids=()
i=0
for a in vsconst vsuniq vsfrozen vs128k16k; do
  run=$P-rep8-$a-sl4992
  c=$(ls -d $S/$run/2026.09.13/checkpoints/step-* | sort -V | tail -1)
  ( CUDA_VISIBLE_DEVICES=${GPUS[$i]} SIZE=300m CKPT=$c OUT=logs/vocab_token_losses/$run.npz timeout 40m .venv/bin/python -m experiments.references.vocab_token_losses > logs/vocab_token_losses/$run.log 2>&1 ) &
  pids+=($!)
  i=$((i + 1))
done
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "DONE fail=$fail $(date +%H:%M:%S)"
[ "$fail" -eq 0 ]
