#!/bin/bash
#SBATCH --job-name=vo-setmem-c
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=01:10:00
#SBATCH --comment=fetchable
#SBATCH --output=logs/slurm/%x_%j.out
#FETCHABLE self-contained: every setting is in this file; needs nothing from the submitting shell
# Test B extended to the setups (2026-10-25, idle GPUs on two hosts): per-position losses on the repeated subset
# (vocab_subset_losses) and on Paloma (vocab_token_losses) for remedies and their pairs, so memorisation and cost can be
# compared per setup: does a remedy lower what is memorised, or what each memorised bit costs? Outputs skipped when present.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
source scripts/della/job_tmpdir.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=${JOB_XLA_MEM_FRACTION:-0.9} SIZE=300m
unset VOCAB_K VARIANT CV_SIDE CV_K CV_OUT_REDUCE SEQ_LEN ONLY MAX_WINDOWS
S=$MARIN_PREFIX/speedrun P=muonh-qwen3-300m-della4xh100
mkdir -p logs/vocab_subset_losses logs/vocab_token_losses
fail=0
for e in "paloma:-rep8-rs:0:11443:300m-rep8-rs:1024" "paloma:-lrx2-fp2-rep8:0:11443:300m-rep8-rs:1024"; do
  read -r kind tag k step sub nw <<< "${e//:/ }"
  c=$(ls -d $S/$P$tag/*/checkpoints/step-$step 2>/dev/null | head -1); [ -n "$c" ] || { echo "NO CHECKPOINT $tag"; fail=1; continue; }
  kv=""; [ "$k" != 0 ] && kv="VOCAB_K=$k"
  if [ "$kind" = subset ]; then
    o=logs/vocab_subset_losses/$P$tag.npz; [ -s $o ] && { echo "SKIP $o"; continue; }
    echo "SUBSET $tag $(date +%H:%M:%S)"
    env $kv CKPT=$c SUBSET=$MARIN_PREFIX/materialized/$sub SUBSET_VIEW=/scratch/gpfs/GROUP/USER/tmp/subset_views/$sub MAX_WINDOWS=$nw OUT=$o .venv/bin/python -m experiments.references.vocab_subset_losses || fail=1
  else
    o=logs/vocab_token_losses/$P$tag.npz; [ -s $o ] && { echo "SKIP $o"; continue; }
    echo "PALOMA $tag $(date +%H:%M:%S)"
    env $kv CKPT=$c OUT=$o .venv/bin/python -m experiments.references.vocab_token_losses || fail=1
  fi
done
echo "DONE fail=$fail $(date +%H:%M:%S)"
exit $fail
