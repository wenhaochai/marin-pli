#!/bin/bash
#SBATCH --job-name=vo-subset-mem
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=02:30:00
#SBATCH --comment=fetchable
#SBATCH --output=logs/slurm/%x_%j.out
#FETCHABLE self-contained: every setting is in this file; needs nothing from the submitting shell
# Owner popup 2026-10-24 (test B, after the temperature test): how much each tokenizer memorises its repeated text, and of
# which tokens. For every pair (unique-token run, 8-pass run) at 300m: per-position losses on the first windows of the
# repeated subset (vocab_subset_losses, about the same text for every tokenizer: 1024 windows at 128K, scaled by the
# tokens per byte) and, where missing, per-position Paloma losses (vocab_token_losses) for the 64K, 32K and 16K pairs.
# Each output is skipped when present, so a rerun continues.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
source scripts/della/job_tmpdir.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=${JOB_XLA_MEM_FRACTION:-0.9} SIZE=300m
unset VOCAB_K VARIANT CV_SIDE CV_K CV_OUT_REDUCE SEQ_LEN ONLY MAX_WINDOWS
S=$MARIN_PREFIX/speedrun P=muonh-qwen3-300m-della4xh100
mkdir -p logs/vocab_subset_losses logs/vocab_token_losses
fail=0
ckpt() { ls -d $S/$P$1/*/checkpoints/step-$2 2>/dev/null | head -1; }
# tag K step subset-dir windows
for e in "-s1-rerun 0 11443 300m-rep8-rs 1024" "-rep8-s1 0 11443 300m-rep8-rs 1024" \
         "-v64k 64000 11801 300m-v64k-rep8-rs 1056" "-v64k-rep8 64000 11801 300m-v64k-rep8-rs 1056" \
         "-v32k 32000 12616 300m-v32k-rep8-rs 1129" "-v32k-rep8 32000 12616 300m-v32k-rep8-rs 1129" \
         "-v16k 16000 13781 300m-v16k-rep8-rs 1233" "-v16k-rep8 16000 13781 300m-v16k-rep8-rs 1233" \
         "-v8k-s1 8000 15463 300m-v8k-rep8-rs 1384" "-v8k-rep8-s1 8000 15463 300m-v8k-rep8-rs 1384" \
         "-bytes 256 54247 300m-bytes-rep8-rs 4855" "-bytes-rep8 256 54247 300m-bytes-rep8-rs 4855"; do
  read -r tag k step sub nw <<< "$e"
  c=$(ckpt $tag $step); [ -n "$c" ] || { echo "NO CHECKPOINT $tag"; fail=1; continue; }
  kv=""; [ "$k" != 0 ] && kv="VOCAB_K=$k"
  o=logs/vocab_subset_losses/$P$tag.npz
  if [ ! -s $o ]; then
    echo "SUBSET $tag $(date +%H:%M:%S)"
    env $kv CKPT=$c SUBSET=$MARIN_PREFIX/materialized/$sub SUBSET_VIEW=/scratch/gpfs/GROUP/USER/tmp/subset_views/$sub MAX_WINDOWS=$nw OUT=$o \
      .venv/bin/python -m experiments.references.vocab_subset_losses || fail=1
  fi
  o=logs/vocab_token_losses/$P$tag.npz
  if [ ! -s $o ]; then
    echo "PALOMA $tag $(date +%H:%M:%S)"
    env $kv CKPT=$c OUT=$o .venv/bin/python -m experiments.references.vocab_token_losses || fail=1
  fi
done
echo "DONE fail=$fail $(date +%H:%M:%S)"
exit $fail
