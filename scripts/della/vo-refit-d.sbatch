#!/bin/bash
#SBATCH --job-name=vo-refit-d
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=01:01:00
#SBATCH --comment=fetchable
#SBATCH --output=logs/slurm/%x_%j.out
#FETCHABLE self-contained: every setting is in this file; needs nothing from the submitting shell
# Split of w124 vo-refit (controller 2026-10-24: 3 idle GPUs on host 16093165): the same re-fits, one per GPU, in an order
# w124 reaches last; outputs are skipped when present on either side. Then per-position Paloma losses of the 8K LR x3 pair and the 8K LR x2 unique run, for the novel-continuation analysis of the LR raise.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
source scripts/della/job_tmpdir.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=${JOB_XLA_MEM_FRACTION:-0.9} SIZE=300m FT_BATCH=32 FT_LR=1e-3
unset VOCAB_K VARIANT CV_SIDE CV_K CV_OUT_REDUCE SEQ_LEN ONLY MAX_WINDOWS
S=$MARIN_PREFIX/speedrun P=muonh-qwen3-300m-della4xh100
fail=0
for e in "refit:-v8k-s1:8000:15463:body:540" "paloma:-lrx3-fp2-v8k:8000:15463" "paloma:-lrx3-fp2-v8k-rep8:8000:15463" "paloma:-lrx2-fp2-v8k:8000:15463"; do
  read -r kind tag k step part fts <<< "${e//:/ }"
  c=$(ls -d $S/$P$tag/*/checkpoints/step-$step 2>/dev/null | head -1); [ -n "$c" ] || { echo "NO CHECKPOINT $tag"; fail=1; continue; }
  kv=""; [ "$k" != 0 ] && kv="VOCAB_K=$k"
  if [ "$kind" = refit ]; then
    o=logs/vocab_refit/$P$tag-$part.json; [ -s $o ] && { echo "SKIP $o"; continue; }
    echo "REFIT $tag $part $(date +%H:%M:%S)"
    env $kv CKPT=$c STEPS_RUN=$step PART=$part RARE_FROM=8000 FT_STEPS=$fts OUT=$o .venv/bin/python -m experiments.references.vocab_refit || fail=1
  else
    o=logs/vocab_token_losses/$P$tag.npz; [ -s $o ] && { echo "SKIP $o"; continue; }
    echo "PALOMA $tag $(date +%H:%M:%S)"
    env $kv CKPT=$c OUT=$o .venv/bin/python -m experiments.references.vocab_token_losses || fail=1
  fi
done
echo "DONE fail=$fail $(date +%H:%M:%S)"
exit $fail
