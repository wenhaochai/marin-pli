#!/bin/bash
#SBATCH --job-name=vo-refit
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=02:45:00
#SBATCH --comment=fetchable
#SBATCH --output=logs/slurm/%x_%j.out
#FETCHABLE self-contained: every setting is in this file; needs nothing from the submitting shell
# Owner popup 2026-10-24 (test A redesigned: re-fit one part on fresh text): where does the cost of 8 passes live? For the
# seed-1 pairs of 128K and 8K, each final checkpoint re-fits only its token rows, only its rare rows (128K: ids >= 8000),
# or only its body, on fineweb-edu sequences neither run saw (vocab_refit.py), then scores Paloma. The rows' share of the
# cost = R's gain minus U's gain. Same text for both tokenizers: 400 steps of 32 sequences at 128K, 540 at 8K.
# Each output is skipped when present, so a rerun continues.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
source scripts/della/job_tmpdir.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=${JOB_XLA_MEM_FRACTION:-0.9} SIZE=300m FT_BATCH=32 FT_LR=1e-3
unset VOCAB_K VARIANT CV_SIDE CV_K CV_OUT_REDUCE SEQ_LEN ONLY MAX_WINDOWS
S=$MARIN_PREFIX/speedrun P=muonh-qwen3-300m-della4xh100
mkdir -p logs/vocab_refit
fail=0
# tag K step(=unique-token run steps) parts ft_steps
for e in "-s1-rerun 0 11443 rows,rare,body 400" "-rep8-s1 0 11443 rows,rare,body 400" \
         "-v8k-s1 8000 15463 rows,body 540" "-v8k-rep8-s1 8000 15463 rows,body 540"; do
  read -r tag k step parts fts <<< "$e"
  c=$(ls -d $S/$P$tag/*/checkpoints/step-$step 2>/dev/null | head -1); [ -n "$c" ] || { echo "NO CHECKPOINT $tag"; fail=1; continue; }
  kv=""; [ "$k" != 0 ] && kv="VOCAB_K=$k"
  for part in ${parts//,/ }; do
    o=logs/vocab_refit/$P$tag-$part.json
    [ -s $o ] && { echo "SKIP $o"; continue; }
    echo "REFIT $tag $part $(date +%H:%M:%S)"
    env $kv CKPT=$c STEPS_RUN=$step PART=$part RARE_FROM=8000 FT_STEPS=$fts OUT=$o .venv/bin/python -m experiments.references.vocab_refit || fail=1
  done
done
echo "DONE fail=$fail $(date +%H:%M:%S)"
exit $fail
