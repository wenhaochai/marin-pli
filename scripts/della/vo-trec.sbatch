#!/bin/bash
#SBATCH --job-name=vo-trec
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=48G
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: every setting is in this file; needs nothing from the submitting shell
# Owner 2026-10-26 (after Bergsma et al. 2025, arXiv 2509.25380): training re-evaluation curves. For each tokenizer pair
# (unique-token run, 8-pass run; seed 1 at 128K and 8K, whose seed-0 unique checkpoints are gone), the final checkpoint
# scored on the unique run's training batches at 192 steps spread over its steps (the 8-pass run read the first 1/8 of
# them 8 times, the rest never) and on 16 batches after them that neither run read (experiments/references/vocab_trec.py).
# Each output is skipped when present, so a rerun continues.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
source scripts/della/job_tmpdir.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=${JOB_XLA_MEM_FRACTION:-0.9} SIZE=300m
unset VOCAB_K VARIANT SEQ_LEN DATA_EPOCHS PASS_DATA
S=$MARIN_PREFIX/speedrun P=muonh-qwen3-300m-della4xh100
mkdir -p logs/vocab_trec
fail=0
ckpt() { ls -d $S/$P$1/*/checkpoints/step-$2 2>/dev/null | head -1; }
nvidia-smi -L
# tag K step
for e in "-s1-rerun 0 11443" "-rep8-s1 0 11443" "-v8k-s1 8000 15463" "-v8k-rep8-s1 8000 15463" "-bytes 256 54247" "-bytes-rep8 256 54247"; do
  read -r tag k step <<< "$e"
  c=$(ckpt $tag $step); [ -n "$c" ] || { echo "NO CHECKPOINT $tag"; fail=1; continue; }
  kv=""; [ "$k" != 0 ] && kv="VOCAB_K=$k"
  o=logs/vocab_trec/$P$tag.npz
  [ -s $o ] && continue
  echo "TREC $tag $(date +%H:%M:%S)"
  env $kv CKPT=$c OUT=$o EVAL_BATCH=8 .venv/bin/python -m experiments.references.vocab_trec || fail=1
done
echo "DONE fail=$fail $(date +%H:%M:%S)"
exit $fail
