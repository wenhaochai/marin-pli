#!/bin/bash
#SBATCH --job-name=trec-smoke
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=48G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/trec/logs/slurm/%x_%j.out
#FETCHABLE self-contained: every setting is in this file; needs nothing from the submitting shell
# Training re-evaluation curves (Bergsma et al. 2025, arXiv 2509.25380; owner 2026-10-26): the final checkpoints of the
# 130m depth control (12 layers, WSD, lr 0.02) and the 300m Marin-tokenizer run (cosine, lr 0.01), scored by
# experiments/references/trec_score.py on smoke: the last 12 and first 4 training batches and 4 unread batches of each checkpoint (the last ones must match the W&B train loss).
# Each output chunk is skipped when present, so a rerun continues.
set -uo pipefail
WT=/scratch/gpfs/GROUP/USER/project/marin-trec
cd $WT
source scripts/della/job_tmpdir.sh
_pp=$WT; for d in "$WT"/lib/*/src; do _pp=$_pp:$d; done; export PYTHONPATH=$_pp   # this worktree's sources, not project/marin's
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=${JOB_XLA_MEM_FRACTION:-0.9}
unset VOCAB_K VARIANT SEQ_LEN DATA_EPOCHS PASS_DATA
S=$MARIN_PREFIX/speedrun O=/scratch/gpfs/GROUP/USER/project/trec/out/smoke
nvidia-smi -L
fail=0
# run model step
for e in "muonh-qwen3-130m-della4xh100-d12 130m-d12 4958" "muonh-qwen3-300m-della4xh100-s1-rerun 300m 11443"; do
  read -r run model step <<< "$e"
  c=$(ls -d $S/$run/*/checkpoints/step-$step 2>/dev/null | head -1); [ -n "$c" ] || { echo "NO CHECKPOINT $run"; fail=1; continue; }
  echo "TREC $run $(date +%H:%M:%S)"
  CKPT=$c MODEL=$model STEPS=smoke OUT=$O/$run /scratch/gpfs/GROUP/USER/project/marin/.venv/bin/python -m experiments.references.trec_score || fail=1
done
echo "DONE fail=$fail $(date +%H:%M:%S)"
exit $fail
