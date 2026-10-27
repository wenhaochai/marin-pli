#!/bin/bash
#SBATCH --job-name=vo-trec2
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
# Follow-up of w438 (vo-trec.sbatch) for the page question "Which text does repetition make worse?" (owner popup
# 2026-10-26): (1) the same training re-evaluation for a second seed of 8K (seed 0) and bytes (seed 1), whose unique and
# 8-pass checkpoints both exist (128K seed 0 has no unique checkpoint), into logs/vocab_trec; (2) every sequence of the 16
# unread blocks with its per-position losses (KEEP=128, no training blocks) for all ten checkpoints, into
# logs/vocab_trec_tail, so the target-frequency deciles of fig-trec-freq rest on 1,920 sequences instead of 60. Data order
# is data_seed 42 for every seed, so the 8-pass subset is the same. Each output is skipped when present.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
source scripts/della/job_tmpdir.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=${JOB_XLA_MEM_FRACTION:-0.9} SIZE=300m
unset VOCAB_K VARIANT SEQ_LEN DATA_EPOCHS PASS_DATA N_STEPS N_TAIL KEEP
S=$MARIN_PREFIX/speedrun P=muonh-qwen3-300m-della4xh100
mkdir -p logs/vocab_trec logs/vocab_trec_tail
fail=0
ckpt() { ls -d $S/$P$1/*/checkpoints/step-$2 2>/dev/null | head -1; }
nvidia-smi -L
run() {   # tag K step outdir extra-env...
  local tag=$1 k=$2 step=$3 dir=$4; shift 4
  local c; c=$(ckpt $tag $step); [ -n "$c" ] || { echo "NO CHECKPOINT $tag"; fail=1; return; }
  local kv=""; [ "$k" != 0 ] && kv="VOCAB_K=$k"
  local o=$dir/$P$tag.npz
  [ -s $o ] && return
  echo "TREC $tag -> $dir $(date +%H:%M:%S)"
  env $kv "$@" CKPT=$c OUT=$o EVAL_BATCH=8 .venv/bin/python -m experiments.references.vocab_trec || fail=1
}
for e in "-v8k 8000 15463" "-v8k-rep8 8000 15463" "-bytes-s1 256 54247" "-bytes-rep8-s1 256 54247"; do
  run $e logs/vocab_trec
done
for e in "-s1-rerun 0 11443" "-rep8-s1 0 11443" "-v8k-s1 8000 15463" "-v8k-rep8-s1 8000 15463" "-bytes 256 54247" "-bytes-rep8 256 54247" \
         "-v8k 8000 15463" "-v8k-rep8 8000 15463" "-bytes-s1 256 54247" "-bytes-rep8-s1 256 54247"; do
  run $e logs/vocab_trec_tail N_STEPS=0 N_TAIL=16 KEEP=128
done
echo "DONE fail=$fail $(date +%H:%M:%S)"
exit $fail
