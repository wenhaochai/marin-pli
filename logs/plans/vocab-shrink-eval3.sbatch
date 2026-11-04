#!/bin/bash
#SBATCH --job-name=vocab-shrink-eval3
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=24G
#SBATCH --time=02:45:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell
# A vocabulary that shrinks as data repeats, what in path 1 does the work (owner 2026-11-04): a mixed run has one score per view, so this job
# scores the six follow-ups on all 16 Paloma subsets at 128K, 64K, 32K, 16K and 8K, each at its equal-text window. A run's last checkpoint
# is found on disk. Each GPU works through its own list, so no GPU waits for another. Reads checkpoints only; writes logs/vocab_shrink_eval/<arm>/k<K>.json.
set -uo pipefail
CODE=/scratch/gpfs/GROUP/USER/project/marin
cd /scratch/gpfs/GROUP/USER/project/marin
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
OUT=$CODE/logs/vocab_shrink_eval
fail=0
one() {   # GPU index, arm, K
  local d=$OUT/$2 c
  mkdir -p "$d"
  [ -f "$d/k$3.json" ] && return 0
  c=$(ls -d $S/$P-rep8-$2-sl4992/2026.09.13/checkpoints/step-* | sort -V | tail -1)
  CUDA_VISIBLE_DEVICES=${GPUS[$1]} CKPT=$c EVAL_K=$3 OUT=$d/k$3.json timeout 50m .venv/bin/python -m experiments.references.vocab_shrink_eval > "$d/k$3.log" 2>&1 || { echo "FAILED $2 $3"; return 1; }
}
lane() {   # GPU index, then "<arm> <K>" per argument
  local g=$1 a bad=0
  shift
  for a in "$@"; do one "$g" $a || bad=$((bad + 1)); done
  return $bad
}
T=()
for a in vsconst vsfrozen vsuniq vsgrow vs128k16k vs16k8k; do for k in 128000 64000 32000 16000 8000; do T+=("$a $k"); done; done
pids=()
for g in 0 1 2 3; do
  L=()
  for ((i = g; i < ${#T[@]}; i += 4)); do L+=("${T[$i]}"); done
  lane "$g" "${L[@]}" & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + $?)); done
grep -H "exact macro" $OUT/*/k*.log | cut -c1-160
echo "DONE fail=$fail $(date +%H:%M:%S)"
[ "$fail" -eq 0 ]
