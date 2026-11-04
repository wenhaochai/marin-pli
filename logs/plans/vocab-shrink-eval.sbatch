#!/bin/bash
#SBATCH --job-name=vocab-shrink-eval
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
# A vocabulary that shrinks as data repeats, path 1 (owner 2026-11-04): a mixed run has one score per view, so this job scores
# both mixed runs on all 16 Paloma subsets at 128K, 64K, 32K, 16K and 8K, each at its equal-text window. The plain 128K 8-pass
# checkpoint at 128K checks the script: it must give 1.4667. Reads checkpoints only; writes logs/vocab_shrink_eval/<arm>/k<K>.json.
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
one() {   # GPU index, checkpoint, K, tag
  local d=$OUT/$4
  mkdir -p "$d"
  [ -f "$d/k$3.json" ] && return 0
  CUDA_VISIBLE_DEVICES=${GPUS[$1]} CKPT=$2 EVAL_K=$3 OUT=$d/k$3.json timeout 50m .venv/bin/python -m experiments.references.vocab_shrink_eval > "$d/k$3.log" 2>&1
}
round() {   # "<gpu> <ckpt> <K> <tag>" per argument, side by side
  local pids=() a p
  for a in "$@"; do one $a & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
}
M8=$S/$P-rep8-vsmix8-sl4992/2026.09.13/checkpoints/step-10883
M4=$S/$P-rep8-vsmix4-sl4992/2026.09.13/checkpoints/step-10372
round "0 $M8 128000 vsmix8" "1 $M8 64000 vsmix8" "2 $M8 32000 vsmix8" "3 $M8 16000 vsmix8"
round "0 $M4 128000 vsmix4" "1 $M4 64000 vsmix4" "2 $M4 32000 vsmix4" "3 $M4 16000 vsmix4"
round "0 $M8 8000 vsmix8" "1 $M4 8000 vsmix4" "2 $S/$P-rep8/2026.09.13/checkpoints/step-11443 128000 plain128k" "3 $S/$P-rep8/2026.09.13/checkpoints/step-11443 64000 plain128k"
grep -H "exact macro" $OUT/*/k*.log | cut -c1-160
echo "DONE fail=$fail $(date +%H:%M:%S)"
[ "$fail" -eq 0 ]
