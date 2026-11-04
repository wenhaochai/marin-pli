#!/bin/bash
#SBATCH --job-name=vocab-shrink-eval5
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=24G
#SBATCH --time=02:15:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell
# A vocabulary that shrinks as data repeats, what in path 1 does the work (owner 2026-11-04): a mixed run has one score per view, so this job
# scores the follow-ups B, C and D on all 16 Paloma subsets at 128K, 64K, 32K, 16K and 8K, each at its equal-text window. A run's last checkpoint
# is found on disk. Each GPU takes the next task from one shared list, so all GPUs end within one task of each other. Reads checkpoints only; writes logs/vocab_shrink_eval/<arm>/k<K>.json.
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
  local env=()
  case "$2" in
    *-stack) c=$(ls -d $S/$P-tied-wsd0.4-rep8-${2%-stack}-ema0.999-sl4992/2026.09.13/checkpoints/step-* | sort -V | tail -1); env=(TIE=1 EMA=1) ;;
    *) c=$(ls -d $S/$P-rep8-$2-sl4992/2026.09.13/checkpoints/step-* | sort -V | tail -1) ;;
  esac
  env "${env[@]}" CUDA_VISIBLE_DEVICES=${GPUS[$1]} CKPT=$c EVAL_K=$3 OUT=$d/k$3.json timeout 50m .venv/bin/python -m experiments.references.vocab_shrink_eval > "$d/k$3.log" 2>&1 || { echo "FAILED $2 $3"; return 1; }
}
T=()
for a in vs128k16ku vsp50 vsp50u vsp80 vsp80u vsl128 vsconst-stack vsuniq-stack; do for k in 128000 64000 32000 16000 8000; do T+=("$a $k"); done; done
T+=("vs16k8ku 16000" "vs16k8ku 8000")
NEXT=$TMPDIR/next_task
echo 0 > "$NEXT"
lane() {   # GPU index: takes the next task of the shared list until none is left
  local g=$1 i bad=0
  while :; do
    i=$(flock "$NEXT.lock" bash -c 'read -r n < "$0"; echo $((n + 1)) > "$0"; echo "$n"' "$NEXT")
    [ "$i" -lt "${#T[@]}" ] || break
    one "$g" ${T[$i]} || bad=$((bad + 1))
  done
  return $bad
}
pids=()
for g in 0 1 2 3; do
  lane "$g" & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + $?)); done
grep -H "exact macro" $OUT/*/k*.log | cut -c1-160
echo "DONE fail=$fail $(date +%H:%M:%S)"
[ "$fail" -eq 0 ]
