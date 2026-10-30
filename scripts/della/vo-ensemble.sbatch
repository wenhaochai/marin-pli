#!/bin/bash
#SBATCH --job-name=vo-ensemble
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=110G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell
# Does ensembling close the gap between 8-pass and unique-token training? (Kim et al. 2025, arXiv 2509.14786: ensembles
# lower the loss asymptote in data-constrained pretraining.) experiments/references/vocab_ensemble_eval.py on four
# seed-0 + seed-1 pairs at 300m, one per GPU: 128K 8 passes, 128K unique, 8K 8 passes, 8K unique. The 128K pairs use the
# -nc1026 runs (new code, w392/w393/w596/w597): the old-code 128K unique seed-0 checkpoint is gone. The 8K pairs use the
# old-code runs. A smoke run (4 windows per subset) on the first GPU comes first. Output logs/vocab_ensemble/<pair>.json.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
source scripts/della/job_tmpdir.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=$PWD/lib/levanter/src:$PWD/lib/haliax/src:$PWD
export EVAL_BATCH=4   # two models per step
S=$MARIN_PREFIX/speedrun
OUT=logs/vocab_ensemble
mkdir -p $OUT
nvidia-smi -L
echo "START $(date +%H:%M:%S) $(git log --oneline -1 | cut -c1-80)"
# this job's own GPUs as the host set them (a guest on GPUs 4-7 must not touch 0-3)
IFS=, read -r -a GPUS <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# name, VOCAB_K (0 = 128K), run A suffix, run B suffix, final step
PAIRS=("128k-rep8 0 -rep8-nc1026 -rep8-s1-nc1026 11443" "128k-unique 0 -nc1026 -s1-nc1026 11443"
       "8k-rep8 8000 -v8k-rep8 -v8k-rep8-s1 15463" "8k-unique 8000 -v8k -v8k-s1 15463")
ck() { echo "$S/muonh-qwen3-300m-della4xh100$1/2026.09.13/checkpoints/step-$2"; }
arm() {   # arm <i> <out> [extra env...]
  local i=$1 out=$2; shift 2
  read -r name k a b step <<<"${PAIRS[$i]}"
  local vk=(); [ "$k" != 0 ] && vk=(VOCAB_K=$k)
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} env "${vk[@]}" "$@" SIZE=300m CKPT_A=$(ck $a $step) CKPT_B=$(ck $b $step) OUT=$out \
    timeout 45m .venv/bin/python -m experiments.references.vocab_ensemble_eval
}
for i in 0 1 2 3; do
  read -r name k a b step <<<"${PAIRS[$i]}"
  for r in $a $b; do [ -f "$(ck $r $step)/metadata.json" ] || { echo "missing checkpoint $(ck $r $step)"; echo "DONE fail=missing"; exit 0; }; done
done
if ! arm 0 $OUT/smoke.json MAX_WINDOWS=4 > $OUT/smoke.log 2>&1; then
  grep -v -E "Unknown leaf|[Ww]arn" $OUT/smoke.log | tail -20; echo "DONE fail=smoke $(date +%H:%M:%S)"; exit 0
fi
grep -E "^partial" $OUT/smoke.log
pids=()
for i in 0 1 2 3; do
  read -r name _ <<<"${PAIRS[$i]}"
  arm $i $OUT/$name.json > $OUT/$name.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
for i in 0 1 2 3; do read -r name _ <<<"${PAIRS[$i]}"; echo "== $name"; grep -v -E "Unknown leaf|[Ww]arn" $OUT/$name.log | tail -4; done
echo "DONE fail=$fail $(date +%H:%M:%S)"
exit 0
