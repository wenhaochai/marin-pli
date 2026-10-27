#!/bin/bash
#SBATCH --job-name=vo-bin-bias
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-gpu=110G
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell
# Is the 8-pass cost on rare targets a global prior error? (owner popup 2026-10-27, "A 分频段偏置校准"):
# experiments/references/vocab_bin_bias.py on the three token pairs of w506 (128K seed 1, 8K seeds 0 and 1), one per GPU;
# the bytes pair is left out (its cost is 0.002 nats/byte). The three arms take about the same time (w506: 10-12 min).
# A smoke run (4 windows per subset) on the first GPU comes first. Output logs/vocab_bin_bias/<pair>.json.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
source scripts/della/job_tmpdir.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=$PWD/lib/levanter/src:$PWD/lib/haliax/src:$PWD
S=$MARIN_PREFIX/speedrun
OUT=logs/vocab_bin_bias
mkdir -p $OUT
nvidia-smi -L
echo "START $(date +%H:%M:%S) $(git log --oneline -1 | cut -c1-80)"
# this job's own GPUs as the host set them (a guest on GPUs 4-7 must not touch 0-3)
IFS=, read -r -a GPUS <<<"${CUDA_VISIBLE_DEVICES:-0,1,2}"
PAIRS=("128k-s1 0 -s1-rerun -rep8-s1 11443" "8k-s0 8000 -v8k -v8k-rep8 15463" "8k-s1 8000 -v8k-s1 -v8k-rep8-s1 15463")
ck() { echo "$S/muonh-qwen3-300m-della4xh100$1/2026.09.13/checkpoints/step-$2"; }
arm() {   # arm <i> <out> [extra env...]
  local i=$1 out=$2; shift 2
  read -r name k u r step <<<"${PAIRS[$i]}"
  local vk=(); [ "$k" != 0 ] && vk=(VOCAB_K=$k)
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} env "${vk[@]}" "$@" SIZE=300m CKPT_U=$(ck $u $step) CKPT_R=$(ck $r $step) OUT=$out \
    timeout 40m .venv/bin/python -m experiments.references.vocab_bin_bias
}
if ! arm 0 $OUT/smoke.json MAX_WINDOWS=4 > $OUT/smoke.log 2>&1; then
  grep -v -E "Unknown leaf|[Ww]arn" $OUT/smoke.log | tail -20; echo "DONE fail=smoke $(date +%H:%M:%S)"; exit 0
fi
grep -E "^cost|^offsets" $OUT/smoke.log
pids=()
for i in 0 1 2; do
  read -r name _ <<<"${PAIRS[$i]}"
  arm $i $OUT/$name.json > $OUT/$name.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
for i in 0 1 2; do read -r name _ <<<"${PAIRS[$i]}"; echo "== $name"; grep -v -E "Unknown leaf|[Ww]arn" $OUT/$name.log | tail -5; done
echo "DONE fail=$fail $(date +%H:%M:%S)"
exit 0
