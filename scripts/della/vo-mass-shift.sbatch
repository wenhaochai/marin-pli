#!/bin/bash
#SBATCH --job-name=vo-mass-shift
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=96G
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell
# Where the 8-pass model moves probability mass on Paloma (owner popup 2026-10-27, "A 质量去向诊断"):
# experiments/references/vocab_mass_shift.py on four unique / 8-pass checkpoint pairs at 300m, one per GPU: 128K seed 1
# (the seed-0 unique checkpoint is gone), 8K seeds 0 and 1, bytes seed 0. Output logs/vocab_mass_shift/<pair>.npz.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
source scripts/della/job_tmpdir.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=$PWD/lib/levanter/src:$PWD/lib/haliax/src:$PWD
S=$MARIN_PREFIX/speedrun
OUT=logs/vocab_mass_shift
mkdir -p $OUT
nvidia-smi -L
echo "START $(date +%H:%M:%S) $(git log --oneline -1 | cut -c1-80)"
# this job's own GPUs as the host set them (a guest on GPUs 4-7 must not touch 0-3)
IFS=, read -r -a GPUS <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PAIRS=("128k-s1 0 -s1-rerun -rep8-s1 11443" "8k-s0 8000 -v8k -v8k-rep8 15463" "8k-s1 8000 -v8k-s1 -v8k-rep8-s1 15463" "bytes-s0 256 -bytes -bytes-rep8 54247")
pids=()
for i in 0 1 2 3; do
  read -r name k u r step <<<"${PAIRS[$i]}"
  ck() { echo "$S/muonh-qwen3-300m-della4xh100$1/2026.09.13/checkpoints/step-$step"; }
  vk=(); [ "$k" != 0 ] && vk=(VOCAB_K=$k)
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} env "${vk[@]}" SIZE=300m CKPT_U=$(ck $u) CKPT_R=$(ck $r) OUT=$OUT/$name.npz \
    timeout 40m .venv/bin/python -m experiments.references.vocab_mass_shift > $OUT/$name.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
for i in 0 1 2 3; do read -r name _ <<<"${PAIRS[$i]}"; echo "== $name"; grep -v -E "Unknown leaf|[Ww]arn" $OUT/$name.log | tail -4; done
echo "DONE fail=$fail $(date +%H:%M:%S)"
exit 0
