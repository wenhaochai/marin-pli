#!/bin/bash
#SBATCH --job-name=k3-lens
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96   # 8-GPU jobs ask for all 96 CPUs: priority JobSize counts CPUs
#SBATCH --mem-per-gpu=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
#FETCHABLE: self-contained (paths below; K3_MAX_LAYERS / K3_GPUS only for a pli-cp smoke submitted with them in the script copy)

# The logit lens of Kimi K3 (owner, 2026-10-17), k3_lens.py on every GPU of the node: rank r takes every 8th sequence of
# the token file and runs all 93 layers; k3_lens_merge.py adds the ranks up. A smoke copy of this script sets
# K3_MAX_LAYERS=4 (layers 0-3: KDA with the dense MLP, KDA with MoE, MLA with MoE) and K3_GPUS=1.
set -uo pipefail
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
WT=/scratch/gpfs/GROUP/USER/project/marin-pls-auto10
PY=/scratch/gpfs/GROUP/USER/envs/k3lens/bin/python
MODEL=/scratch/gpfs/GROUP/USER/models/Kimi-K3
TOKS=/scratch/gpfs/GROUP/USER/models/k3_lens_tokens_64x1024.npy
OUT=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/k3_lens/$JID
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID; mkdir -p "$TMPDIR"; trap 'rm -rf -- "$TMPDIR"' EXIT
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=$TMPDIR/triton
G=${K3_GPUS:-8}; ML=${K3_MAX_LAYERS:-}
nvidia-smi -L || true
mkdir -p "$OUT"
pids=()
for r in $(seq 0 $((G - 1))); do
  CUDA_VISIBLE_DEVICES=$r $PY $WT/scripts/della/k3_lens.py "$MODEL" "$TOKS" "$OUT" $r $G $ML > "$OUT/rank$r.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
tail -n 3 "$OUT"/rank*.log
$PY $WT/scripts/della/k3_lens_merge.py "$OUT" || rc=$?
exit $rc
