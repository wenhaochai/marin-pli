#!/bin/bash
#SBATCH --job-name=k3-checkup
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96   # 8-GPU jobs ask for all 96 CPUs: priority JobSize counts CPUs
#SBATCH --mem-per-gpu=64G
#SBATCH --time=TIME_SET_AFTER_SMOKE
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
#FETCHABLE: self-contained (every path below is literal; nothing comes from the submit shell)

# Check-up of Kimi K3's Block AttnRes (owner, 2026-10-19), k3_checkup.py on every GPU of the node: rank r takes every
# 8th sequence of the Paloma c4_en token file (64 x 1,024 K3 tokens, k3_checkup_tokens.py) and runs all 93 layers with
# the baseline and 12 AttnRes ablations batched on each loaded layer; k3_checkup_merge.py adds the ranks up into
# merged.json. The smoke test is a copy of this script (k3_checkup_smoke.sbatch).
set -uo pipefail
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
WT=/scratch/gpfs/GROUP/USER/project/marin-pls-auto10
PY=/scratch/gpfs/GROUP/USER/envs/k3lens/bin/python
MODEL=/scratch/gpfs/GROUP/USER/models/Kimi-K3
TOKS=/scratch/gpfs/GROUP/USER/models/k3_checkup_c4en_64x1024.npy
TEXT="Paloma c4_en validation (Marin store, Llama-3 ids decoded and re-tokenized with the K3 tokenizer, [BOS] before each document), 64 x 1024 tokens"
OUT=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/k3_checkup/$JID
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID; mkdir -p "$TMPDIR"; trap 'rm -rf -- "$TMPDIR"' EXIT
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=$TMPDIR/triton
# one process per visible GPU (a fetched run inherits its host's GPU slots in CUDA_VISIBLE_DEVICES)
IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
G=${#DEVS[@]}
nvidia-smi -L || true
echo "job $JID on $(hostname): $G GPUs (${DEVS[*]}), out $OUT"
mkdir -p "$OUT"
pids=()
for r in $(seq 0 $((G - 1))); do
  CUDA_VISIBLE_DEVICES=${DEVS[$r]} $PY $WT/scripts/della/k3_checkup.py "$MODEL" "$TOKS" "$OUT" $r $G > "$OUT/rank$r.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
tail -n 3 "$OUT"/rank*.log
$PY $WT/scripts/della/k3_checkup_merge.py "$OUT" "$TEXT" || rc=$?
exit $rc
