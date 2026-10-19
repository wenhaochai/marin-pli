#!/bin/bash
#SBATCH --job-name=k3-checkup-v3
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96   # 8-GPU jobs ask for all 96 CPUs: priority JobSize counts CPUs
#SBATCH --mem-per-gpu=64G
#SBATCH --time=01:01:00   # v2 job 15844650 ran 14 min with 13 variants; v3 has 8
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
#FETCHABLE: self-contained (every path below is literal; nothing comes from the submit shell)

# K3 check-up v3 (no_embedding split: head / reads / reads from layer L; owner standing approval 2026-10-19), pipeline of v2: k3_checkup_loader.py reads each layer's shard once per
# node into /dev/shm (striped parallel GPFS reads, up to 3 layers ahead); k3_checkup_v3.py runs on every GPU (rank r
# takes every 8th of 128 c4_en sequences of 1,024 K3 tokens = 16 per rank) with the baseline and 12 AttnRes ablations
# batched on each loaded layer; k3_checkup_v3_merge.py writes merged.json. nvidia-smi samples GPU utilization every 2 s
# into gpu_util.csv. The smoke test is a copy of this script (k3_checkup_v3_smoke.sbatch).
set -uo pipefail
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
WT=/scratch/gpfs/GROUP/USER/project/marin-pls-auto10
PY=/scratch/gpfs/GROUP/USER/envs/k3lens/bin/python
MODEL=/scratch/gpfs/GROUP/USER/models/Kimi-K3
TOKS=/scratch/gpfs/GROUP/USER/models/k3_checkup_c4en_128x1024.npy
TEXT="Paloma c4_en validation (Marin store, Llama-3 ids decoded and re-tokenized with the K3 tokenizer, [BOS] before each document), 128 x 1024 tokens"
OUT=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/k3_checkup/v3_$JID
NL=93
SHM=/dev/shm/k3ck_$JID
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID; mkdir -p "$TMPDIR"
bg=()
cleanup() { for p in "${bg[@]}"; do kill "$p" 2>/dev/null; done; rm -rf -- "$TMPDIR"; [[ $JID =~ ^[0-9]+$ ]] && rm -rf -- "/dev/shm/k3ck_$JID"; return 0; }
trap cleanup EXIT
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=$TMPDIR/triton
IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"   # a fetched run inherits its host's GPU slots
G=${#DEVS[@]}
nvidia-smi -L || true
echo "job $JID on $(hostname): $G GPUs (${DEVS[*]}), out $OUT, shm $SHM"
mkdir -p "$OUT" "$SHM"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 2000 > "$OUT/gpu_util.csv" 2>/dev/null &
bg+=($!)
$PY $WT/scripts/della/k3_checkup_loader.py "$MODEL" "$SHM" $G $NL 3 16 > "$OUT/loader.log" 2>&1 &
bg+=($!)
pids=()
for r in $(seq 0 $((G - 1))); do
  K3_SHM=$SHM CUDA_VISIBLE_DEVICES=${DEVS[$r]} $PY $WT/scripts/della/k3_checkup_v3.py "$MODEL" "$TOKS" "$OUT" $r $G > "$OUT/rank$r.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
tail -n 3 "$OUT"/rank*.log "$OUT/loader.log"
$PY $WT/scripts/della/k3_checkup_util.py "$OUT" || true
$PY $WT/scripts/della/k3_checkup_v3_merge.py "$OUT" "$TEXT" || rc=$?
exit $rc
