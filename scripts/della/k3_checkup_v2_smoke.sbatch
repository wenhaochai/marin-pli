#!/bin/bash
#SBATCH --job-name=k3-checkup-v2-smoke
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-gpu=64G
#SBATCH --time=01:01:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
# Smoke test (pli-cp) of the K3 AttnRes check-up v2: a copy of k3_checkup_v2.sbatch on 8 GPUs, K3_MAX_LAYERS=6
# (layers 0-5), node loader + /dev/shm, GPU utilization sampled every 10 s:
#   pass 1: the old k3_lens WikiText token file (8 sequences per rank); baseline next/exit/sum CE of layers 0-3 must
#           equal k3_lens job 15736820's merged.json to 1e-3 (else exit 3);
#   pass 2: the full job's c4_en file, 128 x 1024 (16 sequences per rank): per-layer read/copy/run times and GPU use.
set -uo pipefail
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
WT=/scratch/gpfs/GROUP/USER/project/marin-pls-auto10
PY=/scratch/gpfs/GROUP/USER/envs/k3lens/bin/python
MODEL=/scratch/gpfs/GROUP/USER/models/Kimi-K3
ML=6
OLD=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/k3_lens/15736820/merged.json
BASE=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/k3_checkup/v2_smoke_$JID
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID; mkdir -p "$TMPDIR"
bg=()
cleanup() { for p in "${bg[@]}"; do kill "$p" 2>/dev/null; done; rm -rf -- "$TMPDIR"; [[ $JID =~ ^[0-9]+$ ]] && rm -rf -- /dev/shm/k3ck_${JID}_*; return 0; }
trap cleanup EXIT
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=$TMPDIR/triton
IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
G=${#DEVS[@]}
nvidia-smi -L || true
echo "job $JID on $(hostname): $G GPUs (${DEVS[*]})"
rc=0
run_pass() {   # $1 tag, $2 token file, $3 text
  local OUT=$BASE/$1 SHM=/dev/shm/k3ck_${JID}_$1 pids=() r p ldr smi t0=$SECONDS
  mkdir -p "$OUT" "$SHM"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -l 10 > "$OUT/gpu_util.csv" 2>/dev/null &
  smi=$!; bg+=($smi)
  $PY $WT/scripts/della/k3_checkup_loader.py "$MODEL" "$SHM" $G $ML 3 16 > "$OUT/loader.log" 2>&1 &
  ldr=$!; bg+=($ldr)
  for r in $(seq 0 $((G - 1))); do
    K3_SHM=$SHM CUDA_VISIBLE_DEVICES=${DEVS[$r]} $PY $WT/scripts/della/k3_checkup_v2.py "$MODEL" "$2" "$OUT" $r $G $ML > "$OUT/rank$r.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=$?; done
  sleep 3; kill $ldr $smi 2>/dev/null
  rm -rf -- "$SHM"
  echo "== pass $1: $((SECONDS - t0)) s"
  cat "$OUT/loader.log"
  grep -h "layer [0-9]*:" "$OUT"/rank0.log "$OUT"/rank7.log
  grep -h "final\|done in\|Error\|Traceback" "$OUT"/rank*.log | head -20
  awk -F', ' '{s+=$3; n++; if ($3 > 0) b++} END {if (n) printf "GPU utilization: mean %.1f%%, samples with util>0 %.0f%%, %d samples (10 s, all GPUs)\n", s/n, 100*b/n, n}' "$OUT/gpu_util.csv"
  $PY $WT/scripts/della/k3_checkup_merge.py "$OUT" "$3" || rc=$?
}
run_pass wikitext /scratch/gpfs/GROUP/USER/models/k3_lens_tokens_64x1024.npy "WikiText-103 test (k3_lens token file), smoke"
$PY - "$BASE/wikitext/merged.json" "$OLD" <<'EOF' || rc=$?
import json, sys
new, old = (json.load(open(f)) for f in sys.argv[1:3])
worst = 0.0
for a, b in list(zip(new["layers"], old["layers"]))[:4]:
    d = max(abs(a[k] - b[k]) for k in ("next_ce", "exit_ce", "sum_ce"))
    worst = max(worst, d)
    print(f"EQUIV L{a['layer']}: new next/exit/sum {a['next_ce']:.4f}/{a['exit_ce']:.4f}/{a['sum_ce']:.4f} old {b['next_ce']:.4f}/{b['exit_ce']:.4f}/{b['sum_ce']:.4f} | max |dCE| {d:.2e}")
print(f"EQUIV worst |dCE| = {worst:.2e} -> {'PASS' if worst < 1e-3 else 'FAIL'} (tokens new {new['tokens']} old {old['tokens']})")
sys.exit(0 if worst < 1e-3 else 3)
EOF
run_pass c4en /scratch/gpfs/GROUP/USER/models/k3_checkup_c4en_128x1024.npy "Paloma c4_en validation (K3 re-tokenized), 128 x 1024, smoke"
echo "smoke exit $rc"
exit $rc
