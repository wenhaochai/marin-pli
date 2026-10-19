#!/bin/bash
#SBATCH --job-name=k3-checkup-smoke
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-gpu=96G
#SBATCH --time=01:01:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
# Smoke test (pli-cp) of the Kimi K3 AttnRes check-up: a copy of k3_checkup.sbatch on 2 GPUs (32 sequences per rank,
# 4x the full job's 8) through layers 0-3 (K3_MAX_LAYERS=4: KDA with the dense MLP, KDA with MoE, MLA with MoE):
#   pass 1: the old k3_lens WikiText token file; the baseline next/exit/sum CE of layers 0-3 must equal k3_lens job
#           15736820's merged.json to ~1e-3;
#   pass 2: the new Paloma c4_en token file.
set -uo pipefail
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
WT=/scratch/gpfs/GROUP/USER/project/marin-pls-auto10
PY=/scratch/gpfs/GROUP/USER/envs/k3lens/bin/python
MODEL=/scratch/gpfs/GROUP/USER/models/Kimi-K3
ML=4
OLD=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/k3_lens/15736820/merged.json
BASE=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/k3_checkup/smoke_$JID
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/$JID; mkdir -p "$TMPDIR"; trap 'rm -rf -- "$TMPDIR"' EXIT
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=$TMPDIR/triton
IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
G=${#DEVS[@]}
nvidia-smi -L || true
rc=0
run_pass() {   # $1 tag, $2 token file, $3 text
  local OUT=$BASE/$1 pids=() r p
  mkdir -p "$OUT"
  for r in $(seq 0 $((G - 1))); do
    CUDA_VISIBLE_DEVICES=${DEVS[$r]} $PY $WT/scripts/della/k3_checkup.py "$MODEL" "$2" "$OUT" $r $G $ML > "$OUT/rank$r.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=$?; done
  tail -n 12 "$OUT"/rank*.log
  $PY $WT/scripts/della/k3_checkup_merge.py "$OUT" "$3" || rc=$?
}
run_pass wikitext /scratch/gpfs/GROUP/USER/models/k3_lens_tokens_64x1024.npy "WikiText-103 test (k3_lens token file), smoke"
$PY - "$BASE/wikitext/merged.json" "$OLD" <<'EOF' || rc=$?
import json, sys
new, old = (json.load(open(f)) for f in sys.argv[1:3])
worst = 0.0
for a, b in zip(new["layers"], old["layers"]):
    d = {k: a[k] - b[k] for k in ("next_ce", "exit_ce", "sum_ce", "next_acc", "exit_acc", "sum_acc")}
    worst = max(worst, *(abs(d[k]) for k in ("next_ce", "exit_ce", "sum_ce")))
    print(f"EQUIV L{a['layer']}: new next/exit/sum {a['next_ce']:.4f}/{a['exit_ce']:.4f}/{a['sum_ce']:.4f} old {b['next_ce']:.4f}/{b['exit_ce']:.4f}/{b['sum_ce']:.4f} "
          f"| max |dCE| {max(abs(d[k]) for k in ('next_ce', 'exit_ce', 'sum_ce')):.2e} | probs new {[round(p, 4) for p in a['attn_in_probs']]} old {[round(p, 4) for p in b['attn_in_probs']]}")
print(f"EQUIV worst |dCE| = {worst:.2e} -> {'PASS' if worst < 1e-3 else 'FAIL'} (tokens new {new['tokens']} old {old['tokens']})")
sys.exit(0 if worst < 1e-3 else 3)
EOF
run_pass c4en /scratch/gpfs/GROUP/USER/models/k3_checkup_c4en_64x1024.npy "Paloma c4_en validation (K3 re-tokenized), smoke"
echo "smoke exit $rc"
exit $rc
