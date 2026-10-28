#!/bin/bash
# The 1K truncation of the Marin tokenizer (owner 2026-10-28: "试一下1k tokenizer是多少"; popup "MuonH 和 Muon+WD 都跑"):
# build it, validate it on sampled documents, then convert fineweb-edu-10B and the 16 Paloma caches, on a vis node (nice 19).
#     bash scripts/della/convert_v1000_vis.sh
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/v1000-convert-vis
mkdir -p "$TMPDIR"
export JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
OUT=$MARIN_PREFIX/tokenizers/marin-small/v1000
echo "== build $(date +%H:%M:%S)"
[ -f "$OUT/tokenizer.json" ] || nice -n 19 .venv/bin/python -m experiments.references.small_vocab_tokenizer build 1000 "$OUT"
echo "== validate $(date +%H:%M:%S)"
CKPT=unused nice -n 19 .venv/bin/python -m experiments.references.small_vocab_tokenizer validate 1000 "$OUT" 300
echo "== convert $(date +%H:%M:%S)"
K=1000 nice -n 19 .venv/bin/python -m experiments.references.convert_small_vocab_cache > logs/slurm/v1000-convert-vis.out 2>&1
tail -3 logs/slurm/v1000-convert-vis.out
rm -rf "$TMPDIR"
echo "done $(date +%H:%M:%S)"
