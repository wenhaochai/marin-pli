#!/bin/bash
# Truncations of the Marin tokenizer to K tokens (owner 2026-10-28: 1K; 2026-10-30: "还有补一下2k 4k tokenizer的结果"): build,
# validate on sampled documents, then convert fineweb-edu-10B and the 16 Paloma caches, on a vis node (nice 19).
#     bash scripts/della/convert_vk_vis.sh 2000 4000
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/vk-convert-vis
mkdir -p "$TMPDIR"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
for K in "$@"; do
OUT=$MARIN_PREFIX/tokenizers/marin-small/v$K
echo "== build $(date +%H:%M:%S)"
[ -f "$OUT/tokenizer.json" ] || nice -n 19 .venv/bin/python -m experiments.references.small_vocab_tokenizer build $K "$OUT"
echo "== validate $(date +%H:%M:%S)"
CKPT=unused nice -n 19 .venv/bin/python -m experiments.references.small_vocab_tokenizer validate $K "$OUT" 300
echo "== convert $(date +%H:%M:%S)"
K=$K nice -n 19 .venv/bin/python -m experiments.references.convert_small_vocab_cache > logs/slurm/v$K-convert-vis.out 2>&1
tail -3 logs/slurm/v$K-convert-vis.out
done
rm -rf "$TMPDIR"
echo "done $(date +%H:%M:%S)"
