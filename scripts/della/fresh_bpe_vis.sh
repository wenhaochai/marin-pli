#!/bin/bash
# The fresh 8K BPE control (owner 2026-10-31, "检查有没有实验遗漏我们补一下"): train it on a uniform sample of fineweb-edu,
# convert the training cache and the 16 Paloma caches through the text, and write its exact bpb tables; on a vis node:
#     ssh della-vis1 'setsid nohup bash scripts/della/fresh_bpe_vis.sh > logs/slurm/bpe8k-vis.out 2>&1 < /dev/null &'
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/bpe8k-vis
mkdir -p "$TMPDIR"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RAYON_NUM_THREADS=7 TOKENIZERS_PARALLELISM=true
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
OUT=$MARIN_PREFIX/tokenizers/marin-small/bpe8k
echo "== train $(date +%T)"
[ -f "$OUT/tokenizer.json" ] || nice -n 19 .venv/bin/python -m experiments.references.fresh_bpe_tokenizer train 8000 400000 "$OUT"
echo "== convert $(date +%T)"
VOCAB_SET=bpe8k nice -n 19 .venv/bin/python -m experiments.references.convert_small_vocab_cache
echo "== tables $(date +%T)"
nice -n 19 .venv/bin/python /scratch/gpfs/GROUP/USER/tmp_plots/vocab/bpb_exact_tables_extra.py
rm -rf "$TMPDIR"
echo "done $(date +%T)"
