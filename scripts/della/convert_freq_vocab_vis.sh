#!/bin/bash
# Converts fineweb-edu-10B and the 16 Paloma validation caches to each frequency set given (owner popup 2026-10-28,
# "阈值扫描 f100 + f10000"), one after another on a vis node (nice 19): the CPU queue held f1000's job ~16 h (16341922).
#     bash scripts/della/convert_freq_vocab_vis.sh f10000 f100
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
export TMPDIR=/scratch/gpfs/GROUP/USER/tmp/job/fv-convert-vis
mkdir -p "$TMPDIR"
export JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
for s in "$@"; do
  echo "== $s $(date +%H:%M:%S)"
  VOCAB_SET=$s nice -n 19 .venv/bin/python -m experiments.references.convert_small_vocab_cache > logs/slurm/fv-convert-vis-$s.out 2>&1
  tail -1 logs/slurm/fv-convert-vis-$s.out
done
rm -rf "$TMPDIR"
echo "done $(date +%H:%M:%S)"
