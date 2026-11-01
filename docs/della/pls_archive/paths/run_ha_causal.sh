#!/bin/bash
# 2026-10-31: stream-collapse (mean) and stream-difference shuffle ablations (scripts/della/pls_hidden_ablate.py) on the
# causal-test checkpoints w711-w714 (arch3-workspace-causal), as run_ha.sh / run_has.sh did for the earlier HC / mHC pairs.
# Usage (vis node): bash run_ha_causal.sh TAG...   e.g. -sep-bbfrozen-amhc-mhcid-mhcrwlin-fa4thd-vs-d48
cd /scratch/gpfs/GROUP/USER/project/marin-pls
PY=/scratch/gpfs/GROUP/USER/project/marin-pls/.venv/bin/python
O=/scratch/gpfs/GROUP/USER/tmp/pls_paths
S=/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1
for t in "$@"; do
  c=$(ls -d $S$t/*/checkpoints/step-4958 2>/dev/null | head -1)
  [ -n "$c" ] || { echo "$t: no step-4958 checkpoint"; continue; }
  [ -f $O/cfg$t.json ] || (cd $O && $PY getcfg.py $t > /dev/null 2>&1)
  for mode in mean shuffle; do
    p=ha; [ $mode = shuffle ] && p=has
    [ -f $O/$p$t.json ] && { echo "$p$t cached"; continue; }
    HA_MODE=$mode PYTHONPATH=. JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= nice -n 19 $PY scripts/della/pls_hidden_ablate.py $c $O/cfg$t.json $O/$p$t.json > $O/$p$t.log 2>&1
    echo "$p$t rc=$? $(grep 'base CE' $O/$p$t.log)"
  done
done
