#!/bin/bash
# 2026-10-30: HC learned residual mixing on all 48-layer HC pairs (scripts/della/pls_hc_mixing.py).
cd /scratch/gpfs/GROUP/USER/project/marin-pls
PY=/scratch/gpfs/GROUP/USER/project/marin-pls/.venv/bin/python
O=/scratch/gpfs/GROUP/USER/tmp/pls_paths
S=/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1
for a in ahc-vs-d48-g4 ahc-hcs1-vs-d48 ahc-hcs2-vs-d48 ahc-hcs8-vs-d48 ahc-hcrwsig-vs-d48 ahc-vs-d48-s1; do
  for m in -sep -sep-bbfrozen; do
    t=$m-$a
    [ -f $O/mix$t.json ] && { echo "$t cached"; continue; }
    [ -f $O/cfg$t.json ] || (cd $O && $PY getcfg.py $t > /dev/null 2>&1)
    c=$(ls -d $S$t/*/checkpoints/step-4958 | head -1)
    PYTHONPATH=. JAX_PLATFORMS=cpu nice -n 19 $PY scripts/della/pls_hc_mixing.py $c $O/cfg$t.json $O/mix$t.json > $O/mix$t.log 2>&1
    echo "$t rc=$? $(tail -1 $O/mix$t.log)"
  done
done
