#!/bin/bash
# 2026-10-30: stream-difference shuffle ablation (scripts/della/pls_hidden_ablate.py) on the HC / mHC 48-layer pairs.
cd /scratch/gpfs/GROUP/USER/project/marin-pls
PY=/scratch/gpfs/GROUP/USER/project/marin-pls/.venv/bin/python
O=/scratch/gpfs/GROUP/USER/tmp/pls_paths
S=/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1
for a in ahc-vs-d48-g4 amhc-vs-d48-g4 amhc-mhcid-vs-d48 amhc-mhcrwlin-vs-d48 ahc-hcrwsig-vs-d48 ahc-hcs2-vs-d48 ahc-hcs8-vs-d48 ahc-vs-d48-s1; do
  for m in -sep -sep-bbfrozen; do
    t=$m-$a
    [ -f $O/has$t.json ] && { echo "$t cached"; continue; }
    [ -f $O/cfg$t.json ] || (cd $O && $PY getcfg.py $t > /dev/null 2>&1)
    c=$(ls -d $S$t/*/checkpoints/step-4958 | head -1)
    HA_MODE=shuffle PYTHONPATH=. JAX_PLATFORMS=cpu nice -n 19 $PY scripts/della/pls_hidden_ablate.py $c $O/cfg$t.json $O/has$t.json > $O/has$t.log 2>&1
    echo "$t rc=$? $(grep 'base CE' $O/has$t.log)"
  done
done
