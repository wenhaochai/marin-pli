#!/bin/bash
cd /scratch/gpfs/GROUP/USER/project/marin-pls
PY=/scratch/gpfs/GROUP/USER/project/marin-pls/.venv/bin/python
O=/scratch/gpfs/GROUP/USER/tmp/pls_paths
S=/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1
for a in apreln-vs-d48-g4 ahc-vs-d48-g4 amhc-vs-d48-g4 aattnres-vs-d48-s1-g4 amhc-mhcid-vs-d48; do
  for m in -sep -sep-bbfrozen; do
    t=$m-$a
    [ -f $O/cfg$t.json ] || $PY $O/getcfg.py $t > /dev/null 2>&1
    c=$(ls -d $S$t/*/checkpoints/step-4958 | head -1)
    PYTHONPATH=. JAX_PLATFORMS=cpu nice -n 19 $PY scripts/della/pls_write_size.py $c $O/cfg$t.json $O/ws$t.json > $O/ws$t.log 2>&1
    echo "$t rc=$?"
  done
done
