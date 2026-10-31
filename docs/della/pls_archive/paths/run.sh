#!/bin/bash
cd /scratch/gpfs/GROUP/USER/project/marin-pls
PY=/scratch/gpfs/GROUP/USER/project/marin-pls/.venv/bin/python
O=/scratch/gpfs/GROUP/USER/tmp/pls_paths
S=/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1
for t in -sep-amhc-mhcid-vs-d48 -sep-bbfrozen-amhc-mhcid-vs-d48; do
  PYTHONPATH=. JAX_PLATFORMS=cpu nice -n 19 $PY scripts/della/pls_branch_paths.py $S$t/2026.09.13/checkpoints/step-4958 $O/cfg$t.json $O/paths$t.npz > $O/paths$t.log 2>&1
  echo "$t rc=$?"
done
