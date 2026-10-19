#!/bin/bash
# attnres_followup.py on the four 48-layer AttnRes checkpoints, CPU on a vis node (nice -n 19), one model at a time.
# Block-ord picks its giant block slot (auto); Block-sep reuses that slot as the control; Full: layers 44 and 47 MLP.
#   ssh della-vis2 'nohup setsid bash /scratch/gpfs/GROUP/USER/project/marin-pls-auto10/scripts/della/attnres_followup.sh > /scratch/gpfs/GROUP/USER/tmp/pls_status/attnres_followup/run.log 2>&1 &'
set -euo pipefail
WT=/scratch/gpfs/GROUP/USER/project/marin-pls-auto10
cd $WT
source scripts/della/pls_env.sh
export JAX_PLATFORMS=cpu
C=/scratch/gpfs/GROUP/USER/tmp/pls_status/depthmed/cfg
K=/scratch/gpfs/GROUP/USER/marin_store_big/speedrun
O=/scratch/gpfs/GROUP/USER/tmp/pls_status/attnres_followup
P=muonh-qwen3-130m-della4xh100-pls1-sep
mkdir -p $O
run() {  # out-name, checkpoint dir, targets
  echo "=== $1 $(date +%T)"
  nice -n 19 $PY scripts/della/attnres_followup.py $K/$2/2026.09.13/checkpoints/step-4958 $C/$2.json $O/$1.json $1 $3 2>&1 | grep -v "Unknown leaf"
}
run block $P-bbfrozen-aattnres_block-vs-d48-g4 auto
SLOT=$($PY -c "import json; print(json.load(open('$O/block.json'))['auto_pick']['slot'])")
run block-sep $P-aattnres_block-vs-d48-g4 $SLOT
run full $P-bbfrozen-aattnres-vs-d48-g4 44.mlp,47.mlp
run full-sep $P-aattnres-vs-d48-g4 44.mlp,47.mlp
echo "all done $(date +%T)"
