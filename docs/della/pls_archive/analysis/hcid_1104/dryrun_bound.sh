#!/bin/bash
# Dry run (CPU, no training) of arch3-hcidfb100-sep (owner popup 2026-11-04): prints the run id; the same with no bound must give w810's run id, and wrong settings must fail.
cd /scratch/gpfs/GROUP/USER/project/marin-pls
PY=.venv/bin/python; LAUNCH=experiments.references.della_muonh_qwen3_scaling
pp=$PWD; for d in "$PWD"/lib/*/src; do pp=$pp:$d; done
export PYTHONPATH=$pp VARIANT=pls MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big PLS_FOLD_NORM=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
run() { out=$(env SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=$1 PARAM_MAP=vocab NUM_GPUS=4 LAYERS=48 PDP=4 $2 DRY_RUN=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu nice -n 19 $PY -m $LAUNCH 2>&1); echo "[$2] BB=$1 -> $(basename "$(dirname "$(echo "$out" | sed -n 's/^output: //p')")")"; echo "$out" | grep -E "Traceback|ValueError|depth_branch_bound" | head -3 | cut -c1-200; }
run 0 "ARCH=hc HC_RES=identity BRANCH_BOUND=100"
run 0 "ARCH=hc HC_RES=identity"
run 1 "ARCH=hc HC_RES=identity"
run 0 "ARCH=hc"
run 0 "ARCH=preln BRANCH_BOUND=100"
