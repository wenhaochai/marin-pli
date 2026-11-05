#!/bin/bash
# Dry run (CPU, no training): run ids of HC with and without mixing at 12 and 24 layers, ordinary model, on the current code. Nothing is submitted.
cd /scratch/gpfs/GROUP/USER/project/marin-pls
PY=.venv/bin/python; LAUNCH=experiments.references.della_muonh_qwen3_scaling
pp=$PWD; for d in "$PWD"/lib/*/src; do pp=$pp:$d; done
export PYTHONPATH=$pp VARIANT=pls MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big PLS_FOLD_NORM=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
run() { out=$(env SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=1 PARAM_MAP=vocab NUM_GPUS=4 PDP=4 $1 DRY_RUN=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu nice -n 19 $PY -m $LAUNCH 2>&1); echo "[$1] -> $(echo "$out" | grep -oE 'muonh-qwen3-130m-della4xh100-pls1[^/ "]*' | sort -u | head -2 | tr '\n' ' ')"; echo "$out" | grep -E "Error|error:" | head -2 | cut -c1-200; }
run "LAYERS=12 ARCH=hc HC_RES=identity"; run "LAYERS=24 ARCH=hc HC_RES=identity"; run "LAYERS=12 ARCH=hc"; run "LAYERS=24 ARCH=hc"
