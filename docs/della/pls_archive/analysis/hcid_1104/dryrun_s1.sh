#!/bin/bash
# Dry run (CPU, no training) of the candidate one-stream arms: prints the run id of each; nothing is submitted.
cd /scratch/gpfs/GROUP/USER/project/marin-pls
PY=.venv/bin/python; LAUNCH=experiments.references.della_muonh_qwen3_scaling
pp=$PWD; for d in "$PWD"/lib/*/src; do pp=$pp:$d; done
export PYTHONPATH=$pp VARIANT=pls MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big PLS_FOLD_NORM=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
run() { out=$(env SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=$1 PARAM_MAP=vocab NUM_GPUS=4 LAYERS=48 PDP=4 $2 DRY_RUN=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu nice -n 19 $PY -m $LAUNCH 2>&1); echo "[$2] BB=$1 -> $(echo "$out" | grep -oE 'muonh-qwen3-130m-della4xh100-pls1[^/ "]*' | sort -u | head -2 | tr '\n' ' ')"; echo "$out" | grep -E "Error|error:" | head -2 | cut -c1-200; }
run 1 "ARCH=hc HC_STREAMS=1"
run 1 "ARCH=hc HC_STREAMS=1 HC_RES=identity"
run 1 "ARCH=hc HC_STREAMS=1 HC_RES=scalar"
run 1 "ARCH=hc HC_STREAMS=1 ROUTE=static"
run 1 "ARCH=preln PRELN_LEAK=1 SHRINK_POWER=0.1"
