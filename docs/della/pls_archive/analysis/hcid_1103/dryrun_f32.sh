#!/bin/bash
# Dry runs (CPU, no training) of the two arch3-hcsumf32 jobs (third popup of 2026-11-03): prints each run id.
cd /scratch/gpfs/GROUP/USER/project/marin-pls
PY=.venv/bin/python; LAUNCH=experiments.references.della_muonh_qwen3_scaling
pp=$PWD; for d in "$PWD"/lib/*/src; do pp=$pp:$d; done
export PYTHONPATH=$pp VARIANT=pls MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big PLS_FOLD_NORM=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
for bb in 0 1; do
  out=$(env SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=$bb PARAM_MAP=vocab NUM_GPUS=4 LAYERS=48 PDP=4 ARCH=hc HC_RES=sum HC_STATE_F32=1 DRY_RUN=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu nice -n 19 $PY -m $LAUNCH 2>&1)
  echo "BB=$bb -> $(basename "$(dirname "$(echo "$out" | sed -n 's/^output: //p')")")"; echo "$out" | grep -E "Traceback|Error" | head -3
done
