#!/bin/bash
# Dry runs (CPU, no training) of the six new arch3 jobs (second popup of 2026-11-03): prints each run id.
cd /scratch/gpfs/GROUP/USER/project/marin-pls
PY=.venv/bin/python; LAUNCH=experiments.references.della_muonh_qwen3_scaling
pp=$PWD; for d in "$PWD"/lib/*/src; do pp=$pp:$d; done
export PYTHONPATH=$pp VARIANT=pls MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big PLS_FOLD_NORM=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
base_sep="SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=0 PARAM_MAP=vocab NUM_GPUS=4"
base_pr="SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=1 PARAM_MAP=vocab NUM_GPUS=4"
while read -r name base rest; do
  b=$base_sep; [ "$base" = pr ] && b=$base_pr
  out=$(env $b $rest DRY_RUN=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu nice -n 19 $PY -m $LAUNCH 2>&1)
  echo "$name -> $(basename "$(dirname "$(echo "$out" | sed -n 's/^output: //p')")")"
  echo "$out" | grep -E "Traceback|Error" | head -3
done <<'L'
arch3-hcmove-sep sep LAYERS=48 PDP=4 ARCH=hc HC_RES=move
arch3-hcmove-probes pr LAYERS=48 PDP=4 ARCH=hc HC_RES=move
arch3-hcshrink05-sep sep LAYERS=48 PDP=4 ARCH=hc HC_RES=sumshrink SHRINK_POWER=0.5
arch3-hcshrink05-probes pr LAYERS=48 PDP=4 ARCH=hc HC_RES=sumshrink SHRINK_POWER=0.5
arch3-plleak-sep sep LAYERS=48 PDP=4 ARCH=preln PRELN_LEAK=1
arch3-plleak-probes pr LAYERS=48 PDP=4 ARCH=preln PRELN_LEAK=1
L
