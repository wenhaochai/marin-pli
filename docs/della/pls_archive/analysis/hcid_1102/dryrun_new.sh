#!/bin/bash
# Dry runs (CPU, no training) of the six new arch3 jobs: prints each run id and the depth_* config the launcher builds.
cd /scratch/gpfs/GROUP/USER/project/marin-pls
PY=.venv/bin/python; LAUNCH=experiments.references.della_muonh_qwen3_scaling
# the launch script's environment without sourcing pls_env.sh (its TMPDIR trap belongs in jobs only)
pp=$PWD; for d in "$PWD"/lib/*/src; do pp=$pp:$d; done
export PYTHONPATH=$pp VARIANT=pls MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big PLS_FOLD_NORM=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
base_sep="SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=0 PARAM_MAP=vocab NUM_GPUS=4"
base_pr="SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=1 PARAM_MAP=vocab NUM_GPUS=4"
while read -r name base rest; do
  b=$base_sep; [ "$base" = pr ] && b=$base_pr
  out=$(env $b $rest DRY_RUN=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu nice -n 19 $PY -m $LAUNCH 2>&1)
  echo "$name -> $(basename "$(dirname "$(echo "$out" | sed -n 's/^output: //p')")")"
  echo "$out" | grep -oE "depth_hc_rwdyn='?[a-z]+'?|depth_hc_res='?[a-z]+'?|depth_mhc_rw='?[a-z]+'?|depth_mhc_res='?[a-z]+'?|seed=[0-9]+" | sort -u | tr '\n' ' '; echo
done <<'L'
arch3-hcrwj-sep sep LAYERS=48 PDP=4 ARCH=hc HC_RWDYN=joint
arch3-hcrwj-probes pr LAYERS=48 PDP=4 ARCH=hc HC_RWDYN=joint
arch3-hc-s1-probes pr LAYERS=48 PDP=4 ARCH=hc SEED=1
arch3-hcid-s1-probes pr LAYERS=48 PDP=4 ARCH=hc HC_RES=identity SEED=1
arch3-hcid-s1-sep sep LAYERS=48 PDP=4 ARCH=hc HC_RES=identity SEED=1
arch3-mhcidlin-s1-probes pr LAYERS=48 PDP=4 ARCH=mhc MHC_RES=identity MHC_RW=linear SEED=1
L
