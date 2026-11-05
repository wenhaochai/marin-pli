#!/bin/bash
# Run on della-vis2: state gain through the mixing at 12 and 24 layers, one design per process (2026-11-05).
R=/scratch/gpfs/GROUP/USER/project/marin-pls; H=/scratch/gpfs/GROUP/USER/tmp/pls_hcid; PY=$R/.venv/bin/python
cd $R
WANDB_BASE_URL=https://api.wandb.ai CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu nice -n 19 $PY $H/cfg_depth_1105.py 2>&1 | grep -v wandb:
for t in -sep-bbfrozen-ahc-fa4thd-vs-d12 -sep-bbfrozen-ahc-hcid-fa4thd-vs-d12 -sep-bbfrozen-ahc-fa4thd-vs-d24 -sep-bbfrozen-ahc-hcid-fa4thd-vs-d24; do
  OUT=$H/res_gain_depth$t.npz CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=. nice -n 19 timeout 900 $PY $H/res_gain_depth.py $t 2>&1 | grep -E '^ *L[0-9]|^-sep|Traceback|Error' | cut -c1-200
done
