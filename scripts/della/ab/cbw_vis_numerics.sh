#!/bin/bash
# cbw_vis_numerics.sh <gpu> <base*|cbw*> <memfrac> [cbw_numerics.py run args]: one numerics run on a vis node (ssh della-vis2,
# nice; never a login node). Writes <out>/<label>.npz (~1.6 GB at 12 layers, v64k: delete after `compare`).
set -u
W=/scratch/gpfs/GROUP/USER/tmp/speed_wt/command-buffer-while
OUT=${CBW_OUT:-$W/_num}
GPU=$1; LABEL=$2; MEMF=$3; shift 3
mkdir -p $OUT/tmp
export CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMF LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0
export TMPDIR=$OUT/tmp TRITON_CACHE_DIR=$OUT/tmp/triton JAX_COMPILATION_CACHE_DIR=
case $LABEL in
  base*) export XLA_FLAGS="" ;;
  cbw*)  export XLA_FLAGS="--xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL" ;;
  *) echo "label must start with base or cbw"; exit 2 ;;
esac
cd $OUT   # with JAX_COMPILATION_CACHE_DIR empty, XLA puts its autotune cache in the cwd: here, shared by the base and cbw runs (same kernel picks)
timeout 2400 nice -n 19 /scratch/gpfs/GROUP/USER/project/marin/.venv/bin/python $W/scripts/della/ab/cbw_numerics.py run --label $LABEL --out $OUT "$@" 2>&1 \
  | grep -E "RESULT|Error|error|Traceback|File \"" | tail -40
