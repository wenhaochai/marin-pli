#!/bin/bash
# cbw_vis_capture.sh <gpu> <base|cbw>: compile + 1 step of a 2-layer model with the XLA VLOGs of the command-buffer passes
# (vis node only), then count which while loops became command-buffer WhileCmds and which stayed host-driven.
W=/scratch/gpfs/GROUP/USER/tmp/speed_wt/command-buffer-while
OUT=${CBW_OUT:-$W/_num}
GPU=$1; LABEL=$2
mkdir -p $OUT/tmp
export CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0
export TMPDIR=$OUT/tmp TRITON_CACHE_DIR=$OUT/tmp/triton JAX_COMPILATION_CACHE_DIR=
export TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=command_buffer_conversion_pass=2,command_buffer_cmd_emitter=2
case $LABEL in base*) export XLA_FLAGS="" ;; cbw*) export XLA_FLAGS="--xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL" ;; esac
cd $OUT   # with JAX_COMPILATION_CACHE_DIR empty, XLA puts its autotune cache in the cwd: here, shared by the base and cbw runs (same kernel picks)
timeout 1200 nice -n 19 /scratch/gpfs/GROUP/USER/project/marin/.venv/bin/python $W/scripts/della/ab/cbw_numerics.py run --label v$LABEL --out $OUT/v --layers 2 --steps 1 > $OUT/$LABEL.vlog 2>&1
echo "exit $? log $OUT/$LABEL.vlog"
echo "while loops emitted as command-buffer WhileCmds in jit__train_step: $(awk '/Module jit__train_step/{t=1} t && /WhileThunk: while/{n++} END{print n+0}' $OUT/$LABEL.vlog)"
grep -a "command_buffer" $OUT/$LABEL.vlog | grep -a -E "WhileThunk is not convertible|kWhile lowering is not enabled" | sed -E 's/^[IWE][0-9]+ [0-9:.]+ +[0-9]+ //' | sort | uniq -c
