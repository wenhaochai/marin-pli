#!/bin/bash
#SBATCH --job-name=speedall-arms4
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell; run through wq --prio 10 (owner 2026-10-25: tests go to wq, not pli-cp)

# H100 check of branch speed-all (owner 2026-10-25 "全部都上"): one 300m config, SMOKE_STEPS steps at the real batch, the
# arms named on the command line in turn on the same 4 GPUs:
#   old     = the main checkout (vocab-overfitting, every run so far): JAX flash, full remat, power-of-two flash block
#   base    = speed-all with no switch on (new flash block rule only)
#   cb      = CUDA-graph while loops
#   jax     = REMAT=flash + cb        none    = REMAT=none (no per-layer remat) + cb
#   fa4     = FA4/CuTe attention + cb  fa4none = FA4 + REMAT=none + cb
#   a1      = CE label term as a select (pls 5ab04d3e45)   gif = a1 + CE gradient in the forward
#   dbase, dbase2, da1, dgif = base, base again, a1, gif with --xla_gpu_exclude_nondeterministic_ops (per-step loss bitwise?)
#   now     = the defaults since 2026-10-26 04:30 (FA4, REMAT=flash, CE A1 + GIF, cb)   nowdots = now with REMAT=fdots
#   nowpipe = now with XLA collective pipelining (all-gather, reduce-scatter) and the latency-hiding scheduler;
#   nowpipe2 = nowpipe with while-loop double buffering
#   nowlse  = now with LEVANTER_CE_LSE_KERNEL=1 (the Triton logits + logsumexp kernel in the GIF block)
#   nowthd  = now with ATTN_BACKEND=fa4thd (upstream FA4 varlen kernels, native SM90)
#   live    = the live defaults since 2026-10-26 05:41 (fa4thd, REMAT=flash, CE A1 + GIF + lse kernel, cb); configs
#             4096post / 5504post run 80 steps with an eval every 20 and a checkpoint every 20 s
#   full    = the defaults since 2026-10-26 05:40 (fa4thd, REMAT=fdots, CE A1 + GIF + lse kernel, cb); fullnocb = full with
#             XLA's default command buffers (no while/conditional capture)
# Per arm a SUMMARY line (exit code, peak memory, mean and median step time over steps >= 3, final loss); per later arm a
# PAIR line, its per-step train/loss minus the first arm's (same seed, data, schedule; bf16 noise is ~1e-3).
set -uo pipefail
CFG=${1:?usage: ab_arms.sbatch <4096|4480> <arm>...}
shift
ARMS=("$@")
[ ${#ARMS[@]} -ge 1 ] || { echo "no arms"; exit 2; }
MAIN=/scratch/gpfs/GROUP/USER/project/marin
WT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/async   # branch async-hooks
PY=$MAIN/.venv/bin/python
cd "$WT"
source scripts/della/job_tmpdir.sh
module load proxy/default  # wandb only
source scripts/della/wandb_guard.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
unset VARIANT DATA_EPOCHS PASS_DATA TOTAL_STEPS INIT_FROM SEED BATCH PDP NUM_GPUS TIE PRECISION XLA_FLAGS ATTN_BACKEND REMAT SMOKE_EVAL_EVERY SMOKE_CKPT_SECONDS SMOKE_FULL_EVAL US_EPS US_FROM_PASS
export NUM_GPUS=4 PDP=32   # 4 GPUs at 32 sequences per device: one microbatch per step, as the vocab runs
case $CFG in
  4096) export SIZE=300m SEQ_LEN=4096; unset VOCAB_K ;;
  4480) export SIZE=300m SEQ_LEN=4480 VOCAB_K=32000 ;;
  5504) export SIZE=300m SEQ_LEN=5504 VOCAB_K=8000 ;;   # charwin v8k (w138)
  4096post) export SIZE=300m SEQ_LEN=4096 SMOKE_STEPS=80 SMOKE_EVAL_EVERY=20 SMOKE_CKPT_SECONDS=20; unset VOCAB_K ;;   # train steps after evals and checkpoints
  4096long) export SIZE=300m SEQ_LEN=4096 SMOKE_STEPS=200; unset VOCAB_K ;;   # 200 steps, one eval at the end: steady-state wall per step
  4096postfull) export SIZE=300m SEQ_LEN=4096 SMOKE_STEPS=120 SMOKE_EVAL_EVERY=20 SMOKE_CKPT_SECONDS=20 SMOKE_FULL_EVAL=1; unset VOCAB_K ;;   # 128K with full Paloma evals and checkpoints mid-run
  w160repro) export SIZE=300m SEQ_LEN=4096 VOCAB_K=8000 VARIANT=unismooth US_EPS=0.02 US_FROM_PASS=2 DATA_EPOCHS=8 SMOKE_STEPS=120 SMOKE_EVAL_EVERY=20 SMOKE_CKPT_SECONDS=20 SMOKE_FULL_EVAL=1 ;;   # the w160 job on 4 GPUs, full evals
  5504post) export SIZE=300m SEQ_LEN=5504 VOCAB_K=8000 SMOKE_STEPS=80 SMOKE_EVAL_EVERY=20 SMOKE_CKPT_SECONDS=20 ;;
  4096v8k) export SIZE=300m SEQ_LEN=4096 VOCAB_K=8000 ;;
  v8klong) export SIZE=300m SEQ_LEN=4096 VOCAB_K=8000 SMOKE_STEPS=200 ;;   # the w160 shape, 200 steps
  v8kpostfull) export SIZE=300m SEQ_LEN=4096 VOCAB_K=8000 SMOKE_STEPS=120 SMOKE_EVAL_EVERY=20 SMOKE_CKPT_SECONDS=20 SMOKE_FULL_EVAL=1 ;;   # 8K (w393 shape) with full Paloma evals and checkpoints mid-run
  4480postfull) export SIZE=300m SEQ_LEN=4480 VOCAB_K=32000 SMOKE_STEPS=120 SMOKE_EVAL_EVERY=20 SMOKE_CKPT_SECONDS=20 SMOKE_FULL_EVAL=1 ;;   # 32K charwin (w125 shape) with full Paloma evals and checkpoints mid-run
  *) echo "unknown config $CFG"; exit 2 ;;
esac
export SMOKE_STEPS=${SMOKE_STEPS:-60}
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}   # wq guest: SLURM_JOB_ID is the host's
LOGDIR=$MAIN/logs/slurm
nvidia-smi -L
fail=0
declare -A RID
for ARM in "${ARMS[@]}"; do
  if [ $ARM = old ] || [[ $ARM == main* ]]; then DIR=$MAIN; PP=""; else DIR=$WT; PP="$WT/lib/levanter/src:$WT/lib/haliax/src:$WT"; fi   # old and main* = the main checkout; every other arm = this worktree
  unset XLA_FLAGS ATTN_BACKEND REMAT LEVANTER_CE_BTILED_ONEHOT LEVANTER_CE_GRAD_IN_FWD LEVANTER_CE_XLA_ONEHOT LEVANTER_CE_LSE_KERNEL LEVANTER_ASYNC_STEP XLA_PYTHON_CLIENT_ALLOCATOR LEVANTER_CE_LSE_KERNEL_EVAL LEVANTER_EMBED_GRAD_F32 LEVANTER_NORM_GUARD SCAN_LAYERS LEVANTER_CE_LSE_MGPU LEVANTER_CE_BTILED_MIN_VOCAB MESH LEVANTER_SCAN_EMPTY LEVANTER_CE_GRADW_F32 LEVANTER_CE_BTILED_F32_LOGITS LEVANTER_SWIGLU_VJP
  export REMAT=full   # an arm that names no REMAT keeps full remat (the code default is flash since 2026-10-26)
  export ATTN_BACKEND=fa4   # an arm that names no backend keeps the segmented FA4 kernel (the code default at 300m is fa4thd since 2026-10-26)
  DET="--xla_gpu_nccl_termination_timeout_seconds=600 --xla_gpu_exclude_nondeterministic_ops=true"
  CB="--xla_gpu_nccl_termination_timeout_seconds=600 --xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL"
  PIPE="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true"   # FSDP collectives of the next layer inside this layer's scan iteration
  case $ARM in
    old) export ATTN_BACKEND=jax_flash ;;   # the old path: the main checkout defaults to fa4 since 2026-10-26
    base) ;;
    cb) export XLA_FLAGS="$CB" ;;
    jax) export REMAT=flash XLA_FLAGS="$CB" ;;
    none) export REMAT=none XLA_FLAGS="$CB" ;;
    fa4) export ATTN_BACKEND=fa4 XLA_FLAGS="$CB" ;;
    jaxflash) export ATTN_BACKEND=jax_flash XLA_FLAGS="$CB" ;;
    fa4none) export ATTN_BACKEND=fa4 REMAT=none XLA_FLAGS="$CB" ;;
    dots) export REMAT=dots XLA_FLAGS="$CB" ;;
    fa4dots) export ATTN_BACKEND=fa4 REMAT=dots XLA_FLAGS="$CB" ;;
    fa4flash) export ATTN_BACKEND=fa4 REMAT=flash XLA_FLAGS="$CB" ;;
    now) export ATTN_BACKEND=fa4 REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 XLA_FLAGS="$CB" ;;   # the defaults since 2026-10-26 04:30
    nowdots) export ATTN_BACKEND=fa4 REMAT=fdots LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 XLA_FLAGS="$CB" ;;
    nowlse) export ATTN_BACKEND=fa4 REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 XLA_FLAGS="$CB" ;;
    nowpipe) export ATTN_BACKEND=fa4 REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 XLA_FLAGS="$CB $PIPE" ;;
    nowpipe2) export ATTN_BACKEND=fa4 REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 XLA_FLAGS="$CB $PIPE --xla_gpu_enable_while_loop_double_buffering=true" ;;
    nowthd) export ATTN_BACKEND=fa4thd REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 XLA_FLAGS="$CB" ;;
    live) export ATTN_BACKEND=fa4thd REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 XLA_FLAGS="$CB" ;;   # the live defaults since 2026-10-26 05:41
    mainlive) export ATTN_BACKEND=fa4thd REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 XLA_FLAGS="$CB" ;;   # live flags on the main checkout (without this branch's host fixes)
    liveasync) export ATTN_BACKEND=fa4thd REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 XLA_FLAGS="$CB" ;;   # live with the async host loop
    livefdots) export ATTN_BACKEND=fa4thd REMAT=fdots LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 XLA_FLAGS="$CB" ;;   # live with fdots (w160 ran out of memory)
    livedm) export ATTN_BACKEND=fa4thd REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 XLA_FLAGS="$CB --xla_gpu_dot_merger_threshold_mb=4096 --xla_dump_to=/scratch/gpfs/GROUP/USER/tmp/speed_wt/profiles/dm-$JID --xla_dump_hlo_module_re=jit__train_step" ;;   # XLA merges dots that share an operand (q/k/v, gate/up) up to 4 GiB
    liveevalk) export ATTN_BACKEND=fa4thd REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_CE_LSE_KERNEL_EVAL=1 XLA_FLAGS="$CB" ;;   # live with the lse kernel in evaluation too
    livefproj) export ATTN_BACKEND=fa4thd REMAT=fproj LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 XLA_FLAGS="$CB" ;;   # flash + the attention projections saved
    liveasyncfproj) export ATTN_BACKEND=fa4thd REMAT=fproj LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 XLA_FLAGS="$CB" ;;
    liveasyncfprojcua) export ATTN_BACKEND=fa4thd REMAT=fproj LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_FLAGS="$CB" ;;
    liveemb) export ATTN_BACKEND=fa4thd REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_EMBED_GRAD_F32=1 XLA_FLAGS="$CB" ;;   # the input-embedding gradient summed in f32
    liveasyncemb) export ATTN_BACKEND=fa4thd REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 XLA_FLAGS="$CB" ;;
    newlive) export ATTN_BACKEND=fa4thd REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 XLA_FLAGS="$CB" ;;   # the live defaults since 2026-10-26 12:17
    newfgate) export ATTN_BACKEND=fa4thd REMAT=fgate LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 XLA_FLAGS="$CB" ;;   # + the MLP gate output saved
    newfmlp) export ATTN_BACKEND=fa4thd REMAT=fmlp LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 XLA_FLAGS="$CB" ;;   # + the MLP gate and up outputs saved
    newguard) export ATTN_BACKEND=fa4thd REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 LEVANTER_NORM_GUARD=1e-3 XLA_FLAGS="$CB" ;;   # newlive + the constant-norm guard
    newfmlpguard) export ATTN_BACKEND=fa4thd REMAT=fmlp LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 LEVANTER_NORM_GUARD=1e-3 XLA_FLAGS="$CB" ;;
    newunroll) export ATTN_BACKEND=fa4thd REMAT=fmlp LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 LEVANTER_NORM_GUARD=1e-3 SCAN_LAYERS=0 XLA_FLAGS="$CB" ;;   # the live defaults, layers unrolled
    newfmlp2) export ATTN_BACKEND=fa4thd REMAT=fmlp LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 LEVANTER_NORM_GUARD=1e-3 XLA_FLAGS="$CB" ;;   # the live defaults since 13:09
    newmgpu) export ATTN_BACKEND=fa4thd REMAT=fmlp LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_CE_LSE_MGPU=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 LEVANTER_NORM_GUARD=1e-3 XLA_FLAGS="$CB" ;;   # the live defaults with lse kernel v3
    newbt) export ATTN_BACKEND=fa4thd REMAT=fmlp LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 LEVANTER_NORM_GUARD=1e-3 LEVANTER_CE_BTILED_MIN_VOCAB=4096 XLA_FLAGS="$CB" ;;   # the live defaults, the b-tiled CE also below 65536
    newbtf) export ATTN_BACKEND=fa4thd REMAT=fmlp LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 LEVANTER_NORM_GUARD=1e-3 LEVANTER_CE_BTILED_MIN_VOCAB=4096 LEVANTER_CE_BTILED_F32_LOGITS=1 XLA_FLAGS="$CB" ;;   # newbt with f32 logits in the backward (no lse kernel), the precision of the path it replaces
    newdp) export ATTN_BACKEND=fa4thd REMAT=fmlp LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 LEVANTER_NORM_GUARD=1e-3 MESH=dp XLA_FLAGS="$CB" ;;   # the live defaults with plain data parallelism (parameters replicated)
    newdsf) export ATTN_BACKEND=fa4thd REMAT=fmlp LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 LEVANTER_NORM_GUARD=1e-3 XLA_FLAGS="$CB --xla_gpu_enable_dynamic_slice_fusion=true" ;;   # the live defaults with XLA dynamic-slice fusion (GEMMs read and write the layer scan's stacked buffers in place)
    newempty) export ATTN_BACKEND=fa4thd REMAT=fmlp LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 LEVANTER_NORM_GUARD=1e-3 LEVANTER_SCAN_EMPTY=1 XLA_FLAGS="$CB" ;;   # the live defaults with the layer scan's stacked outputs unwritten at the start (no zero fill)
    newgradw) export ATTN_BACKEND=fa4thd REMAT=fmlp LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 LEVANTER_NORM_GUARD=1e-3 LEVANTER_CE_GRADW_F32=1 XLA_FLAGS="$CB" ;;   # the live defaults with the CE weight-gradient GEMM in f32 (its running sum folded into the GEMM)
    newswiglu) export ATTN_BACKEND=fa4thd REMAT=fmlp LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 LEVANTER_NORM_GUARD=1e-3 LEVANTER_SWIGLU_VJP=1 XLA_FLAGS="$CB" ;;   # the live defaults with the MLP's act(gate) * up -> down_proj as one custom_vjp (the backward's product and its gradients in one pass)
    newce) export ATTN_BACKEND=fa4thd REMAT=fmlp LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 LEVANTER_NORM_GUARD=1e-3 LEVANTER_CE_LSE_MGPU=1 LEVANTER_CE_BTILED_MIN_VOCAB=4096 LEVANTER_CE_LSE_KERNEL_EVAL=1 XLA_FLAGS="$CB" ;;   # the live defaults with the three CE items: lse kernel v3, the b-tiled CE from 4096 entries, the lse kernel in evaluation
    newzie) export ATTN_BACKEND=fa4thd REMAT=fmlp LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 LEVANTER_NORM_GUARD=1e-3 XLA_FLAGS="$CB --xla_gpu_enable_dus_accumulator_zero_init_elimination=true" ;;   # the live defaults with XLA's DUS-accumulator zero-init elimination (the layer scan's zero-filled stacked buffers)
    newpdl) export ATTN_BACKEND=fa4thd REMAT=fmlp LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 LEVANTER_NORM_GUARD=1e-3 XLA_FLAGS="$CB --xla_gpu_enable_pdl=true --xla_gpu_enable_pdl_launch=true" ;;   # the live defaults with programmatic dependent launch (a kernel's prologue overlaps the previous kernel's tail)
    newub) export ATTN_BACKEND=fa4thd REMAT=fmlp LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 LEVANTER_NORM_GUARD=1e-3 XLA_FLAGS="$CB --xla_gpu_enable_nccl_user_buffers=true" ;;   # the live defaults with NCCL user buffers (registered collective memory; NVLink SHARP on NVSwitch)
    newtri) export ATTN_BACKEND=fa4thd REMAT=fmlp LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 LEVANTER_NORM_GUARD=1e-3 XLA_FLAGS="$CB --xla_gpu_triton_gemm_any=true" ;;   # the live defaults with Triton GEMM fusions for every dot (elementwise producers such as the SwiGLU product can fuse into a GEMM); the autotuner may still pick cuBLAS
    newtri2) export ATTN_BACKEND=fa4thd REMAT=fmlp LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 LEVANTER_NORM_GUARD=1e-3 XLA_FLAGS="$CB --xla_gpu_triton_gemm_any=true --xla_gpu_cublas_fallback=false" ;;   # newtri without the cuBLAS fallback (the Triton fusions are kept)
    newdsflt) export ATTN_BACKEND=fa4thd REMAT=fmlp LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 LEVANTER_EMBED_GRAD_F32=1 LEVANTER_NORM_GUARD=1e-3 XLA_FLAGS="$CB --xla_gpu_enable_dynamic_slice_fusion=true --xla_gpu_enable_cublaslt=false" ;;   # dynamic-slice fusion with the legacy cuBLAS GEMM call (the live HLO calls cublasLt, 133 of 133 GEMMs)
    livecua) export ATTN_BACKEND=fa4thd REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_FLAGS="$CB" ;;   # live on the CUDA stream-ordered allocator (no contiguous-block need)
    liveasyncfdotscua) export ATTN_BACKEND=fa4thd REMAT=fdots LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_FLAGS="$CB" ;;   # the proposed stack: async loop, fdots, cuda_async
    livefdotscua) export ATTN_BACKEND=fa4thd REMAT=fdots LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_FLAGS="$CB" ;;   # fdots on that allocator
    livenocb) export ATTN_BACKEND=fa4thd REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 XLA_FLAGS="--xla_gpu_nccl_termination_timeout_seconds=600" ;;
    liveasyncnocb) export ATTN_BACKEND=fa4thd REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 XLA_FLAGS="--xla_gpu_nccl_termination_timeout_seconds=600" ;;
    liveoff) export ATTN_BACKEND=fa4thd REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 XLA_FLAGS="--xla_gpu_nccl_termination_timeout_seconds=600 --xla_gpu_enable_command_buffer=" ;;   # no CUDA graphs at all
    liveasyncoff) export ATTN_BACKEND=fa4thd REMAT=flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_ASYNC_STEP=1 XLA_FLAGS="--xla_gpu_nccl_termination_timeout_seconds=600 --xla_gpu_enable_command_buffer=" ;;
    full) export ATTN_BACKEND=fa4thd REMAT=fdots LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 XLA_FLAGS="$CB" ;;
    fullnocb) export ATTN_BACKEND=fa4thd REMAT=fdots LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1 XLA_FLAGS="--xla_gpu_nccl_termination_timeout_seconds=600" ;;
    a1) export LEVANTER_CE_BTILED_ONEHOT=1 ;;
    gif) export LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 ;;
    dbase|dbase2) export XLA_FLAGS="$DET" ATTN_BACKEND=jax_flash ;;   # det arms on JAX flash (FA4 backward is not run-to-run deterministic)
    da1) export XLA_FLAGS="$DET" ATTN_BACKEND=jax_flash LEVANTER_CE_BTILED_ONEHOT=1 ;;
    dgif) export XLA_FLAGS="$DET" ATTN_BACKEND=jax_flash LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 ;;
    *) echo "unknown arm $ARM"; fail=1; continue ;;
  esac
  export RUN_TAG=-abspd${ARM}-j${JID}
  cd "$DIR"
  got=$(PYTHONPATH=$PP $PY -c 'import levanter; print(levanter.__file__)')
  echo "ARM $ARM dir=$DIR levanter=$got XLA_FLAGS=${XLA_FLAGS:-} ATTN_BACKEND=${ATTN_BACKEND:-} REMAT=${REMAT:-} CE: onehot=${LEVANTER_CE_BTILED_ONEHOT:-0} gif=${LEVANTER_CE_GRAD_IN_FWD:-0}"
  run_id=$(PYTHONPATH=$PP DRY_RUN=1 $PY -m experiments.references.della_muonh_qwen3_scaling 2>/dev/null | sed -n 's#^output: speedrun/\([^/]*\)/.*#\1#p' | head -1)
  case "$run_id" in *-abspd${ARM}-j${JID}*) ;; *) echo "ARM $ARM: dry run gave '$run_id'"; fail=1; continue ;; esac
  RID[$ARM]=$run_id
  log=$LOGDIR/speedall-ab_${JID}_${run_id}.log
  echo "SMOKE $run_id START $(date +%H:%M:%S)"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > $log.mem &
  mpid=$!
  PYTHONPATH=$PP timeout --signal=TERM ${RUN_TIMEOUT:-10m} $PY -m experiments.references.della_muonh_qwen3_scaling > $log 2>&1
  rc=$?
  kill $mpid
  peak=$(sort -n $log.mem | tail -1)
  err=$(grep -a -m1 -E "RESOURCE_EXHAUSTED|out of memory|Traceback|NotImplementedError|CUDA_ERROR" $log | cut -c1-200)
  { [ "$rc" -ne 0 ] || [ -n "$err" ]; } && fail=1
  echo "SMOKE $run_id END rc=$rc $(date +%H:%M:%S)"
  $PY - "$run_id" "$rc" "$err" "$peak" <<'PY'
import sys
import numpy as np
import wandb
run_id, rc, err, peak = sys.argv[1:5]
try:
    run = wandb.Api(timeout=60).run(f"marin-della/{run_id}")
    s = run.summary
    rows = sorted((r for r in run.scan_history(keys=["_step", "_runtime", "throughput/duration"]) if r["_step"] >= 3), key=lambda r: r["_step"])
    d = [r["throughput/duration"] for r in rows]
    wall = (rows[-1]["_runtime"] - rows[0]["_runtime"]) / max(1, rows[-1]["_step"] - rows[0]["_step"])   # evals and checkpoints included
    mid = [r for r in rows if 20 <= r["_step"] <= 190]
    wall_mid = (mid[-1]["_runtime"] - mid[0]["_runtime"]) / max(1, mid[-1]["_step"] - mid[0]["_step"]) if len(mid) > 1 else float("nan")
    print(f"SUMMARY {run_id} rc={rc} peak_mem_MiB={peak} step_ms={np.mean(d) * 1e3:.0f}(n={len(d)}) median_step_ms={np.median(d) * 1e3:.0f} wall_ms={wall * 1e3:.0f} wall_mid_ms={wall_mid * 1e3:.1f} "
          f"loss={s.get('train/loss')} eval_loss={s.get('eval/loss')} err='{err}'", flush=True)
except Exception as e:
    print(f"SUMMARY {run_id} rc={rc} peak_mem_MiB={peak} wandb_error={str(e)[:160]} err='{err}'", flush=True)
PY
  case "$run_id" in *-abspd*-j${JID}*) rm -rf "$MARIN_PREFIX/speedrun/$run_id" ;; esac
done
cd "$WT"
REF=${ARMS[0]}
for ARM in "${ARMS[@]:1}"; do
  [ -n "${RID[$REF]:-}" ] && [ -n "${RID[$ARM]:-}" ] || continue
  $PY - "${RID[$REF]}" "${RID[$ARM]}" <<'PY'
import sys
import numpy as np
import wandb
a, b = sys.argv[1:3]
try:
    api = wandb.Api(timeout=60)
    h = {r: {x["_step"]: x["train/loss"] for x in api.run(f"marin-della/{r}").scan_history(keys=["_step", "train/loss"])} for r in (a, b)}
    steps = sorted(set(h[a]) & set(h[b]))
    d = np.array([h[b][t] - h[a][t] for t in steps])
    print(f"PAIR {b} - {a}: steps={len(steps)} max_abs_loss_diff={np.max(np.abs(d)):.3e} bitwise={bool(np.all(d == 0))} mean_diff={d.mean():+.5f} "
          f"last10_mean_diff={d[-10:].mean():+.5f} first5={np.round(d[:5], 5).tolist()}", flush=True)
except Exception as e:
    print(f"PAIR {a} {b}: wandb_error={str(e)[:160]}", flush=True)
PY
done
echo "SMOKE DONE $(date +%H:%M:%S) fail=$fail"
exit $fail
