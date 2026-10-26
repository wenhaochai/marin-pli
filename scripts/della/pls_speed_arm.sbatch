#!/bin/bash
#SBATCH --job-name=pls-speed-arm
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
#FETCHABLE: self-contained (design, knobs and rounds come from the job name, nothing from the submit shell)
# Speed work (owner 2026-10-25: ship every speed item, test jobs go through wq with --prio 10, not pli-cp). One arm of
# an A/B on 4 GPUs: a smoke-length run of a 48-layer design per round, each with its own run id (RUN_TAG -j<id><round>,
# W&B group muonh-qwen3-smoke). Two arms with the same design and different knobs are compared offline from W&B.
# Job name: pls-speed-arm-<design>-<knobs>[-<rounds>], design preln | hc | attnres, knobs:
#   base   today's programs
#   a1     LEVANTER_CE_BTILED_ONEHOT=1 (label term as a select in the b-tiled CE backward)
#   gif    LEVANTER_CE_GRAD_IN_FWD=1 (b-tiled CE gradients computed in the forward, one logits GEMM per block)
#   a1gif  both
#   a1gifk A1, GIF and the Triton lse kernel (LEVANTER_CE_LSE_KERNEL=1: logits GEMM with bf16 logits, label logit and lse)
#   xoh    LEVANTER_CE_XLA_ONEHOT=1 (label term as a select in the xla streaming CE backward: microbatch 1)
#   cbw    XLA_CMD_BUFFER_WHILE=1 (while/conditional loops as CUDA-graph commands); xohcbw: both
#   fa4    ATTN_BACKEND=fa4 (FA4/CuTe attention; changes the math, run tag -fa4); fa4cbw: with CUDA-graph loops
#   fa4thd ATTN_BACKEND=fa4thd (upstream FA4 SM90 varlen kernels; run tag -fa4thd)
#   bt4k   microbatch-1 shards on the b-tiled CE (LEVANTER_CE_BTILED_MIN_BATCH=4096) with A1 and GIF; fa4bt4k: plus FA4
#          (both OOM on hc/attnres: 96-98 GiB, w266-w269); bt4ka1: the b-tiled CE with A1, no GIF
#   a1     with -prof: the profiling round on today's production knobs (A1 on)
#   evalbase  20 steps, then 160 eval batches at the train microbatch (640 sequences for a microbatch-1 design)
#   evalpdp   20 steps, then the same 640 sequences at EVAL_PDP=4 (40 batches): eval time and memory (A7)
# Rounds (default: det time for preln, time for hc and attnres; a -det, -time or -dettime suffix sets them): det
# (XLA_FLAGS --xla_gpu_exclude_nondeterministic_ops=true, 40 steps: bitwise comparison between arms); time (default
# flags, 40 steps: step time); fix (-fix only: det with --xla_gpu_autotune_level=0, so two arms differ only by the knob);
# prof (-prof only: no command buffers, HLO dumped,
# PROFILE=20:2, for trace_families.py).
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
source scripts/della/pls_env.sh
module load proxy/default 2>/dev/null || true
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VARIANT=pls WANDB_PROJECT=marin-della
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 LOG_JAXPR=0
curl -s -o /dev/null --max-time 20 https://api.wandb.ai/healthz || export WANDB_MODE=offline
trap pls_tmp_cleanup EXIT
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
NAME=${FETCHED_JOB_NAME:-$SLURM_JOB_NAME}
spec=${NAME#pls-speed-arm-}; design=${spec%%-*}; knobs=${spec#*-}; rounds=
case $knobs in *-mem) rounds="mem"; knobs=${knobs%-mem};; *-fix) rounds="fix"; knobs=${knobs%-fix};; *-det) rounds="det"; knobs=${knobs%-det};; *-time) rounds="time"; knobs=${knobs%-time};; *-dettime) rounds="det time"; knobs=${knobs%-dettime};; *-prof) rounds="prof"; knobs=${knobs%-prof};; esac
case $design in
  preln) A="LAYERS=48 PDP=4 ARCH=preln"; ROUNDS="det time";;
  hc) A="LAYERS=48 PDP=4 ARCH=hc"; ROUNDS="time";;
  attnres) A="LAYERS=48 PDP=2 ARCH=attnres"; ROUNDS="time";;
  *) echo "unknown design in '$NAME'"; exit 2;;
esac
case $knobs in
  base) ON="";;
  a1) ON="LEVANTER_CE_BTILED_ONEHOT=1";;
  gif) ON="LEVANTER_CE_GRAD_IN_FWD=1";;
  a1gif) ON="LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1";;
  a1gifk) ON="LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 LEVANTER_CE_LSE_KERNEL=1";;
  # the live job script's settings (prod) and the candidates on top of them, for the check before a default goes live
  prod) ON="LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 EVAL_PDP=4";;
  prodk) ON="LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 EVAL_PDP=4 LEVANTER_CE_LSE_KERNEL=1";;
  prodthd) ON="LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 EVAL_PDP=4 ATTN_BACKEND=fa4thd";;
  prodkthd) ON="LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 EVAL_PDP=4 LEVANTER_CE_LSE_KERNEL=1 ATTN_BACKEND=fa4thd";;
  prodkthdp) ON="LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 EVAL_PDP=4 LEVANTER_CE_LSE_KERNEL=1 ATTN_BACKEND=fa4thd PLS_DETACH_BB=1";;   # the probes model (overrides BASE)
  # LEVANTER_CE_SKIP_UNUSED_DX=1: no input-gradient GEMM for readouts under stop_gradient (the probes model's 47 readouts)
  prodkthdx) ON="LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 EVAL_PDP=4 LEVANTER_CE_LSE_KERNEL=1 ATTN_BACKEND=fa4thd LEVANTER_CE_SKIP_UNUSED_DX=1";;
  prodkthdpx) ON="LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 EVAL_PDP=4 LEVANTER_CE_LSE_KERNEL=1 ATTN_BACKEND=fa4thd PLS_DETACH_BB=1 LEVANTER_CE_SKIP_UNUSED_DX=1";;
  # PLS_FOLD_NORM=1: the probes model's readouts fold their RMSNorm scale into their heads (their CE input is then not
  # differentiated, so its input-gradient GEMM can go); f: fold only, fx: fold and LEVANTER_CE_SKIP_UNUSED_DX
  prodkthdpf) ON="LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 EVAL_PDP=4 LEVANTER_CE_LSE_KERNEL=1 ATTN_BACKEND=fa4thd PLS_DETACH_BB=1 PLS_FOLD_NORM=1";;
  prodkthdpfx) ON="LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 EVAL_PDP=4 LEVANTER_CE_LSE_KERNEL=1 ATTN_BACKEND=fa4thd PLS_DETACH_BB=1 PLS_FOLD_NORM=1 LEVANTER_CE_SKIP_UNUSED_DX=1";;
  prodthdp) ON="LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 EVAL_PDP=4 ATTN_BACKEND=fa4thd PLS_DETACH_BB=1";;
  prodthdpf) ON="LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 EVAL_PDP=4 ATTN_BACKEND=fa4thd PLS_DETACH_BB=1 PLS_FOLD_NORM=1";;
  # LEVANTER_NORM_GUARD: guard (1e-3, the live candidate: must not stop) and guardtrip (1e-9: f32 rounding of ~1e-6 must
  # stop the run at the first compared watch step, which tests the stop path)
  # b-tiled CE for the microbatch-1 shards (4096 tokens) with the live settings, the lse kernel and fa4thd
  prodkthdbt4k) ON="LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 EVAL_PDP=4 LEVANTER_CE_LSE_KERNEL=1 ATTN_BACKEND=fa4thd LEVANTER_CE_BTILED_MIN_BATCH=4096";;
  prodkguard) ON="LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 EVAL_PDP=4 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_NORM_GUARD=1e-3";;
  prodkguardtrip) ON="LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1 EVAL_PDP=4 LEVANTER_CE_LSE_KERNEL=1 LEVANTER_NORM_GUARD=1e-9";;
  xoh) ON="LEVANTER_CE_XLA_ONEHOT=1";;
  cbw) ON="XLA_CMD_BUFFER_WHILE=1";;
  fa4) ON="ATTN_BACKEND=fa4";;
  fa4thd) ON="ATTN_BACKEND=fa4thd";;
  fa4cbw) ON="ATTN_BACKEND=fa4 XLA_CMD_BUFFER_WHILE=1";;
  bt4k) ON="LEVANTER_CE_BTILED_MIN_BATCH=4096 LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1";;
  bt4ka1) ON="LEVANTER_CE_BTILED_MIN_BATCH=4096 LEVANTER_CE_BTILED_ONEHOT=1";;
  fa4bt4k) ON="ATTN_BACKEND=fa4 LEVANTER_CE_BTILED_MIN_BATCH=4096 LEVANTER_CE_BTILED_ONEHOT=1 LEVANTER_CE_GRAD_IN_FWD=1";;
  xohcbw) ON="LEVANTER_CE_XLA_ONEHOT=1 XLA_CMD_BUFFER_WHILE=1";;
  evalbase) ON="SMOKE_STEPS=20 SMOKE_EVAL_BATCHES=160";;
  evalpdp) ON="SMOKE_STEPS=20 SMOKE_EVAL_BATCHES=40 EVAL_PDP=4";;
  *) echo "unknown knobs in '$NAME'"; exit 2;;
esac
ROUNDS=${rounds:-$ROUNDS}
# SMOKE_MID=20: an eval and a checkpoint at step 20, then 20 more train steps (compare_arms.py leaves steps 20-22 out of the time)
BASE="SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=0 PARAM_MAP=vocab NUM_GPUS=4 $A SMOKE_STEPS=40 SMOKE_MID=20"
echo "arm $NAME: design $design, knobs [$ON], rounds $ROUNDS, GPUs ${CUDA_VISIBLE_DEVICES:-all}"; nvidia-smi -L || true
for r in $ROUNDS; do
  echo "round $r $(date +%T)"
  xf=; pr=
  case $r in
    det) xf="--xla_gpu_exclude_nondeterministic_ops=true";;
    # mem (-mem only): the train step's HLO, buffer assignment and memory-usage report dumped at compile time (for an
    # arm that runs out of memory: the report shows which buffers are live at the peak)
    mem) xf="--xla_dump_to=/scratch/gpfs/GROUP/USER/tmp/pls_speed/hlo_${NAME}_${JID} --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train_step.*";;
    fix) xf="--xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_autotune_level=0";;   # det with the autotuner off: no per-run GEMM choices, so arms compare without autotune classes
    time) pr=;;   # no profile here: with command buffers on, jax.profiler.stop_trace hung and left GPUs idle (w237/w238, 2026-10-26)
    # prof: no command buffers (each kernel then shows in the trace under its own HLO instruction) and the optimized HLO
    # dumped, for trace_families.py (the Tokenizer Size session's speed-all 7b2eba3a51); its step time is not the measure
    prof) pr="PROFILE=25:2"; xf="--xla_gpu_enable_command_buffer= --xla_dump_to=/scratch/gpfs/GROUP/USER/tmp/pls_speed/hlo_${NAME}_${JID} --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train_step.*";;
  esac
  # shellcheck disable=SC2086
  env $BASE $ON $pr ${xf:+"XLA_FLAGS=$xf"} RUN_TAG=-j${JID}$r timeout 40m "$PY" -m experiments.references.della_muonh_qwen3_scaling \
    > logs/slurm/${NAME}_${JID}_$r.log 2>&1 || echo "round $r rc=$?"
  grep -E "First train step completed|Saving|peak|Traceback|Error" logs/slurm/${NAME}_${JID}_$r.log | tail -5 || true
done
echo "rounds done $(date +%T)"
out=$MARIN_PREFIX/speedrun
for d in "$out"/*-j${JID}det* "$out"/*-j${JID}time* "$out"/*-j${JID}prof* "$MARIN_PREFIX"/tmp/checkpoints-temp"$out"/*-j${JID}det* "$MARIN_PREFIX"/tmp/checkpoints-temp"$out"/*-j${JID}time* "$MARIN_PREFIX"/tmp/checkpoints-temp"$out"/*-j${JID}prof*; do
  [[ -n $JID && ( $d == *-j${JID}det* || $d == *-j${JID}time* || $d == *-j${JID}prof* ) && -d $d ]] && rm -rf -- "$d"
done
echo done
