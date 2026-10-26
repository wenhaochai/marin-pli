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
#   xoh    LEVANTER_CE_XLA_ONEHOT=1 (label term as a select in the xla streaming CE backward: microbatch 1)
#   cbw    XLA_CMD_BUFFER_WHILE=1 (while/conditional loops as CUDA-graph commands); xohcbw: both
#   fa4    ATTN_BACKEND=fa4 (FA4/CuTe attention; changes the math, run tag -fa4); fa4cbw: with CUDA-graph loops
#   bt4k   microbatch-1 shards on the b-tiled CE (LEVANTER_CE_BTILED_MIN_BATCH=4096) with A1 and GIF; fa4bt4k: plus FA4
#          (both OOM on hc/attnres: 96-98 GiB, w266-w269); bt4ka1: the b-tiled CE with A1, no GIF
#   a1     with -prof: the profiling round on today's production knobs (A1 on)
#   evalbase  20 steps, then 160 eval batches at the train microbatch (640 sequences for a microbatch-1 design)
#   evalpdp   20 steps, then the same 640 sequences at EVAL_PDP=4 (40 batches): eval time and memory (A7)
# Rounds (default: det time for preln, time for hc and attnres; a -det, -time or -dettime suffix sets them): det
# (XLA_FLAGS --xla_gpu_exclude_nondeterministic_ops=true, 40 steps: bitwise comparison between arms); time (default
# flags, 40 steps: step time); prof (-prof only: no command buffers, HLO dumped,
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
case $knobs in *-det) rounds="det"; knobs=${knobs%-det};; *-time) rounds="time"; knobs=${knobs%-time};; *-dettime) rounds="det time"; knobs=${knobs%-dettime};; *-prof) rounds="prof"; knobs=${knobs%-prof};; esac
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
  xoh) ON="LEVANTER_CE_XLA_ONEHOT=1";;
  cbw) ON="XLA_CMD_BUFFER_WHILE=1";;
  fa4) ON="ATTN_BACKEND=fa4";;
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
BASE="SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=0 PARAM_MAP=vocab NUM_GPUS=4 $A SMOKE_STEPS=40"
echo "arm $NAME: design $design, knobs [$ON], rounds $ROUNDS, GPUs ${CUDA_VISIBLE_DEVICES:-all}"; nvidia-smi -L || true
for r in $ROUNDS; do
  echo "round $r $(date +%T)"
  xf=; pr=
  case $r in
    det) xf="--xla_gpu_exclude_nondeterministic_ops=true";;
    time) pr=;;   # no profile here: with command buffers on, jax.profiler.stop_trace hung and left GPUs idle (w237/w238, 2026-10-26)
    # prof: no command buffers (each kernel then shows in the trace under its own HLO instruction) and the optimized HLO
    # dumped, for trace_families.py (the Tokenizer Size session's speed-all 7b2eba3a51); its step time is not the measure
    prof) pr="PROFILE=20:2"; xf="--xla_gpu_enable_command_buffer= --xla_dump_to=/scratch/gpfs/GROUP/USER/tmp/pls_speed/hlo_${NAME}_${JID} --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train_step.*";;
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
