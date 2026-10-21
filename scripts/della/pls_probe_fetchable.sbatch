#!/bin/bash
#SBATCH --job-name=mtpl-probe
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96   # 8-GPU jobs ask for all 96 CPUs: priority JobSize counts CPUs
#SBATCH --mem-per-gpu=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
#FETCHABLE: self-contained (the run's knobs below come from the job name, nothing from the submit shell)

# Post-hoc probes of the MTP ladder (owner popup 2026-10-14; probes after training only, on the final checkpoint), as a
# job a host may run through tools/fetch_pending.sh (owner, 2026-10-16). Submit with --job-name=mtpl-probe-<size>-<arm>
# (--comment=fetchable, a right-sized --time); the knobs are set here from that name, so the job needs no variable from
# the submit shell, and it runs pls_h100x8_single.sbatch (smoke copy, then the probes; ledger, SUCCESS skip, crash marker).
# Fetched, FETCHED_JOB_ID is this job's id and SLURM_JOB_ID the host's: the name comes from sacct.
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls-mtp
if [ -n "${FETCHED_JOB_ID:-}" ]; then NAME=$(sacct -X -n -j "$FETCHED_JOB_ID" -o JobName%80 | head -1 | tr -d ' '); else NAME=$SLURM_JOB_NAME; fi
ROOT=/scratch/gpfs/GROUP/USER/marin_store_big/speedrun
B130="SIZE=130m NUM_GPUS=8 PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=1 PLS_OPT_SPLIT=1 PLS_FUT=1 TOTAL_STEPS=2000"
LAD="NUM_GPUS=8 PLS_W=0 PLS_EVAL=0 PDP=8"   # PDP 8: 24 probe heads keep every layer's output (the pls OOMs at 520m, 1_2b)
# 1_2b: 31 probe heads of 128256x2048. XLA's scheduler runs all 31 head backward loops before any gradient
# reduce-scatter, so 31 full f32 dW (1 GiB each) and 31 all-gathered bf16 heads (0.5 GiB each) live together:
# PDP 8 needed 70.2 GiB of a 71 GiB pool (60.09 GiB temp, smoke 15760472); PDP 2 is worse (121 GiB, 15832439).
# A tighter scheduler memory limit (slop 70% of 80 GB) makes it reduce-scatter each dW early: 68.2 -> 61.1 GiB at
# PDP 8 in a deviceless H100x8 compile of this train step (2026-10-20). Same probes, data and batch.
LAD12="NUM_GPUS=8 PLS_W=0 PLS_EVAL=0 PDP=8 XLA_FLAGS=--xla_gpu_memory_limit_slop_factor=70"
# 520m: under FSDP the train step with 24 layers and 47 heads computed wrong layer outputs (frozen head 9.7 nats against
# eval 3.55; 15738657/8, 2026-10-21), as the 48-layer runs did before PARAM_MAP=vocab (7b02c544f2). Layers whole on each GPU.
LAD5="$LAD PARAM_MAP=vocab"
case $NAME in
  mtpl-probe-130m-ntp) S="$B130 PROBE_FROM=$ROOT/muonh-qwen3-130m-della4xh100-pls1-sep-bbfrozen-os-fut/2026.09.13/checkpoints/step-4958";;
  mtpl-probe-130m-mtp) S="$B130 PLS_MTP_W=0.3 PROBE_FROM=$ROOT/muonh-qwen3-130m-della4xh100-pls1-sep-bbfrozen-os-mtp0.3-fut/2026.09.13/checkpoints/step-4958";;
  mtpl-probe-300m-ntp) S="SIZE=300m $LAD TOTAL_STEPS=2000 PROBE_FROM=$ROOT/muonh-qwen3-300m-della8xh100-pls0-noev/2026.09.13/checkpoints/step-11443";;
  mtpl-probe-300m-mtp) S="SIZE=300m $LAD PLS_MTP_W=0.3 TOTAL_STEPS=2000 PROBE_FROM=$ROOT/muonh-qwen3-300m-della8xh100-pls0-noev-mtp0.3/2026.09.13/checkpoints/step-11443";;
  # 520m and 1_2b train at batch 256: 1000 probe steps are the same 1B tokens as 2000 steps at batch 128
  mtpl-probe-520m-ntp) S="SIZE=520m $LAD5 TOTAL_STEPS=1000 PROBE_FROM=$ROOT/muonh-qwen3-520m-della8xh100-pls0-noev/2026.09.13/checkpoints/step-9917";;
  mtpl-probe-520m-mtp) S="SIZE=520m $LAD5 PLS_MTP_W=0.3 TOTAL_STEPS=1000 PROBE_FROM=$ROOT/muonh-qwen3-520m-della8xh100-pls0-noev-mtp0.3/2026.09.13/checkpoints/step-9917";;
  mtpl-probe-1_2b-ntp) S="SIZE=1_2b $LAD12 TOTAL_STEPS=1000 PROBE_FROM=$ROOT/muonh-qwen3-1_2b-della8xh100-pls0-noev/2026.09.13/checkpoints/step-22887";;
  mtpl-probe-1_2b-mtp) S="SIZE=1_2b $LAD12 PLS_MTP_W=0.3 TOTAL_STEPS=1000 PROBE_FROM=$ROOT/muonh-qwen3-1_2b-della8xh100-pls0-noev-mtp0.3/2026.09.13/checkpoints/step-22887";;
  # the same 130m NTP probes again, with their checkpoint kept (KEEP_PROBE_CKPT): the probe weights W_next / W_after of every
  # layer, for the Lean session's L5 test (owner popup 2026-10-21); RUN_TAG -keep keeps the id apart from the finished run
  mtpl-probe-130m-ntp-keep) S="$B130 RUN_TAG=-keep PROBE_FROM=$ROOT/muonh-qwen3-130m-della4xh100-pls1-sep-bbfrozen-os-fut/2026.09.13/checkpoints/step-4958"; export KEEP_PROBE_CKPT=1;;
  # the 130m separate heads at weight 1 again, for its final checkpoint (the logit lens needs one; owner popup 2026-10-17)
  pls-130m-sep-ckpt) S="SIZE=130m NUM_GPUS=8 PLS_W=1 PLS_SEP=1";;
  *) echo "unknown probe job name '$NAME'"; exit 2;;
esac
if [[ $S == *PROBE_FROM=* ]]; then ck=${S##*PROBE_FROM=}; [ -d "$ck" ] || { echo "missing checkpoint $ck"; exit 2; }; fi
export SINGLE=$S SINGLE_SMOKE_MIN=40
exec bash scripts/della/pls_h100x8_single.sbatch
