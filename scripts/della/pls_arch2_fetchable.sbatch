#!/bin/bash
#SBATCH --job-name=arch2
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96   # 8-GPU jobs ask for all 96 CPUs: priority JobSize counts CPUs
#SBATCH --mem-per-gpu=32G
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
#FETCHABLE: self-contained (the run's knobs below come from the job name, nothing from the submit shell)

# The Lean session's L1 / L2 follow-ups at 48 layers (owner popup 2026-10-21: 48 layers, only the new arms). Each is an
# ordinary model with probes (PLS_DETACH_BB=1), the knobs of its control in pls_d48_g4_submit.sh plus one variant knob;
# the controls ran two to a node on 4 GPUs each; preln-gilns runs alone on 8 GPUs with the same global batch, hc and
# attnres on 4 like their controls.
#   arch2-preln-gilns     Pre-LN whose pre-norm scales start at lns's 1/sqrt(l)   (L1; controls: -apreln-, -alns-)
#   arch2-hc-static       HC with the dynamic terms dropped                      (L2; control: -ahc-)
#   arch2-attnres-static  Full AttnRes with per-sublayer source logits            (L2; control: -aattnres-)
# Resumable: submit each as an afterany chain with --comment=fetchable-partial; a later job of a finished run exits at
# once. Runs pls_h100x8_single.sbatch (smoke copy with the readout gate, then the run; ledger, SUCCESS skip, crash marker).
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
if [ -n "${FETCHED_JOB_ID:-}" ]; then NAME=${FETCHED_JOB_NAME:-$(sacct -X -n -j "$FETCHED_JOB_ID" -o JobName%80 | head -1 | tr -d ' ')}; else NAME=$SLURM_JOB_NAME; fi
D48="SIZE=130m LAYERS=48 PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=1 PARAM_MAP=vocab"
# hc and attnres on 4 GPUs (submit with --gpus 4): on 8 their first train step aborted in XLA (a device thread missed the
# first AllToAll rendezvous for 40 s, wq w16 and w17 on two nodes, 2026-10-22). 4 GPUs is also their controls' layout.
case $NAME in
  arch2-preln-gilns) S="$D48 NUM_GPUS=8 PDP=4 ARCH=preln GAIN_INIT=lns";;
  arch2-hc-static) S="$D48 NUM_GPUS=4 PDP=4 ARCH=hc ROUTE=static";;          # PARAM_MAP runs hc at microbatch 1, as its control
  arch2-attnres-static) S="$D48 NUM_GPUS=4 PDP=2 ARCH=attnres ROUTE=static";;  # likewise attnres
  *) echo "unknown arch2 job name '$NAME'"; exit 2;;
esac
export SINGLE=$S SINGLE_SMOKE_MIN=45 SINGLE_FIRST_STEP_MIN=40
exec bash scripts/della/pls_h100x8_single.sbatch
