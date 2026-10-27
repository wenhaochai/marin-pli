#!/bin/bash
#SBATCH --job-name=arch3
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=32G
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
#FETCHABLE: self-contained (the run's knobs below come from the job name, nothing from the submit shell)

# Internals follow-ups (owner popup 2026-10-22: mHC with H_res = I, AttnRes query Post Clip, HC depth sweep; the approved
# AttnRes null slot was dropped before submission: an RMSNorm follows every AttnRes read and cancels it). Each design runs twice, as its control in pls_d48_g4_submit.sh: separate heads (-sep: PLS_DETACH_BB=0)
# and the ordinary model with probes (-probes: PLS_DETACH_BB=1), each on 4 GPUs (wq puts two 4-GPU jobs on one host).
#   arch3-mhcid-{sep,probes}     mhc with the residual mixing fixed to I        (control -amhc-; survey B5: does rank
#                                                                                collapse of the Sinkhorn product make
#                                                                                mHC's separate heads pay full price?)
#   arch3-arqclip-{sep,probes}   Full AttnRes, pseudo-query logit scale <= 10   (control -aattnres-; A1: are the near-
#                                                                                binary reads (s >= 40) a temperature effect?)
#   arch3-hc-d{12,24}-{sep,probes}  HC at 12 / 24 layers                         (48 layers: separate heads pay 0.048 nats
#                                                                                at the last layer; 6 layers: 0.125)
#   arch3-preln-vis{50,25}-sep   Pre-LN separate heads whose intermediate readouts see only the first 1/2 or 1/4 of the
#                                hidden coordinates (PLS_VIS; owner popup 2026-10-23: is the cost the readouts seeing the
#                                whole state?); control: the -apreln- probes run
#   arch3-preln-s1-{sep,probes}  Pre-LN seed 1 (seed noise of the separate-heads cost)
#   arch3-arvnorm-{sep,probes}   Full AttnRes whose sources enter every mix at unit RMS (AR_VNORM=1; owner popup 2026-10-24:
#                                the ordinary models' largest source is 1,511x to 52,810x the median and their loss spikes;
#                                control -aattnres-)
#   arch3-preln-d{12,24}-{sep,probes}  Pre-LN at 12 / 24 layers (owner popup 2026-10-25: is the cost falling with depth
#                                Hyper-Connections' alone? HC 12 / 24 / 48 layers: 0.121 / 0.077 / 0.048)
#   arch3-hcstatic-sep, arch3-arstatic-sep  separate heads with static routing (owner popup 2026-10-25); their probes runs
#                                are the L2 runs wq w27 / w28 (run ids -sep-bbfrozen-a{hc,attnres}-rstatic-vs-d48)
#   arch3-preln-w47-sep          Pre-LN separate heads, every intermediate loss at weight 1/47 (they sum to 1; owner popup
#                                2026-10-25: at weight 1 their gradient is 41x the last layer's, cos 0.96; is the cost its size?);
#                                control: the -apreln- probes run (a probe's weight does not reach the backbone)
#   arch3-hc-s1-{sep,probes}     HC 48 layers seed 1 (owner popup 2026-10-25: HC's low cost rests on one seed)
#   arch3-preln-vis125-sep       Pre-LN readouts see 1/8 of the coordinates (owner popup 2026-10-25: extends E1)
#   arch3-hc{1,2}-{sep,probes}  HC 48 layers with 1 / 2 residual streams instead of 4 (owner popup 2026-10-25: is HC's low
#                                cost its extra streams or its token-dependent gates? 1 stream = Pre-LN plus HC's gates)
#   arch3-hc8-{sep,probes}       HC 48 layers with 8 streams (owner 2026-10-25, queue fill: the stream-count curve 1/2/4/8)
#   arch3-arblock24-{sep,probes} AttnRes Block with 24 blocks instead of 8 (owner 2026-10-25, queue fill: does the cost fall
#                                with the number of sources, from Block 0.113 toward Full?)
#   arch3-hc-newstack-{sep,probes} HC 48 layers on the new stack (b-tiled CE on the 4096-token shards with readouts one at a
#                                a time, lse kernel, fa4thd; owner popup 2026-10-26 ~16:40): does HC's cost (0.048 on the
#                                streaming CE) hold, and may new HC / AttnRes pairs run twice as fast?
#   arch3-hcrwsig-{sep,probes}   HC with mHC's positive, bounded read / write maps (owner popup 2026-10-26 ~16:40): does
#                                that constraint explain mHC's full cost (0.10) against HC's half (0.048)?
#   arch3-mhcrwlin-{sep,probes}  mHC with HC's signed, unbounded read / write maps (owner popup 2026-10-27, the mirror of
#                                hcrwsig): does a signed read/write lower mHC's cost (0.10) toward HC's (0.048)?
#   arch3-preln-fa4thd-{sep,probes} the same check on the new stack: FA4 THD attention plus the live lse kernel (owner popup
#                                2026-10-26 ~06:40: replaces the fa4 pair w261/w262 before it started)
#   arch3-preln-fa4-{sep,probes} Pre-LN 48 layers with FA4 attention (owner popup 2026-10-26: validate FA4 before new pairs
#                                use it: the cost must match the JAX-flash 0.089 / 0.092 (seeds 0 / 1) within seed noise)
#   arch3-preln-{jax,fa4}rerun-probes  reruns of the Pre-LN ordinary model with probes, seed 0 (its init and data order),
#                                on today's stack (CE side on, PLS_FOLD_NORM=0), with JAX flash or fa4thd attention
#                                (2026-10-27): w334 (fa4thd) ends 0.005 above the JAX-flash seeds (2.9883 against 2.9832 /
#                                2.9826) while its separate-heads twin does not move, and early traces cannot split it (two
#                                runs of identical code already differ by +-0.02 at steps 20-300, wq w495). The CE side
#                                moves gradients by 1e-6 to 1e-5 (w306), inside the GEMM-autotune spread of identical runs
#                                (w205 vs w229), so the JAX rerun is a third control sample; RUN_TAG -rerun keeps run ids apart
#   arch3-preln-off{50,80}-sep, arch3-hc-off80-sep  separate heads whose intermediate losses stop at 50% / 80% of training
#                                (PLS_OFF=2480 / 3967, a switch; the readouts are skipped after it; owner popup 2026-10-27):
#                                does the separate-heads cost at the last layer stay once the intermediate losses stop?
#                                Controls: the seed-0 separate-heads and probes runs of each design. Pre-flight: wq
#                                pls-off-test (PLS_OFF=15 over 40 steps with a mid eval and checkpoint)
# Resumable: submit with --partial. Runs pls_h100x8_single.sbatch (smoke copy with the readout gate, then the run).
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls
if [ -n "${FETCHED_JOB_ID:-}" ]; then NAME=${FETCHED_JOB_NAME:-$(sacct -X -n -j "$FETCHED_JOB_ID" -o JobName%80 | head -1 | tr -d ' ')}; else NAME=$SLURM_JOB_NAME; fi
case $NAME in
  *-sep) BB=0; DESIGN=${NAME%-sep};;
  *-probes) BB=1; DESIGN=${NAME%-probes};;
  *) echo "arch3 job name '$NAME' must end in -sep or -probes"; exit 2;;
esac
base="SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=$BB PARAM_MAP=vocab NUM_GPUS=4"
case $DESIGN in
  arch3-mhcid) S="$base LAYERS=48 PDP=4 ARCH=mhc MHC_RES=identity";;
  arch3-arqclip) S="$base LAYERS=48 PDP=2 ARCH=attnres AR_QCLIP=10";;
  arch3-hc-d12) S="$base LAYERS=12 PDP=4 ARCH=hc";;
  arch3-hc-d24) S="$base LAYERS=24 PDP=4 ARCH=hc";;
  arch3-preln-vis50) S="$base LAYERS=48 PDP=4 ARCH=preln PLS_VIS=0.5";;
  arch3-preln-vis25) S="$base LAYERS=48 PDP=4 ARCH=preln PLS_VIS=0.25";;
  arch3-preln-s1) S="$base LAYERS=48 PDP=4 ARCH=preln SEED=1";;
  arch3-arvnorm) S="$base LAYERS=48 PDP=2 ARCH=attnres AR_VNORM=1";;
  arch3-preln-d12) S="$base LAYERS=12 PDP=4 ARCH=preln";;
  arch3-preln-d24) S="$base LAYERS=24 PDP=4 ARCH=preln";;
  arch3-hcstatic) S="$base LAYERS=48 PDP=4 ARCH=hc ROUTE=static";;
  arch3-arstatic) S="$base LAYERS=48 PDP=2 ARCH=attnres ROUTE=static";;
  arch3-preln-w47) S="$base LAYERS=48 PDP=4 ARCH=preln PLS_W=0.0212766";;   # the last PLS_W wins in env
  arch3-hc-s1) S="$base LAYERS=48 PDP=4 ARCH=hc SEED=1";;
  arch3-preln-vis125) S="$base LAYERS=48 PDP=4 ARCH=preln PLS_VIS=0.125";;
  arch3-hc1) S="$base LAYERS=48 PDP=4 ARCH=hc HC_STREAMS=1";;
  arch3-hc2) S="$base LAYERS=48 PDP=4 ARCH=hc HC_STREAMS=2";;
  arch3-hc8) S="$base LAYERS=48 PDP=4 ARCH=hc HC_STREAMS=8";;
  arch3-arblock24) S="$base LAYERS=48 PDP=2 ARCH=attnres_block AR_BLOCKS=24";;
  arch3-preln-fa4) S="$base LAYERS=48 PDP=4 ARCH=preln ATTN_BACKEND=fa4";;
  arch3-preln-jaxrerun) S="$base LAYERS=48 PDP=4 ARCH=preln PLS_FOLD_NORM=0 RUN_TAG=-rerun";;
  arch3-preln-fa4rerun) S="$base LAYERS=48 PDP=4 ARCH=preln ATTN_BACKEND=fa4thd PLS_FOLD_NORM=0 RUN_TAG=-rerun";;
  arch3-preln-off50) S="$base LAYERS=48 PDP=4 ARCH=preln PLS_OFF=2480";;
  arch3-preln-off80) S="$base LAYERS=48 PDP=4 ARCH=preln PLS_OFF=3967";;
  arch3-hc-off80) S="$base LAYERS=48 PDP=4 ARCH=hc PLS_OFF=3967";;
  arch3-preln-fa4thd) S="$base LAYERS=48 PDP=4 ARCH=preln ATTN_BACKEND=fa4thd";;
  arch3-hc-newstack) S="$base LAYERS=48 PDP=4 ARCH=hc ATTN_BACKEND=fa4thd LEVANTER_CE_BTILED_MIN_BATCH=4096 PLS_SERIAL=1 LEVANTER_CE_GIF_BARRIER=1";;
  arch3-hcrwsig) S="$base LAYERS=48 PDP=4 ARCH=hc HC_RW=sigmoid";;
  arch3-mhcrwlin) S="$base LAYERS=48 PDP=4 ARCH=mhc MHC_RW=linear";;
  *) echo "unknown arch3 design '$DESIGN'"; exit 2;;
esac
export SINGLE=$S SINGLE_SMOKE_MIN=45 SINGLE_FIRST_STEP_MIN=40
exec bash scripts/della/pls_h100x8_single.sbatch
