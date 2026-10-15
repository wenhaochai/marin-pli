#!/bin/bash
# MTP across the ladder (owner, 2026-10-14: four sizes; probes after training only, on the final checkpoint).
# Training: 300m, 520m, 1_2b, each an NTP run and an NTP + MTP run (one DeepSeek-V3 MTP module, weight 0.3), as the
# baseline's training (PLS_W=0: stride-16 stop-gradient monitor readouts, no per-layer eval), one 8-GPU job each with its
# own smoke (pls_h100x8_single.sbatch). Estimated training time on 8 GPUs: half the 4-GPU baselines' (300m 4.6 h,
# 520m ~18 h, 1_2b ~55 h), times ~1.5 / 1.3 / 1.3 with MTP (one more decoder layer and lm_head; 130m measured +0.5 h of
# ~1 h). Each run gets one follow-up job (afterany) that resumes it if the first ran out of time and exits at once if it
# finished; 1_2b gets two 24 h jobs and the follow-up.
# 130m: the Figure 16 runs are the ladder's 130m point (their probes are detached, so the backbones train as the
# baseline); `--probe130` submits their post-hoc probes (2000 steps), the calibration against their in-training probes.
# Usage: scripts/della/pls_mtp_ladder_submit.sh [--dry] [--train] [--probe130]
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls-mtp
DRY=0; TRAIN=0; P130=0
for a in "$@"; do case $a in --dry) DRY=1;; --train) TRAIN=1;; --probe130) P130=1;; *) echo "unknown $a"; exit 2;; esac; done
S=scripts/della/pls_h100x8_single.sbatch
sub() {   # $1 name, $2 --time (h:mm:ss), $3 SINGLE spec, $4 dependency (or empty); prints the job id
  local cmd=(env SINGLE="$3" SINGLE_SMOKE_MIN=40 sbatch --parsable --job-name="$1" --time="$2" ${4:+--dependency=afterany:$4} "$S")
  if [ $DRY = 1 ]; then echo "${cmd[*]}" >&2; echo "DRY$RANDOM"; else "${cmd[@]}"; fi
}
if [ $TRAIN = 1 ]; then
  for arm in ntp mtp; do
    m=""; [ $arm = mtp ] && m=" PLS_MTP_W=0.3"
    for spec in "300m 3:30:00 4:00:00" "520m 14:00:00 12:00:00" "1_2b 24:00:00 24:00:00 24:00:00"; do
      read -r size times <<< "$spec"; dep=""; ids=""
      for t in $times; do
        [ $arm = mtp ] && [ "$size" = 300m ] && [ "$t" = 3:30:00 ] && t=4:30:00
        [ $arm = mtp ] && [ "$size" = 520m ] && [ "$t" = 14:00:00 ] && t=18:00:00
        dep=$(sub "mtpl-$size-$arm" "$t" "SIZE=$size NUM_GPUS=8 PLS_W=0 PLS_EVAL=0$m" "$dep"); ids="$ids $dep"
      done
      echo "mtpl $size $arm:$ids"
    done
  done
fi
if [ $P130 = 1 ]; then
  base="SIZE=130m NUM_GPUS=8 PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=1 PLS_OPT_SPLIT=1 PLS_FUT=1 TOTAL_STEPS=2000"
  root=/scratch/gpfs/GROUP/USER/marin_store_big/speedrun
  for arm in ntp mtp; do
    m=""; tag=-os-fut; [ $arm = mtp ] && m=" PLS_MTP_W=0.3" && tag=-os-mtp0.3-fut
    ck=$root/muonh-qwen3-130m-della4xh100-pls1-sep-bbfrozen$tag/2026.09.13/checkpoints/step-4958
    [ -d "$ck" ] || { echo "missing $ck"; exit 2; }
    echo "probe 130m $arm: $(sub "mtpl-probe-130m-$arm" 2:00:00 "$base$m PROBE_FROM=$ck" "")"
  done
fi
