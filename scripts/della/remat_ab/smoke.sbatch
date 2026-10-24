#!/bin/bash
#SBATCH --job-name=rematab-smoke
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=24G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out

# REMAT smoke and timing on 4xH100 (pli-cp, smoke only): 300m v64k, SMOKE_STEPS steps per arm, REMAT=full then flash, at
# SEQ_LEN 4224 (charwin, flash block 128) and 4096. Each arm runs scripts/della/remat_ab/arm.sh from the worktree. One
# SUMMARY line per arm: exit code, seconds per step between the first progress line at or after step 20 and the last one,
# MEMSTATS peak, last loss. Exit 0 only if every arm passed. Nothing comes from the submitting shell.
set -uo pipefail
WT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/save-flash-residuals-remat
LOGDIR=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm
unset REMAT RUN_TAG SMOKE_STEPS AB_MINUTES SEED SEQ_LEN VOCAB_K SIZE VARIANT DATA_EPOCHS PASS_DATA BATCH PDP TOTAL_STEPS INIT_FROM
STEPS=120
fail=0
for SL in 4224 4096; do
  for R in full flash; do
    log=$LOGDIR/rematab-smoke_${SLURM_JOB_ID}_sl${SL}_${R}.log
    SIZE=300m VOCAB_K=64000 SEQ_LEN=$SL REMAT=$R SMOKE_STEPS=$STEPS MEMSTATS_EVERY=30 RUN_TAG=-rematab-smoke-j${SLURM_JOB_ID}-$R \
      timeout --signal=TERM --kill-after=1m 8m bash "$WT/scripts/della/remat_ab/arm.sh" > "$log" 2>&1
    rc=$?
    [ $rc -eq 0 ] || fail=1
    rate=$(tr '\r' '\n' < "$log" | grep -a "Progress on:train" | awk '{split($2,t,":"); s=t[1]*3600+t[2]*60+t[3]; n=$7; sub(/it\/.*/,"",n);
      if (n ~ /k$/) {sub(/k$/,"",n); n*=1000}; n+=0; if (n>=20 && !a) {a=1; s0=s; n0=n}; if (a) {s1=s; n1=n}}
      END {if (a && n1>n0) printf "%.3f s/step over steps %d-%d", (s1-s0)/(n1-n0), n0, n1; else print "n/a"}')
    peak=$(grep -a MEMSTATS "$log" | sed -n 's/.*peak_bytes_in_use=\([0-9.]*\).*/\1/p' | sort -n | tail -1)
    loss=$(tr '\r' '\n' < "$log" | grep -a "Progress on:train" | tail -1 | sed -n 's/.*postfix:\(.*\)/\1/p')
    err=$(grep -a -m1 -E "RESOURCE_EXHAUSTED|out of memory|Traceback" "$log" | cut -c1-160)
    echo "SUMMARY sl=$SL remat=$R rc=$rc rate=[$rate] peak_GiB=${peak:-n/a} last=[$loss] err='$err' log=$log"
  done
done
echo "SMOKE DONE $(date +%H:%M:%S) fail=$fail"
exit $fail
