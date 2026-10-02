#!/bin/bash
# Event stream for the pls jobs: state changes, first train-progress line, eval lines (main + per-layer), errors, smoke SUMMARY.
L=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm
declare -A st seen first
STATE=/scratch/gpfs/GROUP/USER/tmp/pls/watch_pls.state
[ -f "$STATE" ] && source "$STATE"   # survives re-arming: only new events are emitted
save() { { declare -p st seen first; } > "$STATE.tmp" 2>/dev/null && sed -i 's/^declare -A/declare -gA/' "$STATE.tmp" && mv "$STATE.tmp" "$STATE"; }
JOBS="14728893:pls-smoke 14732138:pls1-130m 14732139:pls1-300m 14732140:pls1-300m 14767382:pls-readout-eval 14769012:pls1dh-130m 14769013:pls1dh-300m 14769014:pls1dh-300m 14769391:pls1sep-130m 14808812:pls-probe 14808813:pls-probe 14808814:pls-probe 14832726:pls-off 14832727:pls-off 14832728:pls-off"
while true; do
  alive=0
  for jn in $JOBS; do
    j=${jn%%:*}; n=${jn##*:}
    s=$(sacct -j $j -X -n -o State%20 2>/dev/null | head -1 | tr -d ' ' || true)
    if [ -n "$s" ] && [ "$s" != "${st[$j]:-}" ]; then echo "[$n $j] state=$s $(date +%m-%d_%H:%M)"; st[$j]=$s; fi
    case "$s" in PENDING|RUNNING|REQUEUED|"") alive=1;; esac
    f=$L/${n}_${j}.out
    [ -f "$f" ] || continue
    if [ -z "${first[$j]:-}" ]; then
      p=$(grep -a -m1 -o "Progress on:train [^ ]* rate:[^ ]*" "$f" 2>/dev/null)
      [ -n "$p" ] && { echo "[$n $j] first progress: $p"; first[$j]=1; }
    fi
    pat='levanter.eval eval loss|paloma macro loss|eval/L[0-9]+ loss|^SUMMARY|^SMOKE|^PROBE|bad run id|Traceback|RESOURCE_EXHAUSTED|out of memory|FloatingPointError|DUE TO TIME LIMIT|CANCELLED AT|oom-kill|Killed'
    c=$(grep -a -c -E "$pat" "$f" 2>/dev/null || true)
    if [ "${c:-0}" -gt "${seen[$j]:-0}" ]; then
      grep -a -E "$pat" "$f" | tail -n $((c - ${seen[$j]:-0})) | sed -E "s/^[IWE][0-9]+ [0-9:]+ [0-9]+ //; s/^/[$n $j] /" | cut -c1-220
      seen[$j]=$c
    fi
  done
  save
  [ $alive -eq 0 ] && { echo "all pls jobs terminal"; break; }
  sleep 150
done
