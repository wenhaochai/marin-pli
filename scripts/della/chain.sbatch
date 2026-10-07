#!/bin/bash
#SBATCH --job-name=chain
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# Dependent jobs merged into one 8-GPU allocation (owner via the terminal controller, 2026-10-07: one queue wait instead
# of two). CHAIN names a file of steps, one per line: "<label> <minutes> <script> [VAR=value ...]" ('+' inside a value
# stands for a space), run in order with bash (their #SBATCH lines are ignored). Each step runs in its own process group
# with only its own VAR=value set on top of the job's environment, and is killed (TERM, then KILL after 60 s) at its
# time limit. A step labelled smoke* that fails ends the job before any training; any other failed step is recorded and
# the chain goes on. PAIR_LOG_TAG=-<label> keeps each pair's run logs apart. Exit code: 0 only if every step passed.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
[ -f "${CHAIN:-}" ] || { echo "set CHAIN to a steps file"; exit 2; }
failed=()
while read -r label minutes script vars; do
  { [ -z "$label" ] || [ "${label:0:1}" = "#" ]; } && continue
  args=()
  for v in $vars; do args+=("${v//+/ }"); done
  echo "STEP $label START $(date +%H:%M:%S) limit ${minutes} min: $script [${args[*]}]"
  setsid env PAIR_LOG_TAG="-$label" "${args[@]}" bash "$script" < /dev/null &
  pid=$!
  ( sleep $((minutes * 60)); echo "STEP $label TIMEOUT"; kill -TERM -- -$pid 2>/dev/null; sleep 60; kill -KILL -- -$pid 2>/dev/null ) &
  watcher=$!
  wait $pid; rc=$?
  kill $watcher 2>/dev/null; wait $watcher 2>/dev/null
  echo "STEP $label END rc=$rc $(date +%H:%M:%S)"
  if [ "$rc" -ne 0 ]; then
    [ "${label#smoke}" != "$label" ] && { echo "smoke failed: no training"; exit $rc; }
    failed+=("$label")
  fi
done < "$CHAIN"
echo "CHAIN DONE failed=[${failed[*]:-}] $(date +%H:%M:%S)"
[ ${#failed[@]} -eq 0 ]
