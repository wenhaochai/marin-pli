#!/bin/bash
#SBATCH --job-name=packed
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

# A packed job (packing rules, owner via the terminal controller, 2026-10-07): the steps of PLAN, each launched through
# /scratch/gpfs/GROUP/USER/tools/packed_task.sh (time limit + ledger logs/packed/<jobid>.tsv) and pgroup.sh (the
# limit stops the whole process group). PLAN lines: "<mode> <task> <run|-> <main|other> <gpus> <timeout> <script>
# [VAR=value ...]" ('+' in a value stands for a space); mode smoke: run, and end the job if it fails; seq: run and go on
# whatever it returns; par: start in the background, and run it once more if it fails before its limit (a training run
# resumes from its last checkpoint; a failed partner would otherwise leave its GPUs idle until Della's 90-minute idle
# cancel takes the whole job); a line "wait" waits for the background tasks with wait -n, logging each as it ends. Each
# task gets only its own VAR=value on top of the job's environment, plus PAIR_LOG_TAG=-<task>, and writes its own log
# logs/slurm/<job>_<jobid>-<task>.log. Exit code: 0 only if every task passed. No set -e: a failed task is handled here.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
[ -f "${PLAN:-}" ] || { echo "set PLAN to a plan file"; exit 2; }
PT=/scratch/gpfs/GROUP/USER/tools/packed_task.sh
failed=() ; declare -A bgname=()
waitall() {   # wait -n: report each background task the moment it ends
  local pid rc
  while [ ${#bgname[@]} -gt 0 ]; do
    wait -n -p pid "${!bgname[@]}"; rc=$?
    echo "TASK ${bgname[$pid]} END rc=$rc $(date +%H:%M:%S)"
    [ $rc -eq 0 ] || failed+=("${bgname[$pid]}")
    unset "bgname[$pid]"
  done
}
retry_once() {   # a par task: one more attempt after a failure that is not its time limit (124/137: timeout(1) fired)
  local task=$1; shift
  "$@"; local rc=$?
  if [ $rc -ne 0 ] && [ $rc -ne 124 ] && [ $rc -ne 137 ]; then echo "TASK $task failed rc=$rc, retrying once $(date +%H:%M:%S)"; "$@"; rc=$?; fi
  return $rc
}
while read -r mode task run kind gpus limit script vars; do
  { [ -z "$mode" ] || [ "${mode:0:1}" = "#" ]; } && continue
  [ "$mode" = wait ] && { waitall; continue; }
  args=(); for v in $vars; do args+=("${v//+/ }"); done
  echo "TASK $task ($mode, $run, $kind, $gpus GPUs, limit $limit) START $(date +%H:%M:%S): $script [${args[*]}]"
  log=logs/slurm/${SLURM_JOB_NAME:-packed}_${SLURM_JOB_ID:-nojob}-$task.log
  cmd=("$PT" "$task" "$run" "$kind" "$gpus" "$limit" -- scripts/della/pgroup.sh bash -c 'exec "$@" >> "$0" 2>&1' "$log" env PAIR_LOG_TAG="-$task" "${args[@]}" bash "$script")
  case $mode in
    smoke) "${cmd[@]}"; rc=$?; echo "TASK $task END rc=$rc $(date +%H:%M:%S)"; [ $rc -eq 0 ] || { echo "smoke failed: no training"; exit $rc; } ;;
    seq)   "${cmd[@]}"; rc=$?; echo "TASK $task END rc=$rc $(date +%H:%M:%S)"; [ $rc -eq 0 ] || failed+=("$task") ;;
    par)   retry_once "$task" "${cmd[@]}" & bgname[$!]=$task ;;
    *)     echo "unknown mode $mode"; exit 2 ;;
  esac
done < "$PLAN"
waitall
echo "PACKED DONE failed=[${failed[*]:-}] $(date +%H:%M:%S)"
[ ${#failed[@]} -eq 0 ]
