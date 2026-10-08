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
# logs/slurm/<job>_<jobid>-<task>.log. Mode lane=<name>: tasks of one lane run one after another in the background (with
# the retry), lanes side by side, so a failed run frees its GPUs for the lane's next task; "wait" also starts the lanes.
# A task with a run (not "-") trains only if the dry run under its own VAR=value gives that run id (run_checked.sh).
# Exit code: 0 only if every task passed. No set -e: a failed task is handled here.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
JOB_TMP_TAG=job source scripts/della/job_tmpdir.sh   # TMPDIR under scratch (never the shared /tmp); its tasks make their own
# subdirectories in $JOBTMP; the whole job directory goes when the job ends, leftovers of killed tasks included
trap 'case "$JOBTMP" in /scratch/gpfs/GROUP/USER/tmp/job/[0-9]*) rm -rf "$JOBTMP" ;; esac' EXIT
# experiment knobs never come from the submitting shell (--export=ALL): each task sets its own
unset SIZE SIZES VARIANT VARIANTS VOCAB_K DATA_EPOCHS SEED BATCH SEQ_LEN RUN_TAG CV_SIDE CV_K NUM_GPUS TOTAL_STEPS INIT_FROM \
      SMOKE_STEPS SMOKE_ENVS PRECISION TIE EMA_BETA DEVICE_TAG PAIR_A PAIR_B PDP SS_SCHEDULE OV_M OV_OD_W OD_M FOCAL_GAMMA
[ -f "${PLAN:-}" ] || { echo "set PLAN to a plan file"; exit 2; }
PT=/scratch/gpfs/GROUP/USER/tools/packed_task.sh
failed=() ; declare -A bgname=() ; declare -A lanes=()
FAILFILE=$(mktemp)   # lanes run in subshells: they report failed tasks here
waitall() {   # start the lanes, then wait -n: report each background task or lane the moment it ends
  local pid rc name
  for name in "${!lanes[@]}"; do
    ( lfail=0
      while IFS= read -r line; do
        [ -n "$line" ] || continue
        eval "set -- $line"
        t=$1; shift
        echo "TASK $t (lane $name) START $(date +%H:%M:%S)"
        retry_once "$t" "$@"; r=$?
        echo "TASK $t (lane $name) END rc=$r $(date +%H:%M:%S)"
        [ $r -eq 0 ] || { echo "$t" >> "$FAILFILE"; lfail=1; }
      done <<< "${lanes[$name]}"
      exit $lfail ) &
    bgname[$!]="lane $name"
  done
  lanes=()
  while [ ${#bgname[@]} -gt 0 ]; do
    pid=""
    wait -n -p pid "${!bgname[@]}"; rc=$?
    if [ -z "$pid" ]; then echo "wait -n returned without a pid (rc=$rc)"; break; fi
    echo "${bgname[$pid]} END rc=$rc $(date +%H:%M:%S)"
    case "${bgname[$pid]}" in lane*) ;; *) [ $rc -eq 0 ] || failed+=("${bgname[$pid]}") ;; esac
    unset "bgname[$pid]"
  done
}
retry_once() {   # a par or lane task: one more attempt after a failure that is not its time limit (124/137), a Slurm stop (143) or a run-id mismatch (3)
  local task=$1; shift
  "$@"; local rc=$?
  if [ $rc -ne 0 ] && [ $rc -ne 124 ] && [ $rc -ne 137 ] && [ $rc -ne 143 ] && [ $rc -ne 3 ]; then echo "TASK $task failed rc=$rc, retrying once $(date +%H:%M:%S)"; "$@"; rc=$?; fi
  return $rc
}
while read -r mode task run kind gpus limit script vars; do
  { [ -z "$mode" ] || [ "${mode:0:1}" = "#" ]; } && continue
  [ "$mode" = wait ] && { waitall; continue; }
  args=(); for v in $vars; do args+=("${v//+/ }"); done
  echo "TASK $task ($mode, $run, $kind, $gpus GPUs, limit $limit) START $(date +%H:%M:%S): $script [${args[*]}]"
  log=logs/slurm/${SLURM_JOB_NAME:-packed}_${SLURM_JOB_ID:-nojob}-$task.log
  if [ "$run" != "-" ]; then inner=(scripts/della/run_checked.sh "$run" "$script" PAIR_LOG_TAG="-$task" "${args[@]}")
  else inner=(env PAIR_LOG_TAG="-$task" "${args[@]}" bash "$script"); fi
  cmd=("$PT" "$task" "$run" "$kind" "$gpus" "$limit" -- scripts/della/pgroup.sh bash -c 'exec "$@" >> "$0" 2>&1' "$log" "${inner[@]}")
  case $mode in
    smoke) "${cmd[@]}"; rc=$?; echo "TASK $task END rc=$rc $(date +%H:%M:%S)"; [ $rc -eq 0 ] || { echo "smoke failed: no training"; exit $rc; } ;;
    seq)   "${cmd[@]}"; rc=$?; echo "TASK $task END rc=$rc $(date +%H:%M:%S)"; [ $rc -eq 0 ] || failed+=("$task") ;;
    par)   retry_once "$task" "${cmd[@]}" & bgname[$!]=$task ;;
    lane=*) lanes[${mode#lane=}]+="$(printf '%q ' "$task" "${cmd[@]}")"$'\n' ;;
    *)     echo "unknown mode $mode"; exit 2 ;;
  esac
done < "$PLAN"
waitall
while IFS= read -r t; do [ -n "$t" ] && failed+=("$t"); done < "$FAILFILE"; rm -f "$FAILFILE"
echo "PACKED DONE failed=[${failed[*]:-}] $(date +%H:%M:%S)"
[ ${#failed[@]} -eq 0 ]
