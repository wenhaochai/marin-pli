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
# whatever it returns; par: start in the background; a line "wait" waits for the background tasks. Each task gets only its
# own VAR=value on top of the job's environment, plus PAIR_LOG_TAG=-<task>. Exit code: 0 only if every task passed.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
[ -f "${PLAN:-}" ] || { echo "set PLAN to a plan file"; exit 2; }
PT=/scratch/gpfs/GROUP/USER/tools/packed_task.sh
failed=() ; bg=() ; bgname=()
waitall() { for i in "${!bg[@]}"; do wait "${bg[$i]}" || failed+=("${bgname[$i]}"); echo "TASK ${bgname[$i]} END $(date +%H:%M:%S)"; done; bg=(); bgname=(); }
while read -r mode task run kind gpus limit script vars; do
  { [ -z "$mode" ] || [ "${mode:0:1}" = "#" ]; } && continue
  [ "$mode" = wait ] && { waitall; continue; }
  args=(); for v in $vars; do args+=("${v//+/ }"); done
  echo "TASK $task ($mode, $run, $kind, $gpus GPUs, limit $limit) START $(date +%H:%M:%S): $script [${args[*]}]"
  cmd=("$PT" "$task" "$run" "$kind" "$gpus" "$limit" -- scripts/della/pgroup.sh env PAIR_LOG_TAG="-$task" "${args[@]}" bash "$script")
  case $mode in
    smoke) "${cmd[@]}"; rc=$?; echo "TASK $task END rc=$rc $(date +%H:%M:%S)"; [ $rc -eq 0 ] || { echo "smoke failed: no training"; exit $rc; } ;;
    seq)   "${cmd[@]}"; rc=$?; echo "TASK $task END rc=$rc $(date +%H:%M:%S)"; [ $rc -eq 0 ] || failed+=("$task") ;;
    par)   "${cmd[@]}" & bg+=($!); bgname+=("$task") ;;
    *)     echo "unknown mode $mode"; exit 2 ;;
  esac
done < "$PLAN"
waitall
echo "PACKED DONE failed=[${failed[*]:-}] $(date +%H:%M:%S)"
[ ${#failed[@]} -eq 0 ]
