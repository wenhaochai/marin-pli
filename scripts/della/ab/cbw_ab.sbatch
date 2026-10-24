#!/bin/bash
#SBATCH --job-name=cbw-ab
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-gpu=24G
#SBATCH --time=01:50:00
#SBATCH --exclude=della-k14g3,della-k17g3
#SBATCH --comment=fetchable
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: every knob is set below; needs nothing from the submitting shell

# A/B of XLA_CMD_BUFFER_WHILE (speed-command-buffer-while) at the real recipe: the charwin-v64k-full config (300m, v64k,
# SEQ_LEN=4224, batch 128, seed 0, data_seed 42, the full 11444-step cosine schedule), two copies side by side on one
# node, each stopped after AB_STOP of wall time (~1600-2000 steps; the 1000-step Paloma eval included):
#   cbw-ab-base  GPUs 0-3  flag off  run muonh-qwen3-300m-della4xh100-v64k-sl4224-abbase
#   cbw-ab-cbw   GPUs 4-7  flag on   run muonh-qwen3-300m-della4xh100-v64k-sl4224-abcbw
# Both read the same data in the same order as the real run w31 (muonh-qwen3-300m-della4xh100-v64k-sl4224): RUN_TAG
# changes only the run id. So abbase vs w31 is the run-to-run (GPU nondeterminism) band, and abcbw is compared with both
# (scripts/della/ab/cbw_ab_compare.py on a vis node). Speed: same node, same time, same data.
# The stop at AB_STOP is the plan (exit 124 of a task is a pass); the job exits 0 when both tasks end with 0 or 124.
# The arms' checkpoints (final, temporary) are deleted at the end; eval_metrics.jsonl stays. Training code: the main
# checkout (unchanged); the launcher: this branch's copy, the only file that reads XLA_CMD_BUFFER_WHILE.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
JOB_TMP_TAG=job source scripts/della/job_tmpdir.sh
trap 'case "$JOBTMP" in /scratch/gpfs/GROUP/USER/tmp/job/[0-9]*) rm -rf "$JOBTMP" ;; esac' EXIT
unset XLA_FLAGS SIZE VOCAB_K SEQ_LEN DATA_EPOCHS SEED BATCH RUN_TAG VARIANT PASS_DATA PDP PRECISION TOTAL_STEPS INIT_FROM XLA_CMD_BUFFER_WHILE SMOKE_STEPS
AB=/scratch/gpfs/GROUP/USER/tmp/speed_wt/command-buffer-while/scripts/della
PT=/scratch/gpfs/GROUP/USER/tools/packed_task.sh
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
AB_STOP=85m
CONF=(SIZE=300m VOCAB_K=64000 SEQ_LEN=4224)
STEM=300m-della4xh100-v64k-sl4224

arm() {   # arm <task> <tag> <gpus> <flag>
  local task=$1 tag=$2 gpus=$3 flag=$4
  local log=logs/slurm/cbw-ab_${JID}-$task.log
  "$PT" "$task" "$STEM$tag" other 4 "$AB_STOP" -- scripts/della/pgroup.sh bash -c 'exec "$@" >> "$0" 2>&1' "$log" \
    scripts/della/run_checked.sh "$STEM$tag" "$AB/muonh_qwen3_h100x4.sbatch" PAIR_LOG_TAG=-$task \
    CUDA_VISIBLE_DEVICES=$gpus "${CONF[@]}" RUN_TAG=$tag XLA_CMD_BUFFER_WHILE=$flag
}
echo "CBW AB START $(date +%H:%M:%S) stop after $AB_STOP"
arm cbw-ab-base -abbase 0,1,2,3 0 & pb=$!
arm cbw-ab-cbw -abcbw 4,5,6,7 1 & pc=$!
wait $pb; rb=$?
wait $pc; rcb=$?
for t in base cbw; do
  log=logs/slurm/cbw-ab_${JID}-cbw-ab-$t.log
  echo "ARM $t rate(40-1000)=[$(bash $AB/ab/cbw_rate.sh $log 40 990)] rate(1000-end)=[$(bash $AB/ab/cbw_rate.sh $log 1010)] flags=[$(grep -a -m1 '^XLA_CMD_BUFFER_WHILE=1' $log)]"
done
bash $AB/ab/cbw_cleanup.sh muonh-qwen3-$STEM-abbase muonh-qwen3-$STEM-abcbw || true
ok=1; for r in $rb $rcb; do [ "$r" -eq 0 ] || [ "$r" -eq 124 ] || ok=0; done
echo "CBW AB DONE $(date +%H:%M:%S) base_rc=$rb cbw_rc=$rcb"
[ $ok = 1 ]
