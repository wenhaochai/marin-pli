#!/bin/bash
# One arm of the REMAT A/B (branch speed-save-flash-residuals-remat): the 300m trainer from THIS worktree (its
# experiments/, lib/levanter, lib/haliax ahead of the main repo's editable installs; the main repo's .venv), the same
# setup as scripts/della/muonh_qwen3_h100x4.sbatch otherwise. Knobs come from the caller: SIZE, VOCAB_K, SEQ_LEN, SEED,
# REMAT (full | flash), RUN_TAG (must start with -rematab, so the run id is a scratch id), and either SMOKE_STEPS or
# AB_MINUTES (stop the real-schedule run after this many minutes; exit 0 then). At the end it deletes this arm's output
# dir and temporary checkpoints (only a run id that contains -rematab); the metrics stay in W&B (or the offline W&B dir).
set -uo pipefail
WT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/save-flash-residuals-remat
M=/scratch/gpfs/GROUP/USER/project/marin
PY=$M/.venv/bin/python
cd "$WT"
source "$M/scripts/della/job_tmpdir.sh"
module load proxy/default  # wandb only
source "$M/scripts/della/wandb_guard.sh"
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della
export XLA_PYTHON_CLIENT_MEM_FRACTION=${JOB_XLA_MEM_FRACTION:-0.9}
export PYTHONPATH=$WT/lib/levanter/src:$WT/lib/haliax/src:$WT
export MEMSTATS_EVERY=${MEMSTATS_EVERY:-300}
case "${RUN_TAG:-}" in -rematab*) ;; *) echo "arm.sh: RUN_TAG='${RUN_TAG:-}' must start with -rematab"; exit 2 ;; esac
case "${REMAT:-}" in full|flash) ;; *) echo "arm.sh: REMAT='${REMAT:-}' must be full or flash"; exit 2 ;; esac
[ -n "${SMOKE_STEPS:-}" ] || [ -n "${AB_MINUTES:-}" ] || { echo "arm.sh: set SMOKE_STEPS or AB_MINUTES"; exit 2; }
got=$("$PY" -c 'import levanter, haliax; print(levanter.__file__, haliax.__file__)')
case "$got" in "$WT/lib/levanter/src/"*" $WT/lib/haliax/src/"*) echo "arm.sh: levanter/haliax from the worktree: $got" ;;
  *) echo "arm.sh: wrong levanter/haliax: $got"; exit 2 ;; esac
run_id=$(DRY_RUN=1 MEMSTATS_EVERY=0 "$PY" -m experiments.references.della_muonh_qwen3_scaling 2>/dev/null | sed -n 's#^output: speedrun/\([^/]*\)/.*#\1#p' | head -1)
case "$run_id" in *-rematab*) ;; *) echo "arm.sh: dry run gave run id '$run_id'"; exit 2 ;; esac
echo "arm.sh: run $run_id REMAT=$REMAT $(date +%H:%M:%S)"
nvidia-smi -L
if [ -n "${AB_MINUTES:-}" ]; then
  timeout --signal=TERM --kill-after=2m "${AB_MINUTES}m" "$PY" -m experiments.references.della_muonh_qwen3_scaling; rc=$?
  { [ $rc -eq 124 ] || [ $rc -eq 137 ]; } && { echo "arm.sh: stopped after ${AB_MINUTES} min as planned"; rc=0; }
else
  "$PY" -m experiments.references.della_muonh_qwen3_scaling; rc=$?
fi
for d in "$MARIN_PREFIX/speedrun/$run_id" "$MARIN_PREFIX/tmp/checkpoints-temp$MARIN_PREFIX/speedrun/$run_id"; do
  case "$d" in */speedrun/*-rematab*) rm -rf "$d" ;; esac   # only this arm's scratch run (never an empty or foreign id)
done
echo "arm.sh: END rc=$rc $(date +%H:%M:%S)"
exit $rc
