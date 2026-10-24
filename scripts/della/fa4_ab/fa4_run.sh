#!/bin/bash
# One 300m run of the FA4 attention A/B (branch speed-fa4-segmented-attention-backend), on the GPUs it is given
# (CUDA_VISIBLE_DEVICES from the plan line, or a whole 4-GPU allocation). Runs THIS worktree's code: the experiment
# script and lib/levanter + lib/haliax from the worktree (PYTHONPATH ahead of the main repo's editable installs), the
# main repo's .venv. Knobs come only from the plan line: ATTN_BACKEND (jax_flash | fa4), SIZE, VOCAB_K, SEQ_LEN, RUN_TAG
# (must contain -ab, so the cleanup below can only ever remove this A/B's own output), AB_WALL (timeout(1) syntax: the
# planned stop; the run is cut there and the exit counts as success), AB_KEEP=1 keeps the output directory.
set -uo pipefail
WT=/scratch/gpfs/GROUP/USER/tmp/speed_wt/fa4-segmented-attention-backend
PY=/scratch/gpfs/GROUP/USER/project/marin/.venv/bin/python
cd "$WT"
source scripts/della/job_tmpdir.sh
module load proxy/default  # wandb only
source scripts/della/wandb_guard.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della XLA_PYTHON_CLIENT_MEM_FRACTION=${JOB_XLA_MEM_FRACTION:-0.9}
export PYTHONPATH=$WT/lib/levanter/src:$WT/lib/haliax/src:$WT${PYTHONPATH:+:$PYTHONPATH}
export SIZE=${SIZE:-300m} ATTN_BACKEND=${ATTN_BACKEND:-jax_flash}
case "${RUN_TAG:-}" in *-ab*) ;; *) echo "fa4_run: RUN_TAG='${RUN_TAG:-}' must contain -ab"; exit 2 ;; esac
export RUN_TAG
got=$($PY -c 'import levanter, haliax; print(levanter.__file__); print(haliax.__file__)')
case "$got" in "$WT"/lib/levanter/src/*"$WT"/lib/haliax/src/*) ;; *) echo "fa4_run: wrong levanter/haliax: $got"; exit 2 ;; esac
run_id=$(DRY_RUN=1 $PY -m experiments.references.della_muonh_qwen3_scaling 2>/dev/null | sed -n 's#^output: speedrun/\([^/]*\)/.*#\1#p' | head -1)
case "$run_id" in *-ab*) ;; *) echo "fa4_run: dry run gave run id '$run_id' (no -ab)"; exit 3 ;; esac
echo "fa4_run: $run_id ATTN_BACKEND=$ATTN_BACKEND SEQ_LEN=${SEQ_LEN:-4096} VOCAB_K=${VOCAB_K:-0} wall=${AB_WALL:-none} GPUs=${CUDA_VISIBLE_DEVICES:-all} $(date +%F' '%T)"
nvidia-smi -L
if [ -n "${AB_WALL:-}" ]; then
  timeout --signal=TERM --kill-after=120 "$AB_WALL" $PY -m experiments.references.della_muonh_qwen3_scaling
  rc=$?
  [ $rc -eq 124 ] && { echo "fa4_run: planned stop at AB_WALL=$AB_WALL"; rc=0; }
else
  $PY -m experiments.references.della_muonh_qwen3_scaling
  rc=$?
fi
echo "fa4_run: $run_id END rc=$rc $(date +%F' '%T)"
if [ "${AB_KEEP:-0}" != 1 ]; then
  case "$run_id" in muonh-qwen3-*-ab*) rm -rf "$MARIN_PREFIX/speedrun/$run_id" && echo "fa4_run: removed speedrun/$run_id" ;; esac
fi
exit $rc
