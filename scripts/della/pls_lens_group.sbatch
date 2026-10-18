#!/bin/bash
#SBATCH --job-name=lens
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96   # 8-GPU jobs ask for all 96 CPUs: priority JobSize counts CPUs
#SBATCH --mem-per-gpu=24G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
#FETCHABLE: self-contained (the checkpoint list below comes from the job name, nothing from the submit shell)

# The logit lens of every finished model behind the page (owner popups 2026-10-17; page Figure 16): for each checkpoint
# of the group named by the job (--job-name=lens-<group>), the launcher's LENS_FROM loads the source's backbone and final
# head, freezes them, and runs the final per-layer eval on all 8 GPUs (eval/L{k}/loss of a -lensat{N} run on W&B), one
# checkpoint after another, so the GPUs stay busy. Each eval's own output (its run dir and temporary checkpoint, which hold
# nothing new) is removed after it, only for an id ending in -lensat<N>[-tag]. Ledger: tools/packed_task.sh, one main task per
# checkpoint (run lens-<group>), in the host's ledger when fetched.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls-auto10
source scripts/della/pls_env.sh
JID=${FETCHED_JOB_ID:-$SLURM_JOB_ID}
if [ -n "${FETCHED_JOB_ID:-}" ]; then NAME=$(sacct -X -n -j "$FETCHED_JOB_ID" -o JobName%80 | head -1 | tr -d ' '); else NAME=$SLURM_JOB_NAME; fi
LAUNCH=experiments.references.della_muonh_qwen3_scaling
PT=/scratch/gpfs/GROUP/USER/tools/packed_task.sh
unset SMOKE_STEPS TOTAL_STEPS INIT_FROM PDP PLS_MON RUN_TAG PROBE_FROM HEADS_FROM LAYERS ARCH PARAM_MAP SEED
module load proxy/default 2>/dev/null || true
export MARIN_PREFIX=${MARIN_PREFIX:-/scratch/gpfs/GROUP/USER/marin_store_big}
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=marin-della VARIANT=pls XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PACKED_LEDGER_DIR=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/packed
mkdir -p "$PACKED_LEDGER_DIR"; ledger=$PACKED_LEDGER_DIR/$SLURM_JOB_ID.tsv
[[ -s $ledger ]] || printf 'task\trun\tkind\tgpus\tstart\tend\texit\n' > "$ledger"
R=$MARIN_PREFIX/speedrun
C=2026.09.13/checkpoints

# spec: "<source id without muonh-qwen3->|<step>|<knobs naming the source>"
G130=(
  "130m-della4xh100-pls1-sep-bbfrozen|4958|SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=1"
  "130m-della4xh100-pls0.2-sep|4958|SIZE=130m PLS_W=0.2 PLS_SEP=1"
  "130m-della4xh100-pls0.1-sep|4958|SIZE=130m PLS_W=0.1 PLS_SEP=1"
  "130m-della4xh100-pls0.05-sep|4958|SIZE=130m PLS_W=0.05 PLS_SEP=1"
  "130m-della4xh100-pls1-sep-local|4958|SIZE=130m PLS_W=1 PLS_SEP=1 PLS_LOCAL=1"
  "130m-della4xh100-pls0.2-sep-local|4958|SIZE=130m PLS_W=0.2 PLS_SEP=1 PLS_LOCAL=1"
  "130m-della4xh100-pls0.2-sep-shdepth|4958|SIZE=130m PLS_W=0.2 PLS_SEP=1 PLS_SHAPE=depth"
  "130m-della4xh100-pls0.2-sep-pcg|4958|SIZE=130m PLS_W=0.2 PLS_SEP=1 PLS_PCGRAD=1"
  "130m-della4xh100-pls0.2-sep-shdepth-pcg|4958|SIZE=130m PLS_W=0.2 PLS_SEP=1 PLS_SHAPE=depth PLS_PCGRAD=1"
  "130m-della4xh100-pls1-sep-pcg|4958|SIZE=130m PLS_W=1 PLS_SEP=1 PLS_PCGRAD=1"
  "130m-della4xh100-pls1-sep-on300-400|4958|SIZE=130m PLS_W=1 PLS_SEP=1 PLS_ON=300:400"
  "130m-della4xh100-pls1-sep-on1500-1600|4958|SIZE=130m PLS_W=1 PLS_SEP=1 PLS_ON=1500:1600"
  "130m-della4xh100-pls1-sep-bbfrozen-os-fut|4958|SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=1 PLS_OPT_SPLIT=1 PLS_FUT=1"
  "130m-della4xh100-pls1-sep-bbfrozen-os-mtp0.3-fut|4958|SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=1 PLS_OPT_SPLIT=1 PLS_FUT=1 PLS_MTP_W=0.3"
  "130m-della4xh100-pls1-sep-aattnres|4958|SIZE=130m PLS_W=1 PLS_SEP=1 ARCH=attnres"
  "130m-della4xh100-pls1-sep-bbfrozen-aattnres|4958|SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=1 ARCH=attnres"
  # no d1: a 1-layer model has no intermediate layer (its lens is its own head; the monitor fails at L=1, 15720392)
  "130m-della4xh100-d2|4958|SIZE=130m PLS_W=0 LAYERS=2"
  "130m-della4xh100-d3|4958|SIZE=130m PLS_W=0 LAYERS=3"
  "130m-della4xh100-d4|4958|SIZE=130m PLS_W=0 LAYERS=4"
  "130m-della4xh100-d5|4958|SIZE=130m PLS_W=0 LAYERS=5"
)
D48="SIZE=130m LAYERS=48 PLS_W=1 PLS_SEP=1 PARAM_MAP=vocab RUN_TAG=-g4"
G48A=() G48B=()
for a in "" preln lns deepnorm keel; do t=${a:+-a$a}; k=${a:+ ARCH=$a}
  G48A+=("130m-della4xh100-pls1-sep$t-vs-d48-g4|4958|$D48$k" "130m-della4xh100-pls1-sep-bbfrozen$t-vs-d48-g4|4958|$D48 PLS_DETACH_BB=1$k"); done
for a in hc mhc attnres attnres_block moda; do
  G48B+=("130m-della4xh100-pls1-sep-a$a-vs-d48-g4|4958|$D48 ARCH=$a" "130m-della4xh100-pls1-sep-bbfrozen-a$a-vs-d48-g4|4958|$D48 PLS_DETACH_BB=1 ARCH=$a"); done
# lens-48c: the designs trained at microbatch 1 (launcher PARAM_MAP_PDP1 applies only when PDP > 1; unset PDP means the
# whole per-device batch, which ran out of memory for attnres, attnres_block in 15720394). PDP only splits the batch.
G48C=()
for a in attnres attnres_block moda; do
  G48C+=("130m-della4xh100-pls1-sep-a$a-vs-d48-g4|4958|$D48 ARCH=$a PDP=1" "130m-della4xh100-pls1-sep-bbfrozen-a$a-vs-d48-g4|4958|$D48 PLS_DETACH_BB=1 ARCH=$a PDP=1"); done
G48A+=("130m-della4xh100-d12|4958|SIZE=130m PLS_W=0 LAYERS=12" "130m-della4xh100-d24|4958|SIZE=130m PLS_W=0 LAYERS=24" "130m-della4xh100-d36|4958|SIZE=130m PLS_W=0 LAYERS=36")
G300=(
  "300m-della4xh100-pls1-sep|11443|SIZE=300m PLS_W=1 PLS_SEP=1"
  "300m-della4xh100-pls1-sep-bbfrozen|11443|SIZE=300m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=1"
  "300m-della8xh100-pls0-noev|11443|SIZE=300m PLS_W=0 PLS_EVAL=0"
  "300m-della8xh100-pls0-noev-mtp0.3|11443|SIZE=300m PLS_W=0 PLS_EVAL=0 PLS_MTP_W=0.3"
  "520m-della8xh100-pls0-noev|9917|SIZE=520m PLS_W=0 PLS_EVAL=0"
  "520m-della8xh100-pls0-noev-mtp0.3|9917|SIZE=520m PLS_W=0 PLS_EVAL=0 PLS_MTP_W=0.3"
)
case $NAME in
  lens-130m) SPECS=("${G130[@]}");;
  lens-48a) SPECS=("${G48A[@]}");;
  lens-48b) SPECS=("${G48B[@]}");;
  lens-48c) SPECS=("${G48C[@]}");;
  lens-300m) SPECS=("${G300[@]}");;
  lens-130sep) SPECS=("130m-della8xh100-pls1-sep|4958|SIZE=130m PLS_W=1 PLS_SEP=1");;   # the separate-heads rerun (15720165) that kept its checkpoint
  *) echo "unknown lens job name '$NAME'"; exit 2;;
esac
LIMIT=${LENS_EVAL_MIN:-40}m   # one checkpoint's eval, load and compile included
nvidia-smi -L || true
ok=0; bad=0
for spec in "${SPECS[@]}"; do
  IFS='|' read -r src step knobs <<< "$spec"
  ck=$R/muonh-qwen3-$src/$C/step-$step
  if [ ! -d "$ck" ]; then echo "missing checkpoint $ck: skipped"; bad=$((bad + 1)); continue; fi
  log=logs/slurm/${NAME}_${JID}_${src}.log
  t0=$(date +%H:%M:%S); r=0
  # shellcheck disable=SC2086
  env $knobs NUM_GPUS=8 LENS_FROM=$ck bash $PT "lens-$src" "${NAME}" main 8 "$LIMIT" -- "$PY" -m $LAUNCH > "$log" 2>&1 || r=$?
  # a real run prints "Using output path: <prefix>/speedrun/<id>/<date>" (only a dry run prints "output: ...", 2026-10-18)
  rid=$(sed -n 's/.*Using output path: //p' "$log" | head -1); rid=$(basename "$(dirname "$rid")")
  echo "$src: rc=$r $t0-$(date +%H:%M:%S) run $rid; $(grep -oE 'eval loss: [0-9.]+' "$log" | tail -1)"
  if [ $r -eq 0 ]; then ok=$((ok + 1)); else bad=$((bad + 1)); fi
  if [[ -n $rid && $rid == *-lensat${step}* ]]; then   # only this eval's own output
    rm -rf -- "$R/$rid" "$MARIN_PREFIX/tmp/checkpoints-temp$R/$rid"
  fi
done
echo "lens group $NAME: $ok evaluated, $bad failed or missing"
[ $bad -eq 0 ]
