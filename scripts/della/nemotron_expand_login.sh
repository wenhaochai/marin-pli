#!/bin/bash
# Login/interactive-node variant of scripts/della/nemotron_expand_tokenize.sbatch: tokenize the expanded Nemotron-CC
# corpus (1% sample + 36.5% expansion under marin_store_exp) tier by tier with single-core-pinned workers, delete
# each tier's expansion raw after SUCCESS, then swap the caches into marin_store_big. Resumable. Logs go to
# marin_store_exp/tmp/ so any node can run or inspect it.
#
#     ZEPHYR_MAX_WORKERS=16 nice -n 19 bash scripts/della/nemotron_expand_login.sh
set -uo pipefail
B=/scratch/gpfs/KARTHIKN/wc9403
EXPP=$B/marin_store_exp
SRC1=$B/marin_store/raw/nemotro-cc-eeb783
EXP=$B/marin_store_big/raw/nemotro-cc-eeb783-expansion
RAW=$EXPP/raw/nemotro-cc-eeb783
LOG=$EXPP/tmp/expand_progress.log
cd $B/project/marin-july
mkdir -p $EXPP/tmp
exec >> $LOG 2>&1
echo "=== START $(date) on $(hostname), workers=${ZEPHYR_MAX_WORKERS:-16}"

[ "$(cat $B/marin_store/tokenized/starcoderdata-12f018/.executor_status 2>/dev/null)" = "SUCCESS" ] || { echo "ABORT: 1% corpus not finished"; exit 1; }
n=$(find $EXP -name '*.jsonl.zstd' | wc -l); [ "$n" -eq 11240 ] || [ -d $RAW ] || { echo "ABORT: expansion has $n files, expected 11240"; exit 1; }

mkdir -p $EXPP/raw $EXPP/tokenized
for d in paloma-fc6827 proof-pile-2-f1b1d8 starcoderdata-720c8c uncheatable-eval; do [ -e $EXPP/raw/$d ] || ln -s $B/marin_store/raw/$d $EXPP/raw/$d; done
for d in paloma uncheatable_eval proofpile_2-4a35c7 starcoderdata-12f018; do [ -e $EXPP/tokenized/$d ] || ln -s $B/marin_store/tokenized/$d $EXPP/tokenized/$d; done
if [ ! -d $RAW ]; then
  mkdir -p $RAW
  for src in $SRC1 $EXP; do
    (cd $src && find contrib -name '*.jsonl.zstd') | while read -r f; do mkdir -p "$RAW/$(dirname "$f")"; ln "$src/$f" "$RAW/$f"; done
  done
fi
echo "ASSEMBLED $(date +%H:%M:%S): $(find $RAW -name '*.jsonl.zstd' | wc -l) files"

export HF_HOME=$B/cache/huggingface MARIN_PREFIX=$EXPP DIM=1024 ZEPHYR_SUBPROCESS=1 ZEPHYR_MAX_WORKERS=${ZEPHYR_MAX_WORKERS:-16}
export ZEPHYR_WORKER_CPU_PIN=${ZEPHYR_WORKER_CPU_PIN:-1} TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
declare -A CACHE=([hq_actual]=hq_actual-5af4cc [hq_synth]=hq_synth-3525e2 [medium_high]=medium_high-d21701 [medium]=medium-d86506
                  [medium_low]=medium_low-0fdb07 [low_actual]=low_actual-cb3f2c [low_synth]=low_synth-3c57b3)
declare -A PAT=([hq_actual]="quality=high/kind=actual" [hq_synth]="quality=high/kind=synthetic" [medium_high]="quality=medium-high"
                [medium]="quality=medium" [medium_low]="quality=medium-low" [low_actual]="quality=low/kind=actual" [low_synth]="quality=low/kind=synthetic")
# TIERS selects a subset so several hosts/jobs can split the work; a tier another host is tokenizing right now
# (status RUNNING with a lock touched in the last 15 min) is skipped instead of waited on.
for split in ${TIERS:-hq_actual hq_synth medium_high medium medium_low low_actual low_synth}; do
  d=$EXPP/tokenized/nemotron_cc/${CACHE[$split]}
  if [ "$(cat $d/.executor_status 2>/dev/null)" = "SUCCESS" ]; then echo "SKIP $split"; continue; fi
  if [ "$(cat $d/.executor_status 2>/dev/null)" = "RUNNING" ] && [ -n "$(find $d -maxdepth 1 -name .executor_status.lock -mmin -15)" ]; then echo "SKIP $split (running elsewhere: $(cat $d/.executor_status.lock))"; continue; fi
  echo "TOKENIZE $split START $(date +%F' '%H:%M:%S) on $(hostname) workers=$ZEPHYR_MAX_WORKERS"
  .venv/bin/python -m experiments.grug.moe.della_july_baseline --run_only "[\"^tokenized/nemotron_cc/$split\$\"]" --max_concurrent 1 > $EXPP/tmp/tokenize_$split.log 2>&1
  status=$(cat $d/.executor_status 2>/dev/null)
  tokens=$(python3 -c "import json; print(json.load(open('$d/train/.stats.json'))['total_tokens'])" 2>/dev/null)
  echo "TOKENIZE $split END $(date +%F' '%H:%M:%S) status=$status tokens=$tokens"
  [ "$status" = "SUCCESS" ] || { echo "STOPPED: $split (log marin_store_exp/tmp/tokenize_$split.log)"; exit 1; }
  (cd $EXP && find contrib -path "*/${PAT[$split]}/*" -name '*.jsonl.zstd') | while read -r f; do rm -f "$EXP/$f" "$RAW/$f"; done
  echo "FREED $split raw $(date +%H:%M:%S); expansion left $(du -sh $EXP | cut -f1)"
done

BT=$B/marin_store_big/tokenized
for split in hq_actual hq_synth medium_high medium medium_low low_actual low_synth; do
  [ "$(cat $EXPP/tokenized/nemotron_cc/${CACHE[$split]}/.executor_status 2>/dev/null)" = "SUCCESS" ] || { echo "NOT ALL TIERS DONE ($split); no swap. $(date)"; exit 0; }
done
if [ ! -L $EXPP/tokenized/nemotron_cc ]; then
  mv $BT/nemotron_cc $BT/nemotron_cc-1pct
  mv $EXPP/tokenized/nemotron_cc $BT/nemotron_cc   # same GPFS fileset: a rename
  ln -s $BT/nemotron_cc $EXPP/tokenized/nemotron_cc
  BR=$B/marin_store_big/raw/nemotro-cc-eeb783
  [ -L $BR ] && rm $BR && ln -s $RAW $BR
  echo "SWAPPED into marin_store_big $(date +%H:%M:%S)"
fi
for split in hq_actual hq_synth medium_high medium medium_low low_actual low_synth; do
  d=$BT/nemotron_cc/${CACHE[$split]}
  echo "$split $(cat $d/.executor_status) $(python3 -c "import json; print(json.load(open('$d/train/.stats.json'))['total_tokens'])")"
done
echo "ALL DONE $(date)"
