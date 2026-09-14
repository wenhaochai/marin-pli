#!/bin/bash
#SBATCH --job-name=nemotron-expand-tokenize
#SBATCH --account=karthikn
#SBATCH --partition=cpu
#SBATCH --qos=vlong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=120:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# Tokenize the expanded Nemotron-CC corpus (1% sample + 36.5% expansion, assembled as hardlinks under
# marin_store_exp/raw/nemotro-cc-eeb783) tier by tier on a CPU node, delete each tier's expansion raw after
# SUCCESS, then rename the seven caches into marin_store_big/tokenized/nemotron_cc for the held d768/d1024 runs.
# Resumable: finished caches are skipped, finished part dirs inside a tier are skipped by the cache writer.
set -uo pipefail
B=/scratch/gpfs/KARTHIKN/wc9403
EXPP=$B/marin_store_exp
EXP=$B/marin_store_big/raw/nemotro-cc-eeb783-expansion
RAW=$EXPP/raw/nemotro-cc-eeb783
cd $B/project/marin-july
[ -d $RAW ] || { echo "ABORT: $RAW not assembled"; exit 1; }
echo "START $(date) on $(hostname): $(find $RAW -name '*.jsonl.zstd' | wc -l) raw files"

export HF_HOME=$B/cache/huggingface MARIN_PREFIX=$EXPP DIM=1024 ZEPHYR_SUBPROCESS=1 ZEPHYR_MAX_WORKERS=${ZEPHYR_MAX_WORKERS:-$((SLURM_CPUS_PER_TASK - 2))}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
declare -A CACHE=([hq_actual]=hq_actual-5af4cc [hq_synth]=hq_synth-3525e2 [medium_high]=medium_high-d21701 [medium]=medium-d86506
                  [medium_low]=medium_low-0fdb07 [low_actual]=low_actual-cb3f2c [low_synth]=low_synth-3c57b3)
declare -A PAT=([hq_actual]="quality=high/kind=actual" [hq_synth]="quality=high/kind=synthetic" [medium_high]="quality=medium-high"
                [medium]="quality=medium" [medium_low]="quality=medium-low" [low_actual]="quality=low/kind=actual" [low_synth]="quality=low/kind=synthetic")
for split in hq_actual hq_synth medium_high medium medium_low low_actual low_synth; do
  d=$EXPP/tokenized/nemotron_cc/${CACHE[$split]}
  if [ "$(cat $d/.executor_status 2>/dev/null)" = "SUCCESS" ]; then echo "SKIP $split"; continue; fi
  echo "TOKENIZE $split START $(date +%F' '%H:%M:%S)"
  .venv/bin/python -m experiments.grug.moe.della_july_baseline --run_only "[\"^tokenized/nemotron_cc/$split\$\"]" --max_concurrent 1 > $EXPP/tmp/tokenize_${split}_slurm.log 2>&1
  status=$(cat $d/.executor_status 2>/dev/null)
  tokens=$(python3 -c "import json; print(json.load(open('$d/train/.stats.json'))['total_tokens'])" 2>/dev/null)
  echo "TOKENIZE $split END $(date +%F' '%H:%M:%S) status=$status tokens=$tokens"
  [ "$status" = "SUCCESS" ] || { echo "STOPPED: $split (log marin_store_exp/tmp/tokenize_${split}_slurm.log)"; exit 1; }
  (cd $EXP && find contrib -path "*/${PAT[$split]}/*" -name '*.jsonl.zstd') | while read -r f; do rm -f "$EXP/$f" "$RAW/$f"; done
  echo "FREED $split raw $(date +%H:%M:%S); expansion left $(du -sh $EXP | cut -f1)"
done

BT=$B/marin_store_big/tokenized
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
