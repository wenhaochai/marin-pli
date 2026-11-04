#!/bin/bash
#SBATCH --job-name=hnet-rescore
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=24G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell
# H-Net no-overflow rescore (owner popup 2026-11-03, analysis D): w1041 found that a slot for every boundary moves the seed 0
# pair from 1.3733 / 1.4045 to 1.3411 / 1.3666. This job repeats hnet_eval.py on seed 1 (with the repeated-eighth routing and
# the per-byte files), on the no-lookahead pair and on the pair with target 9.5, to check that result and to compare regrets
# with no overflow. Reads checkpoints only; writes logs/hnet_eval/<tag>/.
set -uo pipefail
CODE=/scratch/gpfs/GROUP/USER/project/marin
cd /scratch/gpfs/GROUP/USER/project/marin
source scripts/della/job_tmpdir.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=$CODE:$CODE/lib/levanter/src:$CODE/lib/haliax/src
S=$MARIN_PREFIX/speedrun
nvidia-smi -L
echo "START $(date +%H:%M:%S) code $(git -C $CODE log --oneline -1 | cut -c1-60)"
IFS=, read -r -a GPUS <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# 12,931 windows; shards by window count
SHARDS=("m2d2_s2orc_unsplit,ptb,wikitext_103,twitterAAE_HELM_fixed"
        "c4_100_domains,dolma_100_programing_languages,gab,4chan"
        "dolma_100_subreddits,m2d2_wikipedia_unsplit,falcon-refinedweb,c4_en"
        "dolma-v1_5,manosphere_meta_sep,mc4,redpajama")
fail=0
score() {   # tag, unique run, 8-epoch run, summaries only (0 or 1), repeated-eighth stride (0: none)
  export RUN_U=$2 CKPT_U=$S/$2/2026.09.13/checkpoints/step-11420 CKPT_R=$S/$3/2026.09.13/checkpoints/step-11420
  export OUT=$CODE/logs/hnet_eval/$1 SUMMARY_ONLY=$4
  mkdir -p "$OUT"
  local pids=() i p
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} SUBSETS=${SHARDS[$i]} timeout 40m .venv/bin/python -m experiments.references.hnet_eval > "$OUT/shard$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
  if [ "$5" -gt 0 ]; then
    timeout 15m env REP_STRIDE="$5" REP_ONLY=1 .venv/bin/python -m experiments.references.hnet_eval > "$OUT/rep.log" 2>&1 || fail=$((fail + 1))
  fi
  .venv/bin/python -m experiments.references.hnet_eval merge > "$OUT/merge.log" 2>&1 || fail=$((fail + 1))
  echo "== $1 $(date +%H:%M:%S) fail so far $fail"
  grep -E "own_u against|own_r against|exact macro|Error|error:" "$OUT/merge.log" | cut -c1-200
}
P=muonh-qwen3-300m-della4xh100
score hnet2x2n4.74-s1 $P-hnet2x2n4.74-bytes-sl19456-s1 $P-hnet2x2n4.74-bytes-rep8-sl19456-s1 0 4
score hnet2x2n4.74-rcausal $P-hnet2x2n4.74-rcausal-bytes-sl19456 $P-hnet2x2n4.74-rcausal-bytes-rep8-sl19456 1 0
score hnet2x2n9.5 $P-hnet2x2n9.5-bytes-sl19456 $P-hnet2x2n9.5-bytes-rep8-sl19456 1 0
echo "DONE fail=$fail $(date +%H:%M:%S)"
[ "$fail" -eq 0 ]
