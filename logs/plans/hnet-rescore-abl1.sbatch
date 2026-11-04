#!/bin/bash
#SBATCH --job-name=hnet-rescore-abl1
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=24G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell
# H-Net ablation (owner popup 2026-11-03): the pairs with the 128K BPE boundaries (w1047) and with space boundaries (w1046),
# scored by hnet_eval.py byte by byte, with a slot for every boundary as well. w1047 answers whether the 128K tokenizer's regret
# comes from its segmentation or from its lookup rows. Overflow differs between arms, so the slot condition is needed
# to compare them. Reads checkpoints only; writes logs/hnet_eval/<tag>/.
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
SHARDS=("m2d2_s2orc_unsplit,ptb,wikitext_103,twitterAAE_HELM_fixed"
        "c4_100_domains,dolma_100_programing_languages,gab,4chan"
        "dolma_100_subreddits,m2d2_wikipedia_unsplit,falcon-refinedweb,c4_en"
        "dolma-v1_5,manosphere_meta_sep,mc4,redpajama")
fail=0
score() {   # tag, unique run, 8-epoch run
  export RUN_U=$2 CKPT_U=$S/$2/2026.09.13/checkpoints/step-11420 CKPT_R=$S/$3/2026.09.13/checkpoints/step-11420
  export OUT=$CODE/logs/hnet_eval/$1 REP_STRIDE=0
  mkdir -p "$OUT"
  local pids=() i p
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} SUBSETS=${SHARDS[$i]} timeout 40m .venv/bin/python -m experiments.references.hnet_eval > "$OUT/shard$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
  .venv/bin/python -m experiments.references.hnet_eval merge > "$OUT/merge.log" 2>&1 || fail=$((fail + 1))
  echo "== $1 $(date +%H:%M:%S) fail so far $fail"
  grep -E "own_u against|own_r against|exact macro|Error|error:" "$OUT/merge.log" | cut -c1-200
}
P=muonh-qwen3-300m-della4xh100
score hnet2x2n4.74-rbpe $P-hnet2x2n4.74-rbpe-sm1.75-bytes-sl19456 $P-hnet2x2n4.74-rbpe-sm1.75-bytes-rep8-sl19456
score hnet2x2n6.17-rspace $P-hnet2x2n6.17-rspace-sm1.75-bytes-sl19456 $P-hnet2x2n6.17-rspace-sm1.75-bytes-rep8-sl19456
echo "DONE fail=$fail $(date +%H:%M:%S)"
[ "$fail" -eq 0 ]
