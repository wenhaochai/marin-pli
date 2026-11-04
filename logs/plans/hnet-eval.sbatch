#!/bin/bash
#SBATCH --job-name=hnet-eval
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=24G
#SBATCH --time=02:45:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell
# H-Net boundary diagnosis (owner popup 2026-11-03). experiments/references/hnet_eval.py on the 300m
# H-Net pair (unique text w605 / 8 epochs), four processes, one GPU each, four Paloma subsets each (balanced by windows),
# then the merge; one more process routes the repeated eighth of the training text with both models. Reads the two
# checkpoints only; writes logs/hnet_eval/<pair>/.
set -uo pipefail
CODE=/scratch/gpfs/GROUP/USER/project/marin
cd /scratch/gpfs/GROUP/USER/project/marin
source scripts/della/job_tmpdir.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=$CODE:$CODE/lib/levanter/src:$CODE/lib/haliax/src
S=$MARIN_PREFIX/speedrun
RUN_U=muonh-qwen3-300m-della4xh100-hnet2x2n4.74-bytes-sl19456
RUN_R=muonh-qwen3-300m-della4xh100-hnet2x2n4.74-bytes-rep8-sl19456
export RUN_U CKPT_U=$S/$RUN_U/2026.09.13/checkpoints/step-11420 CKPT_R=$S/$RUN_R/2026.09.13/checkpoints/step-11420
export OUT=/scratch/gpfs/GROUP/USER/project/marin/logs/hnet_eval/hnet2x2n4.74
mkdir -p $OUT
nvidia-smi -L
echo "START $(date +%H:%M:%S) code $(git -C $CODE log --oneline -1 | cut -c1-60) + $(git -C $CODE diff --stat | tail -1)"
IFS=, read -r -a GPUS <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# 12,931 windows; shards by window count (m2d2_s2orc 3762, c4_100 2255, subreddits 2134, code 1478, wikipedia 1109, ...)
SHARDS=("m2d2_s2orc_unsplit,ptb,wikitext_103,twitterAAE_HELM_fixed"
        "c4_100_domains,dolma_100_programing_languages,gab,4chan"
        "dolma_100_subreddits,m2d2_wikipedia_unsplit,falcon-refinedweb,c4_en"
        "dolma-v1_5,manosphere_meta_sep,mc4,redpajama")
pids=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} SUBSETS=${SHARDS[$i]} timeout 150m .venv/bin/python -m experiments.references.hnet_eval > $OUT/shard$i.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
# the repeated eighth (182,784 windows x 2 models, encoder only) on all four GPUs, every 4th window, after the Paloma shards
timeout 60m env REP_STRIDE=4 REP_ONLY=1 .venv/bin/python -m experiments.references.hnet_eval > $OUT/rep.log 2>&1 || fail=$((fail + 1))
.venv/bin/python -m experiments.references.hnet_eval merge > $OUT/merge.log 2>&1 || fail=$((fail + 1))
for f in $OUT/shard?.log $OUT/rep.log $OUT/merge.log; do echo "== $f"; grep -v -E "Unknown leaf|[Ww]arn|Metadata mismatch" $f | tail -6; done
echo "DONE fail=$fail $(date +%H:%M:%S)"
exit 0
