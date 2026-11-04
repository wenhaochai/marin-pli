#!/bin/bash
#SBATCH --job-name=hnet-score-bytes
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
# Analysis C of the owner's popup 2026-11-03 asks for H-Net's regret on the same axes as plain bytes. w1041 has the H-Net's
# log-probability of every Paloma byte at the 19456-byte window; the plain-bytes row of mass_shift/groups.json is at window
# 4096, so it does not line up. This job scores the plain-bytes pair (w35) and the pair with a chunk every 5 bytes (w1010) at
# the same windows, byte by byte. Then a regret at a chunk start can be compared with the plain model's regret on the same
# bytes. Reads checkpoints only; writes logs/vocab_token_losses/<run>.npz and logs/hnet_eval/hnet2x2n5-rfixed/.
set -uo pipefail
CODE=/scratch/gpfs/GROUP/USER/project/marin
cd /scratch/gpfs/GROUP/USER/project/marin
source scripts/della/job_tmpdir.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=$CODE:$CODE/lib/levanter/src:$CODE/lib/haliax/src
S=$MARIN_PREFIX/speedrun
P=muonh-qwen3-300m-della4xh100
nvidia-smi -L
echo "START $(date +%H:%M:%S) code $(git -C $CODE log --oneline -1 | cut -c1-60)"
fail=0
for run in $P-bytes-sl19456 $P-bytes-rep8-sl19456; do
  f=$CODE/logs/vocab_token_losses/$run.npz
  if [ -f "$f" ]; then echo "have $f"; continue; fi
  CKPT=$S/$run/2026.09.13/checkpoints/step-11420 SIZE=300m VOCAB_K=256 SEQ_LEN=19456 EVAL_BATCH=8 OUT=$f \
    timeout 60m .venv/bin/python -m experiments.references.vocab_token_losses > "$CODE/logs/vocab_token_losses/$run.log" 2>&1 || fail=$((fail + 1))
  echo "== $run $(date +%H:%M:%S) fail so far $fail"
  grep -E "windows, mean loss|Error|error:" "$CODE/logs/vocab_token_losses/$run.log" | cut -c1-160
done
IFS=, read -r -a GPUS <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
SHARDS=("m2d2_s2orc_unsplit,ptb,wikitext_103,twitterAAE_HELM_fixed"
        "c4_100_domains,dolma_100_programing_languages,gab,4chan"
        "dolma_100_subreddits,m2d2_wikipedia_unsplit,falcon-refinedweb,c4_en"
        "dolma-v1_5,manosphere_meta_sep,mc4,redpajama")
export RUN_U=$P-hnet2x2n5-rfixed-bytes-sl19456 CKPT_U=$S/$P-hnet2x2n5-rfixed-bytes-sl19456/2026.09.13/checkpoints/step-11420
export CKPT_R=$S/$P-hnet2x2n5-rfixed-bytes-rep8-sl19456/2026.09.13/checkpoints/step-11420
export OUT=$CODE/logs/hnet_eval/hnet2x2n5-rfixed REP_STRIDE=0
mkdir -p "$OUT"
pids=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} SUBSETS=${SHARDS[$i]} timeout 40m .venv/bin/python -m experiments.references.hnet_eval > "$OUT/shard$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
.venv/bin/python -m experiments.references.hnet_eval merge > "$OUT/merge.log" 2>&1 || fail=$((fail + 1))
grep -E "own_u against|own_r against|exact macro|Error|error:" "$OUT/merge.log" | cut -c1-200
echo "DONE fail=$fail $(date +%H:%M:%S)"
[ "$fail" -eq 0 ]
