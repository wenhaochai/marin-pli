#!/bin/bash
#SBATCH --job-name=vo-mass-shift-bytes
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --comment=fetchable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=96G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin/logs/slurm/%x_%j.out
#FETCHABLE self-contained: needs nothing from the submitting shell
# The bytes pair of vo-mass-shift.sbatch (bytes seed 0, unique against 8 passes), split over 4 GPUs by Paloma subset so
# the shards take about the same time (~12 min each; one GPU took 40+ min in w506). Output
# logs/vocab_mass_shift/bytes-s0-part<i>.npz; tmp_plots/vocab/mass_shift/groups.py merges the parts.
set -uo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin
source scripts/della/job_tmpdir.sh
export MARIN_PREFIX=/scratch/gpfs/GROUP/USER/marin_store_big
export HF_HOME=/scratch/gpfs/GROUP/USER/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=$PWD/lib/levanter/src:$PWD/lib/haliax/src:$PWD
S=$MARIN_PREFIX/speedrun
OUT=logs/vocab_mass_shift
mkdir -p $OUT
nvidia-smi -L
echo "START $(date +%H:%M:%S) $(git log --oneline -1 | cut -c1-80)"
IFS=, read -r -a GPUS <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
ck() { echo "$S/muonh-qwen3-300m-della4xh100$1/2026.09.13/checkpoints/step-54247"; }
SHARDS=("m2d2_s2orc_unsplit"
        "c4_100_domains,m2d2_wikipedia_unsplit"
        "dolma_100_subreddits,dolma-v1_5,4chan,c4_en,falcon-refinedweb,gab"
        "dolma_100_programing_languages,manosphere_meta_sep,mc4,ptb,redpajama,twitterAAE_HELM_fixed,wikitext_103")
pids=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} VOCAB_K=256 SIZE=300m CKPT_U=$(ck -bytes) CKPT_R=$(ck -bytes-rep8) SUBSETS=${SHARDS[$i]} \
    OUT=$OUT/bytes-s0-part$i.npz timeout 25m .venv/bin/python -m experiments.references.vocab_mass_shift > $OUT/bytes-s0-part$i.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
for i in 0 1 2 3; do echo "== part$i"; grep -v -E "Unknown leaf|[Ww]arn" $OUT/bytes-s0-part$i.log | tail -3; done
echo "DONE fail=$fail $(date +%H:%M:%S)"
exit 0
