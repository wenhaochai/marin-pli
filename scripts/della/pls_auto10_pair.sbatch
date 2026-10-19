#!/bin/bash
#SBATCH --job-name=auto10
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96   # 8-GPU jobs ask for all 96 CPUs: priority JobSize counts CPUs
#SBATCH --mem-per-gpu=24G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm/%x_%j.out
#FETCHABLE: self-contained (both runs' knobs below come from the job name, nothing from the submit shell)

# Ten more attempts at the last-layer cost of separate heads, 130m (owner popup 2026-10-16; page Figure 15): two 130m runs
# of one direction on one 8-GPU node through pls_h100x8_pair.sbatch (smoke copies, readout gap check, training, ledger).
# Submit with --job-name=auto10-<direction> --comment=fetchable and a right-sized --time; the pair script reads the
# limit from the job (fetched: FETCHED_JOB_ID's). PAIR_GRACE_MIN 25: after one half ends, the other may run 25 min
# more on its own, under Della's 30-min idle-GPU mail (a fetched job must keep its host's GPUs busy).
set -euo pipefail
cd /scratch/gpfs/GROUP/USER/project/marin-pls-auto10
if [ -n "${FETCHED_JOB_ID:-}" ]; then NAME=$(sacct -X -n -j "$FETCHED_JOB_ID" -o JobName%80 | head -1 | tr -d ' '); else NAME=$SLURM_JOB_NAME; fi
SEP="SIZE=130m PLS_W=1 PLS_SEP=1"
case $NAME in
  auto10-hc)      A="$SEP ARCH=hc";      B="$SEP PLS_DETACH_BB=1 ARCH=hc";;        # 9: Hyper-Connections, 6 layers
  auto10-attnres) A="$SEP ARCH=attnres"; B="$SEP PLS_DETACH_BB=1 ARCH=attnres";;   # 10: AttnRes (Full), 6 layers
  auto10-os-randl)  A="$SEP PLS_OPT_SPLIT=1"; B="$SEP PLS_RANDL=1";;      # 1: own clipping; 8: one random layer per step
  auto10-mix-mlpo)  A="$SEP PLS_MIX=0.2";     B="$SEP PLS_MLPONLY=1";;    # 2: normalise then mix; 6: MLPs only (both: a 2nd pass)
  auto10-kd-ad)     A="$SEP PLS_KD=1";        B="$SEP PLS_ADAPT=1024";;   # 3: distillation; 5: exit adapters
  auto10-conf-first) A="$SEP PLS_CONF=1";     B="$SEP PLS_FIRST=256";;    # 4: confidence gate; 7: first 256 positions
  auto10-upto)      A="$SEP PLS_SEP_UPTO=3"; B="$SEP PLS_SEP_UPTO=2";;  # owner popup 2026-10-18: only layers 1-3 (1-2) reach the backbone
  auto10-upto1s)    A="$SEP PLS_SEP_UPTO=1"; B="$SEP PLS_SEP_UPTO=2 SEED=1";;  # owner popup 2026-10-18: 13 only layer 1; 14 layers 1-2 with init seed 1 (data order unchanged)
  *) echo "unknown auto10 job name '$NAME'"; exit 2;;
esac
export PAIR_A=$A PAIR_B=$B PAIR_GRACE_MIN=25
exec bash scripts/della/pls_h100x8_pair.sbatch
