#!/bin/bash
# Learned-expansion one-seed training (Workstream C/D). mig-only.
# Usage: sbatch cs_asr_learned_expansion_train.sh <layer> <mode>
#SBATCH --job-name=cs-lexp-train
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cs_asr_lexp_train_%j.log
#SBATCH --error=logs/cs_asr_lexp_train_%j.err
set -euo pipefail
LAYER="${1:?usage: sbatch cs_asr_learned_expansion_train.sh <layer> <mode>}"
MODE="${2:?usage: sbatch cs_asr_learned_expansion_train.sh <layer> <mode>}"
REPO=/home/tungnx/cs-asr-steer
mkdir -p "$REPO/logs"
export PYTHONPATH="$REPO/src:$REPO"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec /home/tungnx/miniconda3/envs/acl1/bin/python \
  "$REPO/experiments/learned_expansion.py" \
  --config "$REPO/configs/expansion/learned_expansion_run.yaml" \
  --layer "$LAYER" --mode "$MODE" \
  --output-dir "$REPO/results/learned_expansion/proposed/L${LAYER}_${MODE}" --run
