#!/bin/bash
# DG-04 frozen-steering baselines (Whisper-large-v3 x CS-Dialogue, L24). Pass --systems B1,B2 or --systems B3.
#SBATCH --job-name=cs-dg04-base
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/cs_asr_dg04_frozen_baselines_%j.log
#SBATCH --error=logs/cs_asr_dg04_frozen_baselines_%j.err
set -euo pipefail
REPO=/home/tungnx/cs-asr-steer
mkdir -p "$REPO/logs"
export PYTHONPATH="$REPO/src:$REPO"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec /home/tungnx/miniconda3/envs/acl1/bin/python "$REPO/experiments/dg04_frozen_baselines.py" \
  --output-dir "$REPO/results/dg04" "$@"
