#!/bin/bash
# DG-03 causal screen at one layer (free decoding, D-dev-select). Pass --layer 16 or --layer 24.
#SBATCH --job-name=cs-dg03-screen
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/cs_asr_dg03_screen_%j.log
#SBATCH --error=logs/cs_asr_dg03_screen_%j.err
set -euo pipefail
REPO=/home/tungnx/cs-asr-steer
mkdir -p "$REPO/logs"
export PYTHONPATH="$REPO/src:$REPO"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec /home/tungnx/miniconda3/envs/acl1/bin/python "$REPO/experiments/dg03_causal_screen.py" \
  --basis-dir "$REPO/results/dg03/basis" "$@"
