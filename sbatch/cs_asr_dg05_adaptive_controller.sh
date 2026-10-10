#!/bin/bash
# DG-05A preparation only. Submit later, after review; this launcher is mig-only.
#SBATCH --job-name=cs-dg05a-train
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cs_asr_dg05a_%j.log
#SBATCH --error=logs/cs_asr_dg05a_%j.err
set -euo pipefail
REPO=/home/tungnx/cs-asr-steer
mkdir -p "$REPO/logs"
export PYTHONPATH="$REPO/src:$REPO"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec /home/tungnx/miniconda3/envs/acl1/bin/python \
  "$REPO/experiments/dg05_adaptive_controller.py" \
  --config "$REPO/configs/dg05_adaptive_controller.yaml" \
  --output-dir "$REPO/results/dg05/controller" --run
