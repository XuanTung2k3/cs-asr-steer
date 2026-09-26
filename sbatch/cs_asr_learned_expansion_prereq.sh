#!/bin/bash
# Frozen full-dev prerequisite screen (Workstream A). mig-only. No training.
#SBATCH --job-name=cs-lexp-prereq
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/cs_asr_lexp_prereq_%j.log
#SBATCH --error=logs/cs_asr_lexp_prereq_%j.err
set -euo pipefail
REPO=/home/tungnx/cs-asr-steer
mkdir -p "$REPO/logs"
export PYTHONPATH="$REPO/src:$REPO"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec /home/tungnx/miniconda3/envs/acl1/bin/python \
  "$REPO/experiments/learned_expansion_prereq.py" \
  --config "$REPO/configs/expansion/learned_expansion_run.yaml" \
  --layers "24,26,27,31" --batch-size 8 --run
