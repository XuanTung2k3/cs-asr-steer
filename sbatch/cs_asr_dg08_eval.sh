#!/bin/bash
# DG-08 locked D-test / timing evaluation. mig-only.
# Usage: sbatch cs_asr_dg08_eval.sh <dtest|timing> <greedy|beam5> "<systems csv>"
#   systems e.g. "F1_DG04_FROZEN_STEERING,F2_SALSA:13,F2_SALSA:42,F2_SALSA:73"
#SBATCH --job-name=cs-dg08-eval
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cs_asr_dg08_eval_%j.log
#SBATCH --error=logs/cs_asr_dg08_eval_%j.err
set -euo pipefail
TASK="${1:?usage: sbatch cs_asr_dg08_eval.sh <dtest|timing> <greedy|beam5> \"<systems>\" [batch]}"
REGIME="${2:?regime required}"
SYSTEMS="${3:-}"
BATCH="${4:-}"
REPO=/home/tungnx/cs-asr-steer
mkdir -p "$REPO/logs"
export PYTHONPATH="$REPO/src:$REPO"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
BATCH_ARG=()
[ -n "$BATCH" ] && BATCH_ARG=(--batch-size "$BATCH")
exec /home/tungnx/miniconda3/envs/acl1/bin/python \
  "$REPO/experiments/dg08_dtest_eval.py" \
  --config "$REPO/configs/dg08_locked_eval.yaml" \
  --task "$TASK" --regime "$REGIME" --systems "$SYSTEMS" "${BATCH_ARG[@]}" --run
