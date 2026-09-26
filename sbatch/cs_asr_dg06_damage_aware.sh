#!/bin/bash
# DG-06 damage-aware training. mig-only. Usage: sbatch cs_asr_dg06_damage_aware.sh <d1|d2>
#SBATCH --job-name=cs-dg06-train
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cs_asr_dg06_%j.log
#SBATCH --error=logs/cs_asr_dg06_%j.err
set -euo pipefail
VARIANT="${1:?usage: sbatch cs_asr_dg06_damage_aware.sh <d1|d2>}"
if [[ "$VARIANT" != "d1" && "$VARIANT" != "d2" ]]; then
  echo "variant must be d1 or d2" >&2; exit 2
fi
REPO=/home/tungnx/cs-asr-steer
mkdir -p "$REPO/logs"
export PYTHONPATH="$REPO/src:$REPO"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec /home/tungnx/miniconda3/envs/acl1/bin/python \
  "$REPO/experiments/dg06_damage_aware.py" \
  --config "$REPO/configs/dg06_damage_aware.yaml" \
  --variant "$VARIANT" \
  --output-dir "$REPO/results/dg06/$VARIANT" --run
