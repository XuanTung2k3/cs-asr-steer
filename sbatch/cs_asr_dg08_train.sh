#!/bin/bash
# DG-08 multi-seed finalist training. mig-only.
# Usage: sbatch cs_asr_dg08_train.sh <SALSA|LORA|MSTAR> <seed>
#SBATCH --job-name=cs-dg08-train
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cs_asr_dg08_train_%j.log
#SBATCH --error=logs/cs_asr_dg08_train_%j.err
set -euo pipefail
METHOD="${1:?usage: sbatch cs_asr_dg08_train.sh <SALSA|LORA|MSTAR> <seed>}"
SEED="${2:?seed required}"
REPO=/home/tungnx/cs-asr-steer
mkdir -p "$REPO/logs"
export PYTHONPATH="$REPO/src:$REPO"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
case "$METHOD" in
  SALSA)
    exec $PY "$REPO/experiments/dg07_baselines_ablations.py" \
      --config "$REPO/configs/dg07_baselines_ablations.yaml" \
      --variant LB1_SALSA_EXACT_GLOBAL --seed "$SEED" \
      --output-dir "$REPO/results/dg08/train/SALSA/seed$SEED" --run ;;
  LORA)
    exec $PY "$REPO/experiments/dg07_baselines_ablations.py" \
      --config "$REPO/configs/dg07_baselines_ablations.yaml" \
      --variant LB2_LORA_MATCHED_BUDGET --seed "$SEED" \
      --output-dir "$REPO/results/dg08/train/LORA/seed$SEED" --run ;;
  MSTAR)
    exec $PY "$REPO/experiments/dg06_damage_aware.py" \
      --config "$REPO/configs/dg06_damage_aware.yaml" \
      --variant d1 --seed "$SEED" \
      --output-dir "$REPO/results/dg08/train/MSTAR/seed$SEED" --run ;;
  *) echo "unknown method $METHOD" >&2; exit 2 ;;
esac
