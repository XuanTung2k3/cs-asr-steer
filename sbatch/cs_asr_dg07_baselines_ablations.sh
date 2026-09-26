#!/bin/bash
# DG-07B launcher prepared by DG-07A.  Submit one variant at a time from the
# frozen batch manifest; never submit more than two DG-07 jobs concurrently.
# Usage: sbatch cs_asr_dg07_baselines_ablations.sh <VARIANT>
# This file is intentionally not submitted during DG-07A.
#SBATCH --job-name=cs-dg07
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cs_asr_dg07_%j.log
#SBATCH --error=logs/cs_asr_dg07_%j.err
set -euo pipefail

VARIANT="${1:?usage: sbatch cs_asr_dg07_baselines_ablations.sh <VARIANT>}"
case "$VARIANT" in
  LB1_SALSA_EXACT_GLOBAL|LB2_LORA_MATCHED_BUDGET|A1_LOCAL_ONLY|A2_CONDITIONING_ONLY|A3_FIXED_MIXTURE_GATE) ;;
  *) echo "unknown or deferred DG-07 variant: $VARIANT" >&2; exit 2 ;;
esac

if [[ "${SLURM_JOB_PARTITION:-mig}" != "mig" ]]; then
  echo "DG-07 requires partition=mig; refusing this job" >&2
  exit 3
fi

REPO=/home/tungnx/cs-asr-steer
mkdir -p "$REPO/logs"
export PYTHONPATH="$REPO/src:$REPO"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

exec /home/tungnx/miniconda3/envs/acl1/bin/python \
  "$REPO/experiments/dg07_baselines_ablations.py" \
  --config "$REPO/configs/dg07_baselines_ablations.yaml" \
  --variant "$VARIANT" \
  --output-dir "$REPO/results/dg07/$VARIANT" --run
