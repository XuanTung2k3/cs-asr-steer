#!/bin/bash
# DG-02 real-Whisper acceptance (Stage-3 Phase D): one utterance, both active
# layers, plumbing-only. Short GPU job; inherits repo conventions.
#SBATCH --job-name=cs-dg02-accept
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cs_asr_dg02_real_acceptance_%j.log
#SBATCH --error=logs/cs_asr_dg02_real_acceptance_%j.err
set -euo pipefail
REPO=/home/tungnx/cs-asr-steer
mkdir -p "$REPO/logs"
export PYTHONPATH="$REPO/src:$REPO"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec /home/tungnx/miniconda3/envs/acl1/bin/python "$REPO/experiments/dg02_real_acceptance.py" \
  --data-config "$REPO/configs/lss/l1b_candidates_dialogue_v2r3.yaml" \
  --output "$REPO/results/dg02_real_acceptance.json" "$@"
