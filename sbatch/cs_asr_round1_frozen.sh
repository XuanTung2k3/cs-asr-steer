#!/bin/bash
# Round 1 Job A: one H100, three-hour hard limit; inherits repo conventions.
#SBATCH --job-name=cs-r1-frozen
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/cs_asr_round1_frozen_%j.log
#SBATCH --error=logs/cs_asr_round1_frozen_%j.err
set -euo pipefail
REPO=/home/tungnx/cs-asr-steer
mkdir -p "$REPO/logs"
export PYTHONPATH="$REPO/src:$REPO"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec /home/tungnx/miniconda3/envs/acl1/bin/python "$REPO/experiments/round1_frozen.py" \
  --config "$REPO/configs/rounds/round1_frozen.yaml" \
  --output-root "$REPO/results/round1/job_a" "$@"
