#!/bin/bash
#SBATCH --job-name=cs-r2-combos
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/cs_asr_round2_combinations_%j.log
set -euo pipefail
REPO=/home/tungnx/cs-asr-steer
export PYTHONPATH="$REPO/src:$REPO"
exec /home/tungnx/miniconda3/envs/acl1/bin/python "$REPO/experiments/round2_combinations.py" --config "$REPO/configs/rounds/round2_combinations.yaml" "$@"
