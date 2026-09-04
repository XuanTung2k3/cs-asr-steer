#!/bin/bash
# Round 3 is implemented but intentionally not submitted in Round 1.
set -euo pipefail
REPO=/home/tungnx/cs-asr-steer
export PYTHONPATH="$REPO/src:$REPO"
exec /home/tungnx/miniconda3/envs/acl1/bin/python "$REPO/experiments/round3_qwen_frozen.py" --config "$REPO/configs/rounds/round3_qwen_frozen.yaml" "$@"
