#!/bin/bash
# BASIS-A5 launcher.  All GPU construction, preflight, and atlas work runs
# through sbatch on MIG; the controller submits at most two jobs concurrently.
# Usage: sbatch basis_a5_unique_shared.sh <rank-stability|construct|preflight|atlas> ...
# Atlas args: <model> <dataset> <side> <direction> [layer_start] [layer_end]
# Construction/preflight args: <model>
#SBATCH --job-name=basis_a5
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/basis_a5_%x_%j.out
#SBATCH --error=logs/basis_a5_%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
COMMAND="${1:?command required}"; shift
case "$COMMAND" in
  rank-stability)
    exec "$PY" experiments/basis_a5_unique_shared.py rank-stability ;;
  construct|preflight)
    MODEL="${1:?model required}"; shift
    exec "$PY" experiments/basis_a5_unique_shared.py "$COMMAND" --model "$MODEL" "$@" ;;
  atlas)
    MODEL="${1:?model required}"; DATASET="${2:?dataset required}"; SIDE="${3:?side required}"; DIRECTION="${4:?direction required}"
    START="${5:-0}"; END="${6:-}"
    ARGS=(atlas --model "$MODEL" --dataset "$DATASET" --side "$SIDE" --direction "$DIRECTION" --layer-start "$START")
    [ -n "$END" ] && ARGS+=(--layer-end "$END")
    exec "$PY" experiments/basis_a5_unique_shared.py "${ARGS[@]}" ;;
  *) echo "unknown command: $COMMAND" >&2; exit 2;;
esac
