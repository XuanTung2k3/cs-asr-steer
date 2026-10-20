#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-$HOME/tung.nx/a5000-bootstrap/tmp}"
mkdir -p "$TMPDIR"
exec "${REFT_PYTHON:-$HOME/tung.nx/reft-a5000/bin/python}" "scripts/a5000/validate.py" "$@"
