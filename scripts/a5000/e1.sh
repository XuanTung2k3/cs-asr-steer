#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$HOME/tung.nx/a5000-bootstrap/tmp"
mkdir -p "$TMPDIR"
exec "$HOME/tung.nx/reft-a5000/bin/python" -u -m reft.a5000.runner "$@"
