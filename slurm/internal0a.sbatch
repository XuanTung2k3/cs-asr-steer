#!/bin/bash
#SBATCH --job-name=internal0a
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
# Engineering smoke only currently authorized; capture requires later approval.
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${PHASE:?use smoke or capture}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
case "$PHASE" in
  smoke) "$PY" experiments/internal0a.py smoke ;;
  capture)
    test "${APPROVED_FULL_CAPTURE:-NO}" = YES
    "$PY" experiments/internal0a.py capture --approved-full-capture ;;
  *) echo "invalid phase"; exit 2 ;;
esac
