#!/bin/bash
#SBATCH --job-name=reft-p21b
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:50:00
#SBATCH --output=/mnt/data/tungnx/slurm_logs/reft-p21b-%j.out
#SBATCH --error=/mnt/data/tungnx/slurm_logs/reft-p21b-%j.err
# P2-1b GPU steps (canonical CUDA 12.8 env). CMD: smoke | train ARM SEED | verify | prefix
set -euo pipefail
cd /home/tungnx/cs-asr-reft
export TMPDIR=/mnt/data/tungnx/tmp/reft-$SLURM_JOB_ID; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONPATH="$PWD/src:$PWD"
NVL=$(ls -d /home/tungnx/miniconda3/envs/acl1/lib/python3.11/site-packages/nvidia/*/lib | tr '\n' ':')
export LD_LIBRARY_PATH="${NVL}/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONFAULTHANDLER=1 SYMPY_GROUND_TYPES=python OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
CMD="${1:?usage: sbatch slurm/reft_p2_1b.sbatch smoke|train ARM SEED|verify|prefix}"
"$PY" -m pytest -q -p no:cacheprovider tests/reft/test_p2_1b.py tests/reft/test_p2_0.py
case "$CMD" in
  smoke) timeout 2700 "$PY" -u reft/p2_1b_train.py smoke ;;
  train) "$PY" -u reft/p2_1b_train.py train --arm "${2:?arm}" --seed "${3:?seed}" ;;
  verify) "$PY" -u reft/p2_1b_train.py verify ;;
  prefix) "$PY" -u reft/p2_1b_prefix_audit.py ;;
  *) echo "unknown $CMD"; exit 2 ;;
esac
