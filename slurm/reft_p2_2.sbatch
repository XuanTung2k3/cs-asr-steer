#!/bin/bash
#SBATCH --job-name=reft-p22
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:55:00
#SBATCH --output=/mnt/data/tungnx/slurm_logs/reft-p22-%j.out
#SBATCH --error=/mnt/data/tungnx/slurm_logs/reft-p22-%j.err
# P2-2 GPU steps (canonical CUDA 12.8 env). CMD: preflight | train SEED | verify | mech-replay
set -euo pipefail
cd /home/tungnx/cs-asr-reft
export TMPDIR=/mnt/data/tungnx/tmp/reft-$SLURM_JOB_ID; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONPATH="$PWD/src:$PWD"
NVL=$(ls -d /home/tungnx/miniconda3/envs/acl1/lib/python3.11/site-packages/nvidia/*/lib | tr '\n' ':')
export LD_LIBRARY_PATH="${NVL}/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONFAULTHANDLER=1 SYMPY_GROUND_TYPES=python OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
CMD="${1:?usage: sbatch slurm/reft_p2_2.sbatch preflight|train SEED|verify|mech-replay}"
"$PY" -m pytest -q -p no:cacheprovider tests/reft/test_p2_2.py
case "$CMD" in
  preflight) "$PY" -u reft/p2_2_train.py preflight ;;
  train) "$PY" -u reft/p2_2_train.py train --seed "${2:?seed}" ;;
  verify) "$PY" -u reft/p2_2_train.py verify ;;
  mech-replay) "$PY" -u reft/p2_2_evaluate.py mech-replay ;;
  *) echo "unknown $CMD"; exit 2 ;;
esac
