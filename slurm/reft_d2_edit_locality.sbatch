#!/bin/bash
#SBATCH --job-name=reft-d2
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:55:00
#SBATCH --output=/mnt/data/tungnx/slurm_logs/reft-d2-%j.out
#SBATCH --error=/mnt/data/tungnx/slurm_logs/reft-d2-%j.err
# D2 GPU steps (canonical CUDA 12.8 env). CMD: calibrate | preflight | train SEED | verify | mech | prefix
# Pre-freeze dry runs: sbatch --export=ALL,D2_DRY_ROOT=<scratch dir> slurm/reft_d2_edit_locality.sbatch preflight
set -euo pipefail
cd /home/tungnx/cs-asr-reft
export TMPDIR=/mnt/data/tungnx/tmp/reft-$SLURM_JOB_ID; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONPATH="$PWD/src:$PWD"
NVL=$(ls -d /home/tungnx/miniconda3/envs/acl1/lib/python3.11/site-packages/nvidia/*/lib | tr '\n' ':')
export LD_LIBRARY_PATH="${NVL}/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONFAULTHANDLER=1 SYMPY_GROUND_TYPES=python OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
CMD="${1:?usage: sbatch slurm/reft_d2_edit_locality.sbatch calibrate|preflight|train SEED|verify|mech|prefix}"
"$PY" -m pytest -q -p no:cacheprovider tests/reft/test_d2_edit_locality.py
case "$CMD" in
  calibrate) "$PY" -u reft/d2_edit_locality_train.py calibrate ;;
  preflight) "$PY" -u reft/d2_edit_locality_train.py preflight ;;
  train) "$PY" -u reft/d2_edit_locality_train.py train --seed "${2:?seed}" ;;
  verify) "$PY" -u reft/d2_edit_locality_train.py verify ;;
  mech) "$PY" -u reft/d2_edit_locality_evaluate.py mech ;;
  prefix) "$PY" -u reft/d2_edit_locality_evaluate.py prefix ;;
  *) echo "unknown $CMD"; exit 2 ;;
esac
