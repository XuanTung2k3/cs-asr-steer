#!/bin/bash
#SBATCH --job-name=reft-p1
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/mnt/data/tungnx/slurm_logs/reft-p1-%j.out
#SBATCH --error=/mnt/data/tungnx/slurm_logs/reft-p1-%j.err
# ReFT P1 GPU steps. CMD: smoke | capture | leverage
set -euo pipefail
cd /home/tungnx/cs-asr-reft
export TMPDIR=/mnt/data/tungnx/tmp/reft-$SLURM_JOB_ID
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONFAULTHANDLER=1 SYMPY_GROUND_TYPES=python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
CMD="${1:?usage: sbatch slurm/reft_p1.sbatch smoke|capture|leverage}"
"$PY" -m pytest -q -p no:cacheprovider tests/reft
case "$CMD" in
  smoke|capture) "$PY" -u reft/gradient_capture.py "$CMD" ;;
  leverage) "$PY" -u reft/evaluate_basis.py leverage ;;
  *) echo "unknown $CMD"; exit 2 ;;
esac
