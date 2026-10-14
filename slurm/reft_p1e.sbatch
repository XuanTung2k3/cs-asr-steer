#!/bin/bash
#SBATCH --job-name=reft-p1e
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/tungnx/slurm_logs/reft-p1e-%j.out
#SBATCH --error=/mnt/data/tungnx/slurm_logs/reft-p1e-%j.err
# ReFT P1-E decoupled read/write subspace audit (CPU only, no GPU requested). CMD: test | audit | run
set -euo pipefail
cd /home/tungnx/cs-asr-reft
export TMPDIR=/mnt/data/tungnx/tmp/reft-$SLURM_JOB_ID
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
CMD="${1:?usage: sbatch slurm/reft_p1e.sbatch test|audit|build|run}"
case "$CMD" in
  test) "$PY" -m pytest -q -p no:cacheprovider tests/reft/test_decoupled_audit.py ;;
  audit) "$PY" -u reft/decoupled_audit.py audit ;;
  build) "$PY" -u reft/decoupled_audit.py build ;;
  run) "$PY" -u reft/decoupled_audit.py run ;;
  *) echo "unknown $CMD"; exit 2 ;;
esac
