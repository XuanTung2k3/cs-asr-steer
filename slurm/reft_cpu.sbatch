#!/bin/bash
#SBATCH --job-name=reft-cpu
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/tungnx/slurm_logs/reft-cpu-%j.out
#SBATCH --error=/mnt/data/tungnx/slurm_logs/reft-cpu-%j.err
# ReFT CPU steps (no GPU requested). CMD: test | audit-p0 | audit-p1 | p0-seal | p0-evaluate | basis | p1-analyze
set -euo pipefail
cd /home/tungnx/cs-asr-reft
export TMPDIR=/mnt/data/tungnx/tmp/reft-$SLURM_JOB_ID
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
CMD="${1:?usage: sbatch slurm/reft_cpu.sbatch test|audit-p0|audit-p1|p0-seal|p0-evaluate|basis|p1-analyze}"
case "$CMD" in
  test) "$PY" -m pytest -q -p no:cacheprovider tests/reft ;;
  audit-p0) "$PY" -u reft/data_audit.py p0 ;;
  audit-p1) "$PY" -u reft/data_audit.py p1 ;;
  p0-seal) "$PY" -u reft/baseline_auto.py seal ;;
  p0-evaluate) "$PY" -u reft/baseline_auto.py evaluate ;;
  basis) "$PY" -u reft/build_basis.py ;;
  p1-analyze) "$PY" -u reft/evaluate_basis.py analyze ;;
  *) echo "unknown $CMD"; exit 2 ;;
esac
