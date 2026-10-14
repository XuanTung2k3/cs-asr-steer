#!/bin/bash
#SBATCH --job-name=reft-p2-cpu
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/tungnx/slurm_logs/reft-p2-cpu-%j.out
#SBATCH --error=/mnt/data/tungnx/slurm_logs/reft-p2-cpu-%j.err
# ReFT P2 CPU steps (no GPU requested). CMD: test | audit | score-baseline | select RUN_DIR | evaluate RUN_DIR...
set -euo pipefail
cd /home/tungnx/cs-asr-reft
export TMPDIR=/mnt/data/tungnx/tmp/reft-$SLURM_JOB_ID
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
CMD="${1:?usage: sbatch slurm/reft_p2_cpu.sbatch test|audit|score-baseline|select DIR|evaluate DIR...}"
case "$CMD" in
  test) "$PY" -m pytest -q -p no:cacheprovider tests/reft ;;
  audit) "$PY" -u reft/train_supervised.py audit ;;
  score-baseline) "$PY" -u reft/evaluate_supervised.py score-baseline ;;
  select) "$PY" -u reft/evaluate_supervised.py select --run "${2:?run dir}" ;;
  evaluate) shift; "$PY" -u reft/evaluate_supervised.py evaluate --runs "$@" ;;
  *) echo "unknown $CMD"; exit 2 ;;
esac
