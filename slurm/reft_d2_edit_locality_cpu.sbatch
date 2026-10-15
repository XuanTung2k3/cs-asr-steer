#!/bin/bash
#SBATCH --job-name=reft-d2-cpu
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/data/tungnx/slurm_logs/reft-d2-cpu-%j.out
#SBATCH --error=/mnt/data/tungnx/slurm_logs/reft-d2-cpu-%j.err
# D2 CPU steps (no GPU). CMD: test | mask | freeze | freeze-amend DOC | select | evaluate | report
# Pre-freeze dry runs: sbatch --export=ALL,D2_DRY_ROOT=<scratch dir> slurm/reft_d2_edit_locality_cpu.sbatch mask
set -euo pipefail
cd /home/tungnx/cs-asr-reft
export TMPDIR=/mnt/data/tungnx/tmp/reft-$SLURM_JOB_ID; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONPATH="$PWD/src:$PWD" MPLCONFIGDIR="$TMPDIR/mpl"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
CMD="${1:?usage: sbatch slurm/reft_d2_edit_locality_cpu.sbatch test|mask|freeze|freeze-amend DOC|select|evaluate|report}"
case "$CMD" in
  test) "$PY" -m pytest -q -p no:cacheprovider tests/reft ;;
  mask) "$PY" -m pytest -q -p no:cacheprovider tests/reft/test_d2_edit_locality.py
           "$PY" -u reft/d2_edit_locality_train.py mask ;;
  freeze) "$PY" -m pytest -q -p no:cacheprovider tests/reft
          "$PY" -u reft/d2_edit_locality_train.py freeze ;;
  freeze-amend) "$PY" -m pytest -q -p no:cacheprovider tests/reft
                "$PY" -u reft/d2_edit_locality_train.py freeze-amend --doc "${2:?amendment doc}" ;;
  select|evaluate|report) "$PY" -u reft/d2_edit_locality_evaluate.py "$CMD" ;;
  *) echo "unknown $CMD"; exit 2 ;;
esac
