#!/bin/bash
#SBATCH --job-name=reft-p22-cpu
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/data/tungnx/slurm_logs/reft-p22-cpu-%j.out
#SBATCH --error=/mnt/data/tungnx/slurm_logs/reft-p22-cpu-%j.err
# P2-2 CPU steps (no GPU). CMD: test | freeze | select | evaluate | mech-anchors | report
set -euo pipefail
cd /home/tungnx/cs-asr-reft
export TMPDIR=/mnt/data/tungnx/tmp/reft-$SLURM_JOB_ID; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONPATH="$PWD/src:$PWD" MPLCONFIGDIR="$TMPDIR/mpl"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
CMD="${1:?usage: sbatch slurm/reft_p2_2_cpu.sbatch test|freeze|select|evaluate|mech-anchors|report}"
case "$CMD" in
  test) "$PY" -m pytest -q -p no:cacheprovider tests/reft ;;
  freeze) "$PY" -m pytest -q -p no:cacheprovider tests/reft
          "$PY" -u reft/p2_2_train.py freeze ;;
  select|evaluate|mech-anchors|report) "$PY" -u reft/p2_2_evaluate.py "$CMD" ;;
  *) echo "unknown $CMD"; exit 2 ;;
esac
