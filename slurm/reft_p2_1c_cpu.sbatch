#!/bin/bash
#SBATCH --job-name=reft-p21c-cpu
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/tungnx/slurm_logs/reft-p21c-cpu-%j.out
#SBATCH --error=/mnt/data/tungnx/slurm_logs/reft-p21c-cpu-%j.err
# P2-1c CPU steps (no GPU requested). CMD: test | prep (tests, freeze, anchors) | analyze | freeze-r1 | analyze-r1
set -euo pipefail
cd /home/tungnx/cs-asr-reft
export TMPDIR=/mnt/data/tungnx/tmp/reft-$SLURM_JOB_ID; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONPATH="$PWD/src:$PWD" MPLCONFIGDIR="$TMPDIR/mpl"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
CMD="${1:?usage: sbatch slurm/reft_p2_1c_cpu.sbatch test|prep|analyze|freeze-r1|analyze-r1}"
case "$CMD" in
  test) "$PY" -m pytest -q -p no:cacheprovider tests/reft ;;
  prep) "$PY" -m pytest -q -p no:cacheprovider tests/reft
        "$PY" -u reft/p2_1c_continuation_audit.py freeze
        "$PY" -u reft/p2_1c_continuation_audit.py anchors ;;
  analyze) "$PY" -u reft/p2_1c_continuation_audit.py analyze ;;
  freeze-r1) "$PY" -m pytest -q -p no:cacheprovider tests/reft
             "$PY" -u reft/p2_1c_continuation_audit.py freeze-r1 ;;
  analyze-r1) "$PY" -u reft/p2_1c_continuation_audit.py analyze-r1 ;;
  *) echo "unknown $CMD"; exit 2 ;;
esac
