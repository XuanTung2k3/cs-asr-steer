#!/bin/bash
#SBATCH --job-name=cf-p2sel-lac
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
# Submit with:
#   sbatch --output=<OUT>/slurm-%j.out --error=<OUT>/slurm-%j.err --export=ALL,STAGE=lac0,OUT=<OUT> slurm/inference_cf_p2sel_lac.sbatch
# STAGE=lac0 requires the committed PASS_TO_P2_SEL_LAC_LAC0 pre-run audit and a frozen manifest in OUT.
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${OUT:?OUT (results directory with a frozen P2-SEL-LAC manifest) is required}"
: "${STAGE:?STAGE (lac0) is required}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
test -f "$OUT/manifest.json"
grep -q '"verdict": "PASS_TO_P2_SEL_LAC_LAC0"' results/inference_cf/p2sel_lac/prerun_audit.json
case "$STAGE" in
  lac0) "$PY" experiments/inference_cf_p2sel_lac.py lac0 --out "$OUT" ;;
  *) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
