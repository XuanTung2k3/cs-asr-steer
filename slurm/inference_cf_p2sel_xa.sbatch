#!/bin/bash
#SBATCH --job-name=cf-p2sel-xa
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
# Submit with:
#   sbatch --output=<OUT>/slurm-%j.out --error=<OUT>/slurm-%j.err --export=ALL,STAGE=xa0,OUT=<OUT> slurm/inference_cf_p2sel_xa.sbatch
# STAGE=xa0 requires the committed PASS_TO_P2_SEL_XA_XA0 pre-run audit and a frozen manifest in OUT.
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${OUT:?OUT (results directory with a frozen P2-SEL-XA manifest) is required}"
: "${STAGE:?STAGE (xa0) is required}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
test -f "$OUT/manifest.json"
grep -q '"verdict": "PASS_TO_P2_SEL_XA_XA0"' results/inference_cf/p2sel_xa/prerun_audit.json
case "$STAGE" in
  xa0) "$PY" experiments/inference_cf_p2sel_xa.py xa0 --out "$OUT" ;;
  *) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
