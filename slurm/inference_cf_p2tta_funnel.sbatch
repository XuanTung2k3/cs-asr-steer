#!/bin/bash
#SBATCH --job-name=cf-p2ttaf
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
# P2-TTA-FUNNEL single-stage allocation (one pending/running job at most; stages never concurrent).
# Submit with:
#   sbatch --output=<OUT>/slurm-%j.out --error=<OUT>/slurm-%j.err --export=ALL,OUT=<OUT>,STAGE=tta1 slurm/inference_cf_p2tta_funnel.sbatch
# Requires the committed stage pre-audit PASS and a frozen manifest in OUT.
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${OUT:?OUT required}"
: "${STAGE:?STAGE required (tta1)}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
test -f "$OUT/manifest.json"
case "$STAGE" in
  tta1)
    grep -q '"verdict": "PASS_TO_P2_TTA1"' results/inference_cf/p2tta_funnel/tta1/prerun_audit.json
    exec "$PY" experiments/inference_cf_p2tta_funnel.py tta1-run --out "$OUT" ;;
  *) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
