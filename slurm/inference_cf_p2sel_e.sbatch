#!/bin/bash
#SBATCH --job-name=cf-p2sel-e
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
# Submit with:
#   sbatch --output=<OUT>/slurm-%j.out --error=<OUT>/slurm-%j.err --export=ALL,STAGE=<e0|e1>,OUT=<OUT> slurm/inference_cf_p2sel_e.sbatch
# STAGE=e0 requires PASS_TO_P2_SEL_E_E0; STAGE=e1 additionally requires the pushed pre-E1 audit PASS. Manifest in OUT.
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${OUT:?OUT (results directory with a frozen P2-SEL-E manifest) is required}"
: "${STAGE:?STAGE (e0|e1) is required}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
test -f "$OUT/manifest.json"
grep -q '"verdict": "PASS_TO_P2_SEL_E_E0"' results/inference_cf/p2sel_e/prerun_audit_r1.json
case "$STAGE" in
  e0) "$PY" experiments/inference_cf_p2sel_e.py e0 --out "$OUT" ;;
  e1) grep -q '"verdict": "P2_SEL_E_PRE_E1_AUDIT: PASS"' results/inference_cf/p2sel_e/pre_e1_audit.json
      "$PY" experiments/inference_cf_p2sel_e.py e1 --out "$OUT" ;;
  *) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
