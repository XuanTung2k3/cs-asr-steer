#!/bin/bash
#SBATCH --job-name=cf-p2sel-t
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
# Submit with:
#   sbatch --output=<OUT>/slurm-%j.out --error=<OUT>/slurm-%j.err --export=ALL,STAGE=<t0|t1>,OUT=<OUT> slurm/inference_cf_p2sel_t.sbatch
# STAGE=t0 requires PASS_TO_P2_SEL_T_T0; STAGE=t1 additionally requires the pushed pre-T1 audit PASS. Manifest in OUT.
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${OUT:?OUT (results directory with a frozen P2-SEL-T manifest) is required}"
: "${STAGE:?STAGE (t0|t1) is required}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
test -f "$OUT/manifest.json"
grep -q '"verdict": "PASS_TO_P2_SEL_T_T0"' results/inference_cf/p2sel_t/prerun_audit.json
case "$STAGE" in
  t0) "$PY" experiments/inference_cf_p2sel_t.py t0 --out "$OUT" ;;
  t1) grep -q '"verdict": "P2_SEL_T_PRE_T1_AUDIT: PASS"' results/inference_cf/p2sel_t/pre_t1_audit.json
      "$PY" experiments/inference_cf_p2sel_t.py t1 --out "$OUT" ;;
  *) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
