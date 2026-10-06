#!/bin/bash
#SBATCH --job-name=cf-p2dir
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
# Submit with:
#   sbatch --output=<OUT>/slurm-%j.out --error=<OUT>/slurm-%j.err --export=ALL,STAGE=<extract|exp1>,OUT=<OUT> slurm/inference_cf_p2dir.sbatch
# STAGE=extract: unedited B/E states only (no outcome). STAGE=exp1: sealed D0/D1/D2 pulses, then the
# post-seal evaluator Jacobian pass, then the independent D2 GPU spot-check (audit module).
# Each stage refuses to run without its own frozen manifest (written by inference_cf_p2dir_prepare.py).
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${OUT:?OUT (results directory with a frozen P2-DIR manifest) is required}"
: "${STAGE:?STAGE (extract|exp1) is required}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
test -f "$OUT/manifest.json"
case "$STAGE" in
  extract)
    "$PY" experiments/inference_cf_p2dir.py extract --out "$OUT" ;;
  exp1)
    test -f results/inference_cf/p2dir/prerun_audit.json
    grep -q '"verdict": "PASS_TO_P2_DIR_EXP1"' results/inference_cf/p2dir/prerun_audit.json
    "$PY" experiments/inference_cf_p2dir.py exp1 --out "$OUT"
    "$PY" experiments/inference_cf_p2dir.py exp1-eval --out "$OUT"
    "$PY" experiments/inference_cf_p2dir_audit.py spot --run "$OUT" --out "$OUT/audit_spot.json" ;;
  *)
    echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
