#!/bin/bash
#SBATCH --job-name=cf-p2sel
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
# Submit with:
#   sbatch --output=<OUT>/slurm-%j.out --error=<OUT>/slurm-%j.err --export=ALL,STAGE=s1,OUT=<OUT> slurm/inference_cf_p2sel.sbatch
# STAGE=s1 requires the committed PASS_TO_P2_SEL_S1 pre-run audit and a frozen manifest in OUT.
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${OUT:?OUT (results directory with a frozen P2-SEL manifest) is required}"
: "${STAGE:?STAGE (s1) is required}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
test -f "$OUT/manifest.json"
grep -q '"verdict": "PASS_TO_P2_SEL_S1"' results/inference_cf/p2sel/prerun_audit.json
case "$STAGE" in
  s1) "$PY" experiments/inference_cf_p2sel.py s1 --out "$OUT" ;;
  *) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
