#!/bin/bash
#SBATCH --job-name=cf-mech0
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
# P2-A2-MECH0: one allocation, FULL300 exact A2 reconstruction barrier (step observer), then MECH_PANEL84 STEP1/DROP decodes and changed44 geometry. Target <30 min, hard 3 h.
# Submit with:
#   sbatch --output=<OUT>/slurm-%j.out --error=<OUT>/slurm-%j.err --export=ALL,OUT=<OUT> slurm/inference_cf_p2a2_mech0.sbatch
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${OUT:?OUT required}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
test -f "$OUT/manifest.json"
grep -q '"verdict": "PASS_TO_P2_A2_MECH0"' results/inference_cf/p2a2_mech0/prerun_audit.json
exec "$PY" experiments/inference_cf_p2a2_mech0.py run --out "$OUT"
