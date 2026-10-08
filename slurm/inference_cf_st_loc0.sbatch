#!/bin/bash
#SBATCH --job-name=cf-stloc0
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
# ST-LOC0: ONE allocation. Passive 4-site B/E replay -> site-specific V1/V2 calibration -> calibration seal, then WAIT for
# GO.json (written on the login node only after the calibration seal is committed and pushed) -> L16 DG-02 historical barrier
# -> remaining 14 primary arms + 4 random controls. Target 15-30 min of compute, hard 3 h (includes the GO wait).
# Submit with:
#   sbatch --output=<OUT>/slurm-%j.out --error=<OUT>/slurm-%j.err --export=ALL,OUT=<OUT> slurm/inference_cf_st_loc0.sbatch
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${OUT:?OUT required}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
test -f "$OUT/manifest.json"
grep -q '"verdict": "PASS_TO_ST_LOC0"' results/inference_cf/st_loc0/prerun_audit.json
exec "$PY" experiments/inference_cf_st_loc0.py run --out "$OUT"
