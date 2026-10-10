#!/bin/bash
#SBATCH --job-name=cf-ttls-r1
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
# TTLS-R1: ONE coordinated allocation, fixed-100 panel, all frozen arms + comparators. Hard ceiling 3 h.
# Submit with:
#   sbatch --output=<OUT>/slurm-%j.out --error=<OUT>/slurm-%j.err --export=ALL,OUT=<OUT> slurm/inference_cf_ttls_r1.sbatch
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${OUT:?OUT required}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
test -f "$OUT/manifest.json"
exec "$PY" experiments/inference_cf_ttls_r1.py run --out "$OUT"
