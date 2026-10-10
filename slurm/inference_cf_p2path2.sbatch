#!/bin/bash
#SBATCH --job-name=cf-path2
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
# P2-PATH2: one allocation, 12 rows, two resident instances: A4 reconstruction barrier, then online consensus guard G0/G1/G2. Hard ceiling 30 min.
# Submit with:
#   sbatch --output=<OUT>/slurm-%j.out --error=<OUT>/slurm-%j.err --export=ALL,OUT=<OUT> slurm/inference_cf_p2path2.sbatch
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${OUT:?OUT required}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
test -f "$OUT/manifest.json"
grep -q '"verdict": "PASS_TO_P2_PATH2"' results/inference_cf/p2path2/prerun_audit.json
exec "$PY" experiments/inference_cf_p2path2.py run --out "$OUT"
