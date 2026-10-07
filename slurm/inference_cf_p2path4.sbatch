#!/bin/bash
#SBATCH --job-name=cf-path4
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
# P2-PATH4-R1 (EOS_BOUNDARY_V1): one allocation, 100 rows, two resident instances: all-100 exact A2 reconstruction barrier, then R1 G1 (CONTENT_G1 / EOS_BOUNDARY H=1). Hard ceiling 30 min.
# Submit with:
#   sbatch --output=<OUT>/slurm-%j.out --error=<OUT>/slurm-%j.err --export=ALL,OUT=<OUT> slurm/inference_cf_p2path4.sbatch
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${OUT:?OUT required}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
test -f "$OUT/manifest.json"
grep -q '"verdict": "PASS_TO_P2_PATH4_R1"' results/inference_cf/p2path4/prerun_audit.json
exec "$PY" experiments/inference_cf_p2path4.py run --out "$OUT"
