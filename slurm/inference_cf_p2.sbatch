#!/bin/bash
#SBATCH --job-name=cf-p2
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
# Submit with: sbatch --output=<out>/slurm-%j.out --error=<out>/slurm-%j.err --export=ALL,OUT=<out> slurm/inference_cf_p2.sbatch
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${OUT:?OUT (results directory with a frozen P2 manifest) is required}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec /home/tungnx/miniconda3/envs/acl1/bin/python experiments/inference_cf_p2.py --out "$OUT"
