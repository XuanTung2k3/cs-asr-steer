#!/bin/bash
#SBATCH --job-name=cf-p1
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/tungnx/cs-asr-steer-inf/results/inference_cf/p1/slurm-%j.out
#SBATCH --error=/home/tungnx/cs-asr-steer-inf/results/inference_cf/p1/slurm-%j.err
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec /home/tungnx/miniconda3/envs/acl1/bin/python experiments/inference_cf_p1.py --out results/inference_cf/p1
