#!/bin/bash
#SBATCH --job-name=a2p-audit
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -euo pipefail
cd /home/tungnx/cs-asr-steer-a2p-dev200-audit
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /home/tungnx/miniconda3/envs/acl1/bin/python experiments/a2p_dev200_independent_replay.py
