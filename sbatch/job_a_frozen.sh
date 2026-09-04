#!/usr/bin/env bash
# Job A: frozen steering confirmation, one independent H100 allocation.
# Submit with: sbatch sbatch/job_a_frozen.sh [--dry-run] [--skip-seame]
#SBATCH --job-name=cs_job_a
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/tungnx/cs-asr-steer/logs/job_a_%j.out
#SBATCH --error=/mnt/data/tungnx/cs-asr-steer/logs/job_a_%j.err

set -euo pipefail

REPO=/home/tungnx/cs-asr-steer
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
export LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="${REPO}/src:${REPO}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${REPO}"
mkdir -p /mnt/data/tungnx/cs-asr-steer/logs results/job_a
echo "job=${SLURM_JOB_ID:-none} host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

exec "${PY}" experiments/job_a_frozen.py "$@"
