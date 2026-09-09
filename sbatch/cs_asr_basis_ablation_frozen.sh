#!/usr/bin/env bash
# BASIS-A: one-process frozen direction ablation, D-dev-select only.
# The scientific matrix is fixed in the pre-run spec and manifest.
#SBATCH --job-name=cs-basis-a
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/tungnx/cs-asr-steer/logs/cs_basis_a_%j.out
#SBATCH --error=/mnt/data/tungnx/cs-asr-steer/logs/cs_basis_a_%j.err

set -euo pipefail
REPO=/home/tungnx/cs-asr-steer
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
export LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="${REPO}/src:${REPO}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "${REPO}"
mkdir -p /mnt/data/tungnx/cs-asr-steer/logs
echo "job ${SLURM_JOB_ID:-none} on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
exec "${PY}" experiments/basis_ablation_frozen.py --mode gpu --batch-size 32
