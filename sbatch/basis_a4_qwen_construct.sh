#!/bin/bash
# One Qwen direction-construction worker over CS D-construct only.
#SBATCH --job-name=basis_a4_qwen_construct
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/basis_a4_%x_%j.out
#SBATCH --error=logs/basis_a4_%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
exec /home/tungnx/miniconda3/envs/acl1/bin/python experiments/basis_a4_qwen.py construct
