#!/bin/bash
# BASIS-A3 scientific Slurm entry point. Submit in waves; the caller must
# enforce <=2 concurrent jobs. Scientific work never runs interactively.
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:30:00
#SBATCH --output=logs/basis_a3_%x_%j.out
#SBATCH --error=logs/basis_a3_%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
exec "$PY" experiments/basis_a3.py grid "$@"
