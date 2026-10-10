#!/bin/bash
# BASIS-A3: exactly the harmonized SEAME encoder Oracle-local repair.
# Existing Global cells are resumed; only quarantined local cells are decoded.
#SBATCH --job-name=basis_a3_seame_repair
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/basis_a3_%x_%j.out
#SBATCH --error=logs/basis_a3_%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
DATASET="${1:?dataset argument required}"
export BASIS_A3_WORKERS=2
"$PY" experiments/basis_a3.py grid --dataset "$DATASET" --side encoder --direction Raw --stage r1
"$PY" experiments/basis_a3.py grid --dataset "$DATASET" --side encoder --direction Raw --stage r2
