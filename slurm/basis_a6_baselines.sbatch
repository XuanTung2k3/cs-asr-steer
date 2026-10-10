#!/usr/bin/env bash
#SBATCH --job-name=basis-a6-baselines
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:45:00
#SBATCH --output=results/basis_a6_expanded/slurm/baselines-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/a6_env.sh
a6_print_environment 1
MODEL="${BASIS_A6_MODEL:?BASIS_A6_MODEL must be set}"
exec "${BASIS_A6_PYTHON}" scripts/basis_a6_execute.py --stage baselines --model "${MODEL}"
