#!/usr/bin/env bash
#SBATCH --job-name=basis-a6-ascend-align
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/basis_a6_expanded/slurm/ascend-align-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/a6_env.sh
a6_print_environment 1
exec "${BASIS_A6_PYTHON}" scripts/basis_a6_ascend_eval_alignment.py --device cuda --output results/basis_a6_expanded/ascend/ASCEND_EVAL_ALIGNMENT.gpu.json
