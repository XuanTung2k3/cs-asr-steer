#!/bin/bash
# Minimal A6 GPU acceptance only. All GPU execution must enter through sbatch.
# This script is deliberately not an atlas launcher.
#SBATCH --job-name=basis-a6-gpu-accept
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=results/basis_a6_expanded/preflight/gpu-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/a6_env.sh
a6_print_environment 1
"${BASIS_A6_PYTHON}" scripts/basis_a6_gpu_acceptance.py
