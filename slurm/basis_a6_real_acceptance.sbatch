#!/bin/bash
# Model-resident BASIS-A6 real acceptance and bounded benchmark. Never an atlas launcher.
#SBATCH --job-name=basis-a6-real
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:45:00
#SBATCH --output=results/basis_a6_expanded/preflight/real-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/a6_env.sh
a6_print_environment 1
MODEL="${BASIS_A6_REAL_MODEL:?BASIS_A6_REAL_MODEL must be whisper or qwen3_asr_1p7b}"
exec "${BASIS_A6_PYTHON}" scripts/basis_a6_real_acceptance.py --model "${MODEL}"
