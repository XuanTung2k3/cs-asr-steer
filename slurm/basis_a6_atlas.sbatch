#!/usr/bin/env bash
#SBATCH --job-name=basis-a6-atlas
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:55:00
#SBATCH --output=results/basis_a6_expanded/slurm/atlas-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/a6_env.sh
a6_print_environment 1
MODEL="${BASIS_A6_MODEL:?BASIS_A6_MODEL must be set}"
STAGE="${BASIS_A6_STAGE:?BASIS_A6_STAGE must be fixed or oracle_tt}"
DATASET="${BASIS_A6_DATASET:?BASIS_A6_DATASET must be set}"
MODE="${BASIS_A6_DECODE_MODE:?BASIS_A6_DECODE_MODE must be set}"
SIDE="${BASIS_A6_SIDE:?BASIS_A6_SIDE must be set}"
LAYER="${BASIS_A6_LAYER:?BASIS_A6_LAYER must be set}"
ARGS=(--stage "${STAGE}" --model "${MODEL}" --dataset "${DATASET}" --decode-mode "${MODE}" --side "${SIDE}" --layer "${LAYER}")
if [[ "${STAGE}" == "fixed" ]]; then
  ARGS+=(--source "${BASIS_A6_SOURCE:?BASIS_A6_SOURCE must be set for fixed}")
fi
if [[ "${BASIS_A6_PILOT:-0}" == "1" ]]; then ARGS+=(--pilot); fi
exec "${BASIS_A6_PYTHON}" scripts/basis_a6_execute.py "${ARGS[@]}"
