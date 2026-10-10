#!/bin/bash
# BASIS-A4 Whisper wave.  One panel per Slurm job; no layer-per-job fanout.
# Submit no more than two of these jobs at once.
#SBATCH --job-name=basis_a4_whisper
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/basis_a4_%x_%j.out
#SBATCH --error=logs/basis_a4_%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
dataset="${1:?dataset required}"; shift
exec /home/tungnx/miniconda3/envs/acl1/bin/python experiments/basis_a4.py whisper-conditioning --dataset "${dataset}" "$@"
