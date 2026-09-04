#!/usr/bin/env bash
# =============================================================================
# Track A -- inference-only steering sweep. One GPU, no training.
#
#   sbatch sbatch/study_track_a.sh --smoke          # always run smoke first
#   sbatch sbatch/study_track_a.sh --all
#   sbatch sbatch/study_track_a.sh --all --resume
#
# --smoke writes to out/smoke/, never out/: a smoke cell and its full-scale
# counterpart share the same config hash, so sharing one directory would let
# --resume silently treat 8-utterance smoke numbers as the real result.
#
# The task specification writes this as
#     CUDA_VISIBLE_DEVICES=0  python run_study.py --track A --all
# This cluster submits through Slurm, so `--gres=gpu:1` supplies the device and
# Slurm sets CUDA_VISIBLE_DEVICES itself. Track A and Track B are two SEPARATE
# jobs and Slurm gives them separate GPUs; they must never share one.
#
# EXIT CODES
#   0   the sweep ran to completion and wrote out/SUMMARY.md. The scientific
#       outcome is in VERDICT_TRACK_A, not in this exit code.
#   1   preflight failed, or an assertion fired. Nothing downstream is valid.
#   64  usage error.
#
# Confirm is NEVER run here. The selected configuration is printed to the log
# and written to out/confirm_candidate.json for human approval.
# =============================================================================
#SBATCH --job-name=cs_study_A
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=/mnt/data/tungnx/cs-asr-steer/logs/cs_study_A_%j.out
#SBATCH --error=/mnt/data/tungnx/cs-asr-steer/logs/cs_study_A_%j.err

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

exec "${PY}" run_study.py --track A "$@"
