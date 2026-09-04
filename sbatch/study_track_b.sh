#!/usr/bin/env bash
# =============================================================================
# Track B -- learned intervention vs strong CS adapter. One GPU, training.
#
#   sbatch sbatch/study_track_b.sh --smoke          # always run smoke first
#   sbatch sbatch/study_track_b.sh --all
#   sbatch sbatch/study_track_b.sh --all --resume
#
# --smoke writes to out/smoke/, never out/: an arm's config hash does not
# depend on training-set/step-count scale, so sharing one directory would let
# --resume silently treat a 20-step smoke arm as the real 2000-step result.
#
# The task specification writes this as
#     CUDA_VISIBLE_DEVICES=1  python run_study.py --track B --all
# Slurm supplies the device. Submit this as a SEPARATE job from Track A so the
# two never share a GPU: concurrent arms on one device make wall-clock and
# peak-memory numbers meaningless, and section 4.6 forbids it.
#
# Arms run STRICTLY SEQUENTIALLY inside this job, in the order of section 4.6:
#   lambda_lid search on B1 -> lr search on B1 -> B0 -> B1 x3 seeds -> B2 -> B4
#   -> B3-reimpl head selection + two-stage training.
#
# EXIT CODES
#   0   the arms ran to completion and wrote out/SUMMARY.md. The scientific
#       outcome is in VERDICT_TRACK_B, not in this exit code.
#   1   preflight failed (in particular a nonzero frozen-backbone gradient
#       norm, or a B3 parameter fraction off by more than 2x), or an assertion
#       fired.
#   64  usage error.
# =============================================================================
#SBATCH --job-name=cs_study_B
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=/mnt/data/tungnx/cs-asr-steer/logs/cs_study_B_%j.out
#SBATCH --error=/mnt/data/tungnx/cs-asr-steer/logs/cs_study_B_%j.err

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

exec "${PY}" run_study.py --track B "$@"
