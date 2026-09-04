#!/usr/bin/env bash
# =============================================================================
# Stage D-extended -- dense (layer x alpha x beam) landscape map for Track A.
#
#   sbatch sbatch/study_stage_d_ext.sh --resume
#   sbatch sbatch/study_stage_d_ext.sh --resume --budget-hours 5
#
# This is a landscape map, not a search: it answers whether the earlier
# Stage D GO cell (decoder layer 28) reflects structure or a draw from a
# 21-cell null, by mapping 70 cells (7 layers x 5 alphas x 2 beam settings)
# densely and looking for connected regions of improvement, never a "best
# cell". Phase 2 (fine layer resolution + multi-layer pairs) runs only if
# Phase 1 finds a connected component of size >= 4; otherwise the null map
# IS the finding and nothing further runs.
#
# Order, per the task: existing 15-check Track A preflight (unmodified) ->
# section 4.1 determinism check (must pass before ANY sweep cell runs, or the
# harness stops without touching the grid) -> fresh C00/C00_beam5 baselines,
# asserted against the prior anchors -> Phase 1 (70 cells) -> section 4.2
# small-alpha sanity gate -> Phase 1 analysis -> Phase 2 (conditional) ->
# out/SUMMARY_D_EXT.md.
#
# Checkpointed to out/results_D_ext.jsonl; --resume skips completed cells by
# config hash, same convention as run_study.py.
#
# EXIT CODES
#   0   ran to completion (or stopped cleanly on --budget-hours) and wrote
#       out/SUMMARY_D_EXT.md. There is no "GO" exit code here: the phase
#       proposes no Confirm candidate and picks no winner.
#   1   the existing preflight failed, the determinism check failed (the
#       sweep was NOT run in that case), the small-alpha sanity gate failed,
#       or a baseline did not reproduce its prior anchor.
#   64  usage error.
#
# D-dev-confirm is never opened by this script.
# =============================================================================
#SBATCH --job-name=cs_study_D_ext
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/tungnx/cs-asr-steer/logs/cs_study_D_ext_%j.out
#SBATCH --error=/mnt/data/tungnx/cs-asr-steer/logs/cs_study_D_ext_%j.err

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

exec "${PY}" run_stage_d_ext.py "$@"
