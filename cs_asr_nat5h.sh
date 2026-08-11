#!/usr/bin/env bash
#SBATCH --job-name=cs_asr_nat5h
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/mnt/data/tungnx/cs-asr-steer/logs/cs_asr_nat5h_%j.out
#SBATCH --error=/mnt/data/tungnx/cs-asr-steer/logs/cs_asr_nat5h_%j.err

set -euo pipefail

COMMAND="${1:-run}"
case "${COMMAND}" in
  run|align_only|resume|qwen_smoke|resume_after_alignment) ;;
  *)
    echo "usage: sbatch cs_asr_nat5h.sh {run|align_only|resume|qwen_smoke|resume_after_alignment}" >&2
    exit 2
    ;;
esac

CSASR_REPO_ROOT="/home/tungnx/cs-asr-steer"
CSASR_ENV_ROOT="/home/tungnx/miniconda3/envs/acl1"
CSASR_OUTPUT_ROOT="/mnt/data/tungnx/cs-asr-steer/artifacts_nat5h_r2"
CSASR_LOG_ROOT="/mnt/data/tungnx/cs-asr-steer/logs"
CSASR_STOP_FILE="${CSASR_OUTPUT_ROOT}/REQUEST_STOP"

mkdir -p "${CSASR_OUTPUT_ROOT}/status" "${CSASR_LOG_ROOT}"
rm -f "${CSASR_STOP_FILE}"

trap 'echo "USR1 received: requesting clean NAT5H checkpoint/summary"; touch "${CSASR_STOP_FILE}"' USR1

cd "${CSASR_REPO_ROOT}"

export PATH="${CSASR_ENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CSASR_ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${CSASR_REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CSASR_AUDIO_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME="/mnt/data/tungnx/cs-asr-steer/model_cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/mnt/data/tungnx/cs-asr-steer/model_cache/huggingface/hub"
export TORCH_HOME="/mnt/data/tungnx/cs-asr-steer/model_cache/torch"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TORCH_HOME}"

if command -v squeue >/dev/null 2>&1; then
  ACTIVE_NAT5H_JOBS="$(squeue -h -u "${USER}" -n cs_asr_nat5h -o '%i' | awk -v self="${SLURM_JOB_ID:-}" '$1 != self {print $1}')"
  if [[ -n "${ACTIVE_NAT5H_JOBS}" ]]; then
    echo "Another cs_asr_nat5h job is active or queued: ${ACTIVE_NAT5H_JOBS}" >&2
    exit 2
  fi
fi

echo "NAT5H command: ${COMMAND}"
echo "Job id: ${SLURM_JOB_ID:-local}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "HF_HOME=${HF_HOME}"
echo "TORCH_HOME=${TORCH_HOME}"

python -m csasr.experiments.nat5h_pipeline "${COMMAND}" --config configs/nat5h.yaml
