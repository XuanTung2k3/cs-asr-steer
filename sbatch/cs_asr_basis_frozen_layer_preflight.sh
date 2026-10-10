#!/usr/bin/env bash
# BASIS-A2 fixed worker/resource preflight only.
#SBATCH --job-name=cs-basis-a2-pf
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/tungnx/cs-asr-steer/logs/cs_basis_a2_pf_%j.out
#SBATCH --error=/mnt/data/tungnx/cs-asr-steer/logs/cs_basis_a2_pf_%j.err
set -euo pipefail
REPO=/home/tungnx/cs-asr-steer
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
export LD_LIBRARY_PATH=/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="${REPO}/src:${REPO}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "${REPO}"; mkdir -p /mnt/data/tungnx/cs-asr-steer/logs
exec "${PY}" experiments/basis_frozen_layer_atlas.py --mode preflight-workers
