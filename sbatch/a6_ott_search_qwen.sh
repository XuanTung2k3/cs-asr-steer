#!/usr/bin/env bash
#SBATCH --job-name=a6ott-q
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%1
#SBATCH --output=results/a6_ott_upper_bound/logs/search-q-%A_%a.out
#SBATCH --error=results/a6_ott_upper_bound/logs/search-q-%A_%a.err

set -euo pipefail
cd /home/tungnx/cs-asr-steer
source slurm/a6_env.sh
a6_print_environment 1
mapfile -t SHARDS < <("${BASIS_A6_PYTHON}" -m csasr.experiments.a6_ott --model qwen --list-shards)
read -r DATASET SIDE START STOP <<< "${SHARDS[${SLURM_ARRAY_TASK_ID}]}"
echo "experiment_phase=search model=qwen3_asr_1p7b shard=${DATASET}/${SIDE}/L${START}:L${STOP}"
exec "${BASIS_A6_PYTHON}" -m csasr.experiments.a6_ott --model qwen \
  --dataset "${DATASET}" --side "${SIDE}" --layer-start "${START}" --layer-stop "${STOP}"
