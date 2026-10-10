#!/usr/bin/env bash
#SBATCH --job-name=a6ott-c-q
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%1
#SBATCH --output=results/a6_ott_upper_bound/logs/transfer-q-%A_%a.out
#SBATCH --error=results/a6_ott_upper_bound/logs/transfer-q-%A_%a.err
set -euo pipefail
cd /home/tungnx/cs-asr-steer
source slurm/a6_env.sh
a6_print_environment 1
case "${SLURM_ARRAY_TASK_ID}" in
  0) DATASET=seame_dev_man; SIDE=encoder;;
  1) DATASET=seame_dev_man; SIDE=decoder;;
  2) DATASET=seame_dev_sge; SIDE=encoder;;
  3) DATASET=seame_dev_sge; SIDE=decoder;;
esac
echo "experiment_phase=transfer model=qwen3_asr_1p7b dataset=${DATASET} side=${SIDE} mode=greedy+official_alias shard=${SLURM_ARRAY_TASK_ID}"
exec "${BASIS_A6_PYTHON}" -m csasr.experiments.a6_ott_phase_bc --model qwen --phase C --dataset "${DATASET}" --side "${SIDE}" --mode greedy
