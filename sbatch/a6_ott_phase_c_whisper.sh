#!/usr/bin/env bash
#SBATCH --job-name=a6ott-c-w
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-7%1
#SBATCH --output=results/a6_ott_upper_bound/logs/transfer-w-%A_%a.out
#SBATCH --error=results/a6_ott_upper_bound/logs/transfer-w-%A_%a.err
set -euo pipefail
cd /home/tungnx/cs-asr-steer
source slurm/a6_env.sh
a6_print_environment 1
case "${SLURM_ARRAY_TASK_ID}" in
  0) DATASET=seame_dev_man; SIDE=encoder; MODE=greedy;;
  1) DATASET=seame_dev_man; SIDE=decoder; MODE=greedy;;
  2) DATASET=seame_dev_sge; SIDE=encoder; MODE=greedy;;
  3) DATASET=seame_dev_sge; SIDE=decoder; MODE=greedy;;
  4) DATASET=seame_dev_man; SIDE=encoder; MODE=official_standard;;
  5) DATASET=seame_dev_man; SIDE=decoder; MODE=official_standard;;
  6) DATASET=seame_dev_sge; SIDE=encoder; MODE=official_standard;;
  7) DATASET=seame_dev_sge; SIDE=decoder; MODE=official_standard;;
esac
echo "experiment_phase=transfer model=whisper dataset=${DATASET} side=${SIDE} mode=${MODE} shard=${SLURM_ARRAY_TASK_ID}"
exec "${BASIS_A6_PYTHON}" -m csasr.experiments.a6_ott_phase_bc --model whisper --phase C --dataset "${DATASET}" --side "${SIDE}" --mode "${MODE}"
