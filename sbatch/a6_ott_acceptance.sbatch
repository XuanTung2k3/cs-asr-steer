#!/usr/bin/env bash
# Physical A6-OTT compact acceptance.  Submit one Whisper and one Qwen job;
# these two jobs are the only concurrent GPU work allowed by the protocol.
#SBATCH --job-name=a6ott-accept
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=results/a6_ott_upper_bound/logs/accept-%x-%j.out
#SBATCH --error=results/a6_ott_upper_bound/logs/accept-%x-%j.err

set -euo pipefail
cd /home/tungnx/cs-asr-steer
source slurm/a6_env.sh
a6_print_environment 1
echo "experiment_phase=physical_acceptance"
echo "model=${A6_OTT_MODEL:?A6_OTT_MODEL must be set}"
exec "${BASIS_A6_PYTHON}" -m csasr.experiments.a6_ott \
  --model "${A6_OTT_MODEL}" --acceptance --acceptance-count "${A6_OTT_ACCEPTANCE_COUNT:-3}"
