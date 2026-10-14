#!/bin/bash
#SBATCH --job-name=reft-p21-report
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/tungnx/slurm_logs/reft-p21-report-%j.out
#SBATCH --error=/mnt/data/tungnx/slurm_logs/reft-p21-report-%j.err
# P2-1 full report and terminal verdict (CPU; rules: docs/reft/P2_1_DECISION_RULES.md)
set -euo pipefail
cd /home/tungnx/cs-asr-reft
export TMPDIR=/mnt/data/tungnx/tmp/reft-$SLURM_JOB_ID; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/tungnx/miniconda3/envs/acl1/bin/python -m pytest -q -p no:cacheprovider tests/reft/test_p2_1_report.py
/home/tungnx/miniconda3/envs/acl1/bin/python -u reft/report_p2_1.py
