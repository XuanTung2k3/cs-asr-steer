#!/bin/bash
#SBATCH --job-name=abts0-k1
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
# ABTS-0 K1 (candidate feasibility only; NO steering, training or K2). NOT AUTHORIZED until a human approves.
# Two sequential allocations, one phase each, hard 1.5 h, no restart for outcome reasons:
#   PHASE=capture  matched-engine B0 / EN (+ cached alignment-head attention) and AUTO for FULL300
#   PHASE=score    hard-zero target / off-target rescoring of the SEALED chosen pairs only
#                  (requires the committed + pushed results/inference_cf/abts0_k1/run1/candidates_sealed.json)
# Between them (CPU): python experiments/abts0_k1.py select --workers 16 ; commit + push the candidate seal.
# Submit with:
#   sbatch --output=results/inference_cf/abts0_k1/run1/slurm-%j.out --error=results/inference_cf/abts0_k1/run1/slurm-%j.err \
#          --export=ALL,PHASE=<capture|score> slurm/abts0_k1.sbatch
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${PHASE:?PHASE required}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
test -f results/inference_cf/abts0_k1/plan_sealed.json
case "$PHASE" in
  capture)
    "$PY" experiments/abts0_k1.py capture ;;
  score)
    test -f results/inference_cf/abts0_k1/run1/candidates_sealed.json
    git ls-files --error-unmatch results/inference_cf/abts0_k1/run1/candidates_sealed.json results/inference_cf/abts0_k1/run1/selection.json
    "$PY" experiments/abts0_k1.py score ;;
  *) echo "unknown PHASE $PHASE"; exit 2 ;;
esac
