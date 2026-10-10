#!/bin/bash
#SBATCH --job-name=cf-s1
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
# S1 acoustic-evidence feasibility: at most TWO sequential allocations, one phase each, hard 3 h, no restart.
#   PHASE=candidates  original-audio M/E/AUTO candidates + regions (requires PASS_TO_S1 + committed manifest_candidates.json)
#   PHASE=acoustic    masked counterfactual rescoring of the SEALED unions (requires the pushed candidate seal +
#                     S1_AUDIT: PASS (CANDIDATES) + committed manifest_acoustic.json)
# Submit with:
#   sbatch --output=results/inference_cf/s1/run1/slurm-%j.out --error=results/inference_cf/s1/run1/slurm-%j.err \
#          --export=ALL,PHASE=<phase> slurm/acoustic_s1.sbatch
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${PHASE:?PHASE required}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
grep -q '"verdict": "PASS_TO_S1"' results/inference_cf/s1/prerun_audit.json
case "$PHASE" in
  candidates)
    test -f results/inference_cf/s1/run1/manifest_candidates.json
    "$PY" experiments/acoustic_s1.py candidates ;;
  acoustic)
    test -f results/inference_cf/s1/run1/manifest_acoustic.json
    grep -q '"verdict": "S1_AUDIT: PASS (CANDIDATES)"' results/inference_cf/s1/candidates_audit.json
    "$PY" experiments/acoustic_s1.py acoustic ;;
  *) echo "unknown PHASE $PHASE"; exit 2 ;;
esac
