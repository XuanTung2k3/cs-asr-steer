#!/bin/bash
#SBATCH --job-name=cf-srccf0p
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
# SRC-CF0-P: at most TWO sequential scientific allocations, one phase each, hard 3 h, no restart, no parallelism across gates.
#   PHASE=construct  P-A reference-free direction construction (requires PASS_TO_SRC_CF0_PILOT committed + pushed and the
#                    committed manifest_construct.json).
#   PHASE=pulse      P-B causal pulse screen (requires the pushed direction seal, SRC_CF0_PILOT_AUDIT: PASS (CONSTRUCTION), the
#                    pushed passing minimal PB_authorization.json and the committed manifest_pulse.json).
# Same MIG GPU type as the historical S1 / ST-LOC0 / ST-PROMPT-R1 runs (bitwise historical reproduction).
# Submit with:
#   sbatch --output=results/inference_cf/src_cf0_pilot/run1/slurm-%j.out --error=results/inference_cf/src_cf0_pilot/run1/slurm-%j.err \
#          --export=ALL,PHASE=<phase> slurm/inference_cf_src_cf0_pilot.sbatch
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${PHASE:?PHASE required}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
R=results/inference_cf/src_cf0_pilot/run1
grep -q '"verdict": "PASS_TO_SRC_CF0_PILOT"' "$R/prerun_audit.json"
case "$PHASE" in
  construct)
    test -f "$R/manifest_construct.json"
    "$PY" experiments/inference_cf_src_cf0_pilot.py construct ;;
  pulse)
    test -f "$R/manifest_pulse.json"
    grep -q '"verdict": "SRC_CF0_PILOT_AUDIT: PASS (CONSTRUCTION)"' "$R/construction_audit.json"
    grep -q '"gate_pass": true' "$R/PB_authorization.json"
    "$PY" experiments/inference_cf_src_cf0_pilot.py pulse ;;
  *) echo "unknown PHASE $PHASE"; exit 2 ;;
esac
