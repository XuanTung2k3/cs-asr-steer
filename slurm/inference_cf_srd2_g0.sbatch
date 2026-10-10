#!/bin/bash
#SBATCH --job-name=cf-srd2g0
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
# SRD2-G0: at most TWO sequential scientific allocations, one phase each, hard 3 h, no chaining/restart/subset.
#   PHASE=capture  Job A (requires pushed PASS_TO_SRD2_G0 for the committed manifest; no D2/steering/reference).
#   PHASE=pulses   Job B (requires the pushed gate seal, SRD2_G0_AUDIT_A: PASS, permutation seal, resource forecast
#                  and the pushed minimal authorization_B with authorized=true).
# The runner re-verifies every guard (sources, config, manifest, audits, remote seals) before loading the model.
# Submit with:
#   sbatch --output=results/inference_cf/srd2_g0/run1/slurm-A-%j.out --error=results/inference_cf/srd2_g0/run1/slurm-A-%j.err \
#          --export=ALL,PHASE=capture slurm/inference_cf_srd2_g0.sbatch
#   sbatch --output=results/inference_cf/srd2_g0/run1/slurm-B-%j.out --error=results/inference_cf/srd2_g0/run1/slurm-B-%j.err \
#          --export=ALL,PHASE=pulses slurm/inference_cf_srd2_g0.sbatch
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${PHASE:?PHASE required}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
R=results/inference_cf/srd2_g0/run1
grep -q '"verdict":"PASS_TO_SRD2_G0"' "$R/audit_PRE.json"
case "$PHASE" in
  capture)
    test ! -e "$R/gate"
    "$PY" experiments/inference_cf_srd2_g0.py capture --out "$R" ;;
  pulses)
    test ! -e "$R/pulses"
    grep -q '"verdict":"SRD2_G0_AUDIT_A: PASS"' "$R/audit_A.json"
    grep -q '"authorized":true' "$R/authorization_B.json"
    "$PY" experiments/inference_cf_srd2_g0.py pulses --out "$R" ;;
  *) echo "unknown PHASE $PHASE"; exit 2 ;;
esac
