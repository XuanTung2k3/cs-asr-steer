#!/bin/bash
#SBATCH --job-name=cf-r0
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
# R0: ONE allocation over the frozen FULL300 in canonical order: fixed-grid native LID, forced-ZH all-layer replay (+repeat),
# forced-EN comparator replay, reference-free per-utterance constructions and controls. Row 0 is the in-allocation
# integrity/cost smoke (projection >3h minus margin => COMPUTE_STOP). No oracle access. Hard 3 h; no restart.
# Submit with:
#   sbatch --output=<OUT>/slurm-%j.out --error=<OUT>/slurm-%j.err --export=ALL,OUT=<OUT> slurm/inference_cf_r0.sbatch
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${OUT:?OUT required}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
test -f "$OUT/manifest.json"
grep -q '"verdict": "PASS_TO_R0"' results/inference_cf/r0/prerun_audit.json
exec "$PY" experiments/inference_cf_r0.py run --out "$OUT"
