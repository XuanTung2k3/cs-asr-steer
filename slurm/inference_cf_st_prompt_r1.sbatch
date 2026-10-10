#!/bin/bash
#SBATCH --job-name=cf-r1a
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
# ST-PROMPT-R1-A: ONE allocation. capture (passive paired 4-layer E/M states + L16 bitwise identity + CPU geometry ledger)
# -> SEPARATE-process geometry coverage gate (stop for human dose revision if any arm < 54/60 or < 12 dialogues per stratum)
# -> pulses (historical barrier on all 180, then 24 primary + 12 random relative-dose single pulses). Hard 3 h; no restart.
# R1-B is NOT run by this script (requires A CAUSAL_PROMISE + pushed selection + audits + its own pre-run PASS).
# Submit with:
#   sbatch --output=<OUT>/slurm-%j.out --error=<OUT>/slurm-%j.err --export=ALL,OUT=<OUT> slurm/inference_cf_st_prompt_r1.sbatch
set -euo pipefail
cd /home/tungnx/cs-asr-steer-inf
: "${OUT:?OUT required}"
export PYTHONPATH="$PWD/src:$PWD"
export LD_LIBRARY_PATH="/home/tungnx/miniconda3/envs/acl1/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/tungnx/miniconda3/envs/acl1/bin/python
test -f "$OUT/manifest.json"
grep -q '"verdict": "PASS_TO_ST_PROMPT_R1_A"' results/inference_cf/st_prompt_r1/prerun_audit_A.json
"$PY" experiments/inference_cf_st_prompt_r1.py capture --out "$OUT"
"$PY" experiments/inference_cf_st_prompt_r1_geometry.py --out "$OUT"
"$PY" experiments/inference_cf_st_prompt_r1.py pulses --out "$OUT"
