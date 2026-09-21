#!/usr/bin/env bash
#SBATCH --job-name=basis-a6-atlas
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:55:00
#SBATCH --array=0-2783%2
#SBATCH --output=results/basis_a6_expanded/slurm/atlas-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/a6_env.sh
a6_print_environment 1
ROW_JSON="$(${BASIS_A6_PYTHON} - "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
rows=json.load(open("results/basis_a6_expanded/manifests/FULL_RUN_SHARDS.json"))["rows"]
print(json.dumps(rows[int(sys.argv[1])]))
PY
)"
MODEL="$(${BASIS_A6_PYTHON} -c 'import json,sys; print(json.loads(sys.argv[1])["model"])' "$ROW_JSON")"
STAGE="$(${BASIS_A6_PYTHON} -c 'import json,sys; print(json.loads(sys.argv[1])["stage"])' "$ROW_JSON")"
DATASET="$(${BASIS_A6_PYTHON} -c 'import json,sys; print(json.loads(sys.argv[1])["dataset"])' "$ROW_JSON")"
MODE="$(${BASIS_A6_PYTHON} -c 'import json,sys; print(json.loads(sys.argv[1])["mode"])' "$ROW_JSON")"
SIDE="$(${BASIS_A6_PYTHON} -c 'import json,sys; print(json.loads(sys.argv[1])["side"])' "$ROW_JSON")"
LAYER="$(${BASIS_A6_PYTHON} -c 'import json,sys; print(json.loads(sys.argv[1])["layer"])' "$ROW_JSON")"
ARGS=(--stage "$STAGE" --model "$MODEL" --dataset "$DATASET" --decode-mode "$MODE" --side "$SIDE" --layer "$LAYER")
if [[ "$STAGE" == "fixed" ]]; then
  SOURCE="$(${BASIS_A6_PYTHON} -c 'import json,sys; print(json.loads(sys.argv[1])["source"])' "$ROW_JSON")"
  ARGS+=(--source "$SOURCE")
fi
echo "basis_a6_shard ${ROW_JSON}"
exec "${BASIS_A6_PYTHON}" scripts/basis_a6_execute.py "${ARGS[@]}"
