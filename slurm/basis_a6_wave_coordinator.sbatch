#!/usr/bin/env bash
# CPU-only scheduler coordinator. Every scientific worker it submits is an
# sbatch GPU array with %2 concurrency; this job never requests a GPU.
#SBATCH --job-name=basis-a6-waves
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=7-00:00:00
#SBATCH --output=results/basis_a6_expanded/slurm/waves-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
total=$(python - <<'PY'
import json
print(len(json.load(open("results/basis_a6_expanded/manifests/FULL_RUN_SHARDS.json"))["rows"]))
PY
)
previous=""
for offset in $(seq 0 10 $((total - 1))); do
  last=$((total - offset - 1)); last=$((last < 9 ? last : 9))
  dep=()
  if [[ -n "$previous" ]]; then dep=(--dependency="afterok:${previous}"); fi
  jid=$(sbatch --parsable --array="0-${last}%2" "${dep[@]}" --export="ALL,BASIS_A6_SHARD_OFFSET=${offset}" slurm/basis_a6_atlas_array.sbatch)
  echo "submitted wave offset=${offset} size=$((last+1)) job=${jid}" | tee -a "results/basis_a6_expanded/slurm/WAVE_SUBMISSIONS.log"
  while squeue -h -j "$jid" | grep -q .; do sleep 30; done
  states=$(sacct -X -n -o State -j "$jid" | tr -d ' ' | sort -u | tr '\n' ' ')
  echo "completed wave job=${jid} states=${states}" | tee -a "results/basis_a6_expanded/slurm/WAVE_SUBMISSIONS.log"
  if echo "$states" | grep -Eqv '^(COMPLETED|)$'; then
    echo "wave ${jid} did not complete cleanly" >&2
    exit 1
  fi
  previous="$jid"
done
echo "all ${total} shards submitted and completed" | tee -a "results/basis_a6_expanded/slurm/WAVE_SUBMISSIONS.log"
