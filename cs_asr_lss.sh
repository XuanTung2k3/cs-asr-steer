#!/usr/bin/env bash
# =============================================================================
# Localize-Select-Steer pipeline launcher.
#
# Automatic path (the primary one -- no human annotation anywhere in it):
#
#   sbatch cs_asr_lss.sh chain            # l0 -> l1a -> l1b prepare + evaluate
#   sbatch cs_asr_lss.sh l0
#   sbatch cs_asr_lss.sh l1a
#   sbatch cs_asr_lss.sh l1b              # --mode automatic is the default
#   sbatch cs_asr_lss.sh l1b --prepare-audit      # evidence only
#   sbatch cs_asr_lss.sh l1b --evaluate-gate      # decide Gate A
#
# Optional manual path (only when a human boundary audit is explicitly wanted):
#
#   sbatch cs_asr_lss.sh l1b --prepare-manual-audit
#   #   humans fill artifacts_lss/audit/l1b/verdicts_raw/*.csv
#   sbatch cs_asr_lss.sh l1b --evaluate-gate --mode manual
#
# EXIT CODES (see csasr.lss.gates.STATUS_EXIT_CODES). A stage's exit code says
# whether the *command* worked, never what it found:
#
#   0   ran to completion and wrote a terminal status. The scientific outcome
#       -- passed, completed_no_go, blocked, completed -- is in the status file.
#   2   failed: an implementation defect, a corrupt artifact, or a malformed
#       result. This is the only nonzero code a stage produces on purpose.
#   64  usage error in this script.
#
# `chain` therefore decides whether to continue by reading each stage's status,
# not its exit code, and a Gate A that concludes `completed_no_go` leaves the
# Slurm allocation reported as successful -- because it was.
#
# Model downloads happen only in the login-node `stage_models` phase; every
# other phase runs fully offline.
# =============================================================================
#SBATCH --job-name=cs_asr_lss
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# The chain holds L0 (pilot + seal), L1a (a 30-minute-deadlined probe) and
# L1b's aligner sweep, which the L0 pilot projects at 6-10 GPU-hours. 8 h was a
# schedule bomb; the chain is also resumable, so an allocation that ends mid-way
# is continued by resubmitting it.
#SBATCH --time=20:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/mnt/data/tungnx/cs-asr-steer/logs/cs_asr_lss_%j.out
#SBATCH --error=/mnt/data/tungnx/cs-asr-steer/logs/cs_asr_lss_%j.err

set -euo pipefail

PHASE="${1:-status}"
shift || true

case "${PHASE}" in
  preflight|tests|stage_models|status|summary) ;;
  l0|l1a|l1b|chain) ;;
  *)
    echo "usage: sbatch cs_asr_lss.sh {preflight|tests|stage_models|chain|l0|l1a|l1b|status|summary} [stage args]" >&2
    echo "  chain = l0 -> l1a -> l1b (prepare + evaluate Gate A automatically)" >&2
    echo "  optional manual audit: l1b --prepare-manual-audit, then" >&2
    echo "                         l1b --evaluate-gate --mode manual" >&2
    echo "  (l1c and later stages are not implemented yet)" >&2
    exit 64
    ;;
esac

REPO_DIR="/home/tungnx/cs-asr-steer"
CONDA_ENV="/home/tungnx/miniconda3/envs/acl1"
DATA_ROOT="/mnt/data/tungnx/cs-asr-steer"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/artifacts_lss}"
LOG_DIR="${DATA_ROOT}/logs"
STOP_FILE="${OUTPUT_ROOT}/REQUEST_STOP"
TMP_ROOT="${DATA_ROOT}/tmp"

mkdir -p "${OUTPUT_ROOT}/status" "${LOG_DIR}" "${TMP_ROOT}/mpl"
rm -f "${STOP_FILE}"
trap 'echo "USR1 received: requesting a clean checkpoint"; touch "${STOP_FILE}"' USR1

cd "${REPO_DIR}"

export PATH="${CONDA_ENV}/bin:${PATH}"
# the conda toolchain's libstdc++ must win over the system one, or scipy's
# compiled extensions fail with a CXXABI version error
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
# csasr.utils.seed sets PYTHONHASHSEED after the interpreter has started, which
# has no effect; setting it here is what actually makes hashing deterministic
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CSASR_AUDIO_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# /tmp is not shared with the compute nodes, and l1b renders synthetic audio
# and a few hundred matplotlib figures
export TMPDIR="${TMP_ROOT}"
export MPLCONFIGDIR="${TMP_ROOT}/mpl"

export HF_HOME="${DATA_ROOT}/model_cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${DATA_ROOT}/model_cache/huggingface/hub"
export TORCH_HOME="${DATA_ROOT}/model_cache/torch"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TORCH_HOME}"

if [[ "${PHASE}" != "stage_models" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

PY="${CONDA_ENV}/bin/python"

# one active production job at a time
if command -v squeue >/dev/null 2>&1; then
  ACTIVE="$(squeue -h -u "${USER}" -n cs_asr_lss -o '%i' | awk -v self="${SLURM_JOB_ID:-}" '$1 != self {print $1}')"
  if [[ -n "${ACTIVE}" ]]; then
    echo "another cs_asr_lss job is active or queued: ${ACTIVE}" >&2
    exit 2
  fi
fi

banner() { echo; echo "=== $* ==="; echo; }

write_environment_lock() {
  local lock="${OUTPUT_ROOT}/environment.lock.txt"
  {
    echo "# generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by cs_asr_lss.sh preflight"
    echo "# slurm_job_id=${SLURM_JOB_ID:-none} partition=${SLURM_JOB_PARTITION:-none}"
    echo "# cpus=${SLURM_CPUS_PER_TASK:-unknown} mem=${SLURM_MEM_PER_NODE:-unknown}"
    echo "## python"
    "${PY}" -V
    echo "## pip freeze"
    "${PY}" -m pip freeze
    echo "## nvidia-smi"
    (nvidia-smi || echo "nvidia-smi unavailable") 2>&1
    echo "## torch"
    "${PY}" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "cudnn", torch.backends.cudnn.version(), "available", torch.cuda.is_available())
EOF
  } > "${lock}"
  # the lock is hashed into every stage's provenance via utils.provenance
  cp -f "${lock}" "${REPO_DIR}/environment.lock.txt"
  echo "wrote ${lock}"
}

preflight() {
  banner "preflight"
  "${PY}" -m csasr.experiments.lss_preflight --config configs/lss/base.yaml
  write_environment_lock
}

run_tests() {
  banner "unit tests"
  "${PY}" -m pytest -q
}

# The status a stage last wrote. This, not the exit code, is what decides
# whether the next stage may run: exit codes report command health.
stage_status() {
  "${PY}" -m csasr.experiments.lss_status --config configs/lss/base.yaml \
    --stage-status "$1" 2>/dev/null || echo "unknown"
}

# Run one stage and report both facets: did the command work, and what did it
# find. A nonzero code here means something is broken, not that the answer is no.
run_stage() {
  local name="$1" stage="$2" module="$3" config="$4"; shift 4
  local rc=0
  banner "${name}"
  "${PY}" -m "${module}" --config "${config}" "$@" || rc=$?
  local status; status="$(stage_status "${stage}")"
  echo "[${name}] status=${status} exit=${rc}"
  if [[ ${rc} -ne 0 ]]; then
    echo "[${name}] the command FAILED (exit ${rc}); this is a defect to fix, "
    echo "           not a scientific result. See the stage log above."
  fi
  return "${rc}"
}

# l0 -> l1a -> l1b, fully automatic: no step of this chain waits for a person.
# It is safe because prerequisites are enforced in code -- a stage that did not
# pass leaves the next one unrunnable -- and because the chain itself checks
# each status before continuing, so it stops with a clear message instead of
# running into a PrerequisiteError.
chain() {
  # Flags that belong to every stage, forwarded explicitly. `chain "$@"` used to
  # pass them to L0 only, so `--force-prereq` silently stopped applying halfway
  # through -- and `--overwrite` recomputed L0 while L1a/L1b reused stale bytes.
  local -a common=()
  local -a l0_extra=()
  while (( $# )); do
    case "$1" in
      --force-prereq|--overwrite|--resume|--dry-run) common+=("$1"); shift ;;
      --set) common+=("$1" "$2"); shift 2 ;;
      --pilot-utterances|--limit|--seed) l0_extra+=("$1" "$2"); shift 2 ;;
      *) echo "chain: unsupported option '$1'." >&2
         echo "       Run the stage directly if you need stage-specific flags." >&2
         exit 64 ;;
    esac
  done
  if (( ${#common[@]} )); then
    echo "chain: forwarding ${common[*]} to every stage"
  fi

  preflight
  local rc=0

  # Each stage is skipped if it already passed, so a chain that ran out of wall
  # clock is resumed by resubmitting it rather than restarted from L0.
  if [[ "$(stage_status l0_freeze)" == "passed" ]]; then
    echo "L0 already passed; skipping (pass --overwrite to redo it)."
  else
    run_stage "L0 freeze" l0_freeze \
      csasr.experiments.lss_l0_freeze configs/lss/l0_freeze.yaml \
      "${common[@]+"${common[@]}"}" "${l0_extra[@]+"${l0_extra[@]}"}" || rc=$?
  fi
  if [[ "$(stage_status l0_freeze)" != "passed" ]]; then
    echo "L0 is $(stage_status l0_freeze), not passed; l1a and l1b will not run."
    exit "${rc}"
  fi

  if [[ "$(stage_status l1a_diag)" == "passed" ]]; then
    echo "L1a already passed; skipping."
  else
    run_stage "L1a diagnostics" l1a_diag \
      csasr.experiments.lss_l1a_diag configs/lss/l1a_diag.yaml \
      "${common[@]+"${common[@]}"}" || rc=$?
  fi
  if [[ "$(stage_status l1a_diag)" != "passed" ]]; then
    echo "L1a is $(stage_status l1a_diag), not passed; l1b will not run."
    exit "${rc}"
  fi

  rc=0
  # --align runs the aligner families over every role a coverage threshold is
  # stated about. Without it consensus falls back to the 20-utterance
  # exploratory table, which Gate A refuses as non-production evidence.
  # This is the expensive step (6-10 GPU-hours); if the allocation ends here,
  # resubmitting `chain` resumes from it because L0 and L1a are already passed.
  run_stage "L1b Gate A (automatic)" l1b_valid \
    csasr.experiments.lss_l1b_valid configs/lss/l1b_valid.yaml --align \
    "${common[@]+"${common[@]}"}" || rc=$?

  banner "pipeline status"
  "${PY}" -m csasr.experiments.lss_status --config configs/lss/base.yaml || true

  local gate; gate="$(stage_status l1b_valid)"
  echo
  echo "Gate A: ${gate}"
  case "${gate}" in
    passed)
      echo "Spans are frozen at ${OUTPUT_ROOT}/freeze/l1b_spans_freeze.json; l1c may run." ;;
    blocked)
      cat <<EOF
Gate A could not conclude: evidence it requires does not exist. The reason
codes and the next action are in
  ${OUTPUT_ROOT}/status/l1b_valid.json      (.blocked_reasons, .next_action)
  ${OUTPUT_ROOT}/reports/l1b_gate_a.md

If the block is blocked_missing_synthetic_calibration, the automatic path has
no source of absolute boundary error yet. The optional manual audit is the
other way to obtain one:

  sbatch cs_asr_lss.sh l1b --prepare-manual-audit
  #   read  ${OUTPUT_ROOT}/audit/l1b/ANNOTATION_GUIDE.md
  #   score ${OUTPUT_ROOT}/audit/l1b/items.jsonl
  #   save each annotator's file as
  #         ${OUTPUT_ROOT}/audit/l1b/verdicts_raw/<annotator-id>.csv
  sbatch cs_asr_lss.sh l1b --evaluate-gate --mode manual
EOF
      ;;
    completed_no_go)
      echo "The experiment ran and a pre-registered threshold was not met."
      echo "Take the response for the failing group in ${OUTPUT_ROOT}/reports/l1b_gate_a.md." ;;
    *)
      echo "See ${OUTPUT_ROOT}/reports/l1b_gate_a.md." ;;
  esac
  exit "${rc}"
}

case "${PHASE}" in
  chain)        chain "$@" ;;
  preflight)    preflight ;;
  tests)        run_tests ;;
  stage_models) "${PY}" -m csasr.experiments.ensure_lss_models --config configs/lss/l1a_diag.yaml "$@" ;;
  status)       "${PY}" -m csasr.experiments.lss_status --config configs/lss/base.yaml "$@" ;;
  summary)      "${PY}" -m csasr.experiments.lss_status --config configs/lss/base.yaml --report "$@" ;;
  l0)           preflight; "${PY}" -m csasr.experiments.lss_l0_freeze --config configs/lss/l0_freeze.yaml "$@" ;;
  l1a)          preflight; "${PY}" -m csasr.experiments.lss_l1a_diag  --config configs/lss/l1a_diag.yaml  "$@" ;;
  l1b)          preflight; "${PY}" -m csasr.experiments.lss_l1b_valid --config configs/lss/l1b_valid.yaml "$@" ;;
esac

banner "done: ${PHASE}"
