#!/bin/bash
# =============================================================================
# Guide-gated P0..E5 pipeline. Safe checkpoint-aware mode:
#   sbatch cs_asr_e1_e5.sh auto
#
# `auto` now runs only through the next required inspection boundary:
# preflight -> tests -> optional smoke -> P0 -> E1. It does not automatically
# launch E2-E5 after decisive reports become available. Submit phase modes
# explicitly: e1, e2_pilot, e3_pilot, e4_pilot, e2_full, e3_full, e4_refine,
# e4_confirm, e5, summary.
# =============================================================================
#SBATCH --job-name=cs_asr_e1_e5
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/data/tungnx/cs-asr-steer/logs/%x_%j.out
#SBATCH --error=/mnt/data/tungnx/cs-asr-steer/logs/%x_%j.err

set -euo pipefail

# --------------------------------------------------------------------------
# paths (resolved for this machine)
# --------------------------------------------------------------------------
REPO_DIR="/home/tungnx/cs-asr-steer"
CONDA_ENV="/home/tungnx/miniconda3/envs/acl1"
DATA_ROOT="/mnt/data/tungnx/cs-asr-steer"          # everything generated lives here
MODEL_DIR="/mnt/data/tungnx/whisper-large-v3"
DATASET_DIR="/mnt/data/tungnx/CS-Dialogue"

OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/artifacts_v2}"
SMOKE_OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-${DATA_ROOT}/artifacts_smoke}"
ARTIFACTS="${OUTPUT_ROOT}"
LOG_DIR="${DATA_ROOT}/logs"
mkdir -p "${ARTIFACTS}" "${LOG_DIR}" "${DATA_ROOT}/cache" "${DATA_ROOT}/tmp"

# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
export PATH="${CONDA_ENV}/bin:${PATH}"
# the conda toolchain's libstdc++ must win over the older system one, otherwise
# scipy's compiled extensions fail with a CXXABI version error
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${DATA_ROOT}/cache/hf"
export HF_HUB_OFFLINE=1                 # model and data are local; never fetch
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${DATA_ROOT}/cache/mpl"
export TMPDIR="${DATA_ROOT}/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${HF_HOME}" "${MPLCONFIGDIR}"

PY="${CONDA_ENV}/bin/python"
cd "${REPO_DIR}"

cleanup_owned_locks() {
  local lock holder
  shopt -s nullglob
  for lock in "${OUTPUT_ROOT}/status/"*.lock; do
    holder=$(<"${lock}")
    if [[ "${holder}" == *"job=${SLURM_JOB_ID:-none}"* ]]; then rm -f "${lock}"; fi
  done
  shopt -u nullglob
}
cleanup_stale_locks() {
  local lock holder_job
  shopt -s nullglob
  for lock in "${OUTPUT_ROOT}/status/"*.lock; do
    holder_job=$(sed -n "s/.* job=\([^ ]*\).*/\1/p" "${lock}")
    if [[ -z "${holder_job}" || "${holder_job}" == "None" ]] || ! squeue -h -j "${holder_job}" | grep -q .; then
      echo "removing stale stage lock: ${lock}"
      rm -f "${lock}"
    fi
  done
  shopt -u nullglob
}
cleanup_stale_locks
trap cleanup_owned_locks EXIT
trap "cleanup_owned_locks; exit 143" TERM INT

# --------------------------------------------------------------------------
# options
# --------------------------------------------------------------------------
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"
OVERWRITE="${OVERWRITE:-0}"
BUILD_MANIFEST="${BUILD_MANIFEST:-0}"

COMMON_ARGS=(--seed "${SEED}" --output-dir "${OUTPUT_ROOT}")
if [[ "${OVERWRITE}" == "1" ]]; then COMMON_ARGS+=(--overwrite); else COMMON_ARGS+=(--resume); fi
if [[ -n "${LIMIT}" ]]; then COMMON_ARGS+=(--limit "${LIMIT}"); fi

banner() {
  echo
  echo "============================================================"
  echo "  $*"
  echo "  $(date -Is)  host=$(hostname)  job=${SLURM_JOB_ID:-none}"
  echo "============================================================"
}

preflight() {
  banner "preflight"
  [[ -d "${MODEL_DIR}" ]]   || { echo "FATAL: model not found: ${MODEL_DIR}"; exit 1; }
  [[ -d "${DATASET_DIR}" ]] || { echo "FATAL: dataset not found: ${DATASET_DIR}"; exit 1; }
  [[ -d "${DATASET_DIR}/data/short_wav/extracted/short_wav/WAVE" ]] || {
    echo "FATAL: CS-Dialogue audio is not extracted; E1-E5 cannot proceed."
    echo "       Extract data/short_wav/short_wav.tar.gz* before running."
    exit 1; }
  "${PY}" - <<'PYCHECK'
import torch, transformers, sys
print(f"python      {sys.version.split()[0]}")
print(f"torch       {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"transformers{transformers.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu         {p.name}  {p.total_memory/1e9:.1f} GB")
else:
    print("WARNING: no CUDA device visible; stages will fall back to CPU float32")
PYCHECK
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>/dev/null || true
}

run_tests() {
  banner "unit tests"
  "${PY}" -m pytest -q
}

# run_stage <name> <module> <config> [extra args...]
run_stage() {
  local name="$1"; shift
  local module="$1"; shift
  local config="$1"; shift
  banner "stage ${name}"
  local rc=0
  "${PY}" -m "${module}" --config "${config}" "${COMMON_ARGS[@]}" "$@" || rc=$?
  if [[ ${rc} -eq 0 ]]; then
    echo "[${name}] GATE PASSED"
  elif [[ ${rc} -eq 3 ]]; then
    echo "[${name}] BLOCKED: alignment finished; complete audit/e1/verdicts.csv."
    "${PY}" -m csasr.experiments.pipeline --config configs/base.yaml --set experiment.output_root="${OUTPUT_ROOT}" --status || true
    exit 0
  elif [[ ${rc} -eq 2 ]]; then
    echo "[${name}] GATE FAILED - stopping here as the guide requires."
    echo "         Evidence: ${ARTIFACTS}/status/  and  ${ARTIFACTS}/reports/"
    "${PY}" -m csasr.experiments.pipeline --config configs/base.yaml --set experiment.output_root="${OUTPUT_ROOT}" --status || true
    exit 2
  else
    echo "[${name}] ERROR (exit ${rc})"
    exit "${rc}"
  fi
}

stage_p0() {
  local extra=()
  [[ "${BUILD_MANIFEST}" == "1" ]] && extra+=(--build-manifest)
  run_stage p0 csasr.experiments.p0_baseline configs/experiments/e1_alignment.yaml "${extra[@]}"
}
stage_e1() { run_stage e1 csasr.experiments.e1_alignment  configs/experiments/e1_alignment.yaml; }
stage_e2_pilot() { run_stage e2_pilot csasr.experiments.e2_directions configs/experiments/e2_directions.yaml; }
stage_e2_full() { run_stage e2_full csasr.experiments.e2_directions configs/experiments/e2_directions.yaml --set directions.subset=train_direction_full; }
stage_e3_pilot() { run_stage e3_pilot csasr.experiments.e3_separability configs/experiments/e3_separability.yaml --phase pilot; }
stage_e3_full() { run_stage e3_full csasr.experiments.e3_separability configs/experiments/e3_separability.yaml --phase full; }
stage_e4_pilot()   { run_stage e4_pilot   csasr.experiments.e4_oracle configs/experiments/e4_oracle.yaml --phase pilot; }
stage_e4_refine()  { run_stage e4_refine  csasr.experiments.e4_oracle configs/experiments/e4_oracle.yaml --phase refine; }
stage_e4_confirm() { run_stage e4_confirm csasr.experiments.e4_oracle configs/experiments/e4_oracle.yaml --phase confirm; }
stage_e5() { run_stage e5 csasr.experiments.e5_boundaries configs/experiments/e5_boundaries.yaml; }

summary() {
  banner "summary"
  "${PY}" -m csasr.experiments.pipeline --config configs/base.yaml --set experiment.output_root="${OUTPUT_ROOT}" --status || true
  echo
  echo "artifacts : ${ARTIFACTS}"
  echo "reports   : ${ARTIFACTS}/reports"
  echo "status    : ${ARTIFACTS}/status/overview.json"
  ls -1 "${ARTIFACTS}/reports" 2>/dev/null || true
}

# --------------------------------------------------------------------------
# smoke: execute every stage once on a handful of utterances
#
# Runs entirely inside a subshell against SMOKE_OUTPUT_ROOT so nothing here can
# touch production artifacts. Gates are expected to fail at this sample size --
# the question is only whether every code path reaches its gate without a
# traceback. Individual stage failures are non-fatal; the caller decides.
# --------------------------------------------------------------------------
run_smoke() {
  banner "smoke run (LIMIT=${LIMIT:-8})"
  (
    set +e
    local LIMIT="${LIMIT:-8}"
    local OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT}"
    local ARTIFACTS="${SMOKE_OUTPUT_ROOT}"
    local SMOKE="--output-dir ${SMOKE_OUTPUT_ROOT} --seed ${SEED} --resume --force-prereq"
    mkdir -p "${SMOKE_OUTPUT_ROOT}"
    cleanup_stale_locks

    "${PY}" -m csasr.experiments.p0_baseline --config configs/experiments/e1_alignment.yaml \
        --output-dir "${SMOKE_OUTPUT_ROOT}" --seed "${SEED}" --resume \
        --build-manifest --limit "${LIMIT}"
    "${PY}" -m csasr.experiments.e1_alignment --config configs/experiments/e1_alignment.yaml \
        ${SMOKE} --limit "${LIMIT}"
    "${PY}" -m csasr.experiments.e2_directions --config configs/experiments/e2_directions.yaml \
        ${SMOKE} --limit "${LIMIT}" \
        --set directions.decoder_max_utterances=4 --set directions.subset=train_direction_full
    "${PY}" -m csasr.experiments.e3_separability --config configs/experiments/e3_separability.yaml \
        ${SMOKE} --limit "${LIMIT}" --set e3.bootstrap_resamples=50

    "${PY}" -m csasr.experiments.e4_oracle --config configs/experiments/e4_oracle.yaml \
        ${SMOKE} --limit 2 --phase pilot \
        --set e4.encoder_alphas_pilot='[1.0]' --set e4.decoder_alphas='[0.1]' \
        --set e4.random_seeds='[142]' --set e4.bootstrap_resamples=50

    # At LIMIT=2 no configuration can be feasible, so the pilot writes no frozen
    # config and refine/confirm/E5 would exit early -- leaving their decode loops
    # untested, which is the whole point of the smoke. Plant a synthetic config so
    # those paths execute. Confined to the smoke root; never read by production.
    local frozen="${SMOKE_OUTPUT_ROOT}/metrics/e4_selected_config.json"
    if [[ ! -f "${frozen}" ]]; then
      echo "[smoke] planting SYNTHETIC ${frozen} to exercise refine/confirm/E5"
      "${PY}" - "${SMOKE_OUTPUT_ROOT}" <<'PYSYN'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
sel = root / "reports" / "e3_selected_layers.json"
layer = 31
if sel.exists():
    picked = json.loads(sel.read_text(encoding="utf-8")).get("selected_layers") or []
    if picked:
        layer = int(picked[0])
out = root / "metrics" / "e4_selected_config.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "layer": layer, "alpha": 1.0, "mask": "EXACT_TAPER", "phase": "smoke",
    "synthetic": True,
    "note": "plumbing placeholder written by run_smoke; not a scientific result",
}, indent=2), encoding="utf-8")
print(f"synthetic frozen config: layer={layer} alpha=1.0")
PYSYN
    fi

    for phase in refine confirm; do
      "${PY}" -m csasr.experiments.e4_oracle --config configs/experiments/e4_oracle.yaml \
          ${SMOKE} --limit 2 --phase "${phase}" \
          --set e4.encoder_alphas_refine='[0.5, 1.0]' --set e4.decoder_alphas='[0.1]' \
          --set e4.random_seeds='[142]' --set e4.bootstrap_resamples=50
    done

    "${PY}" -m csasr.experiments.e5_boundaries --config configs/experiments/e5_boundaries.yaml \
        ${SMOKE} --limit 2 \
        --set e5.masks='[EXACT_HARD, EXACT_TAPER, EXPAND_100, JITTER_100, BOUNDARY_ONLY, WHOLE_WORD_PLUS_CONTEXT]' \
        --set e5.jitter_seeds='[242]' --set e5.plot_examples=2 \
        --set e5.bootstrap_resamples=50

    rm -f "${SMOKE_OUTPUT_ROOT}/status/"*.lock 2>/dev/null

    # The smoke exists to surface defects before production burns hours on them.
    # Job 31413 proved that is worthless if the findings stay buried in stderr:
    # the smoke hit the same E1 assertion that later killed production, and the
    # run continued regardless. Distinct exceptions are now reported loudly.
    banner "smoke findings"
    local errfile="${SLURM_ERROR_FILE:-${LOG_DIR}/${SLURM_JOB_NAME:-cs_asr_e1_e5}_${SLURM_JOB_ID:-none}.err}"
    if [[ -r "${errfile}" ]]; then
      local found
      found=$(grep -oE "^[A-Za-z_.]*(Error|Exception)[^:]*:.*" "${errfile}" 2>/dev/null \
              | sed 's/artifacts_smoke/<SMOKE_ROOT>/g' | sort | uniq -c | sort -rn)
      if [[ -n "${found}" ]]; then
        echo "!! the smoke raised these distinct exceptions:"
        echo "${found}" | sed 's/^/   /'
        echo
        echo "   Missing-prerequisite errors are expected when an earlier smoke stage"
        echo "   fails. Anything else is a real defect that will also hit production."
      else
        echo "no exceptions raised during the smoke"
      fi
    else
      echo "stderr log not readable at ${errfile}; inspect it manually"
    fi
  )
  return 0
}

# --------------------------------------------------------------------------
# checkpoint-aware combined workflow
# --------------------------------------------------------------------------
WORKFLOW_VERSION="guide_v3_20260728"
WORKFLOW_DIR="${OUTPUT_ROOT}/workflow"

status_passed() {
  local key="$1"
  case "${key}" in
    p0|e1|e2_pilot|e2_full|e3_pilot|e3_full|e4_pilot|e4_refine|e4_confirm|e5)
      "${PY}" - "${OUTPUT_ROOT}" "${key}" <<'PYSTATUS'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
stage = sys.argv[2]
path = root / "status" / f"{stage}.json"
if not path.exists():
    raise SystemExit(f"missing status file: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
gate = payload.get("gate") or {}
prov = payload.get("provenance") or {}
ok = (
    payload.get("status") == "passed"
    and payload.get("complete", True)
    and gate.get("passed", True)
    and prov.get("production_artifact", True) is not False
)
if not ok:
    raise SystemExit(f"{stage} is not a complete passed production status")
PYSTATUS
      ;;
    *) return 0 ;;
  esac
}

auto_step() {
  local key="$1"; shift
  local marker="${WORKFLOW_DIR}/${WORKFLOW_VERSION}_${key}.done"
  if [[ -f "${marker}" && "${OVERWRITE}" != "1" ]]; then
    echo "[auto] SKIP ${key}: completion marker exists"
    return 0
  fi
  "$@"
  status_passed "${key}"
  mkdir -p "${WORKFLOW_DIR}"
  touch "${marker}"
  echo "[auto] MARKED ${key} complete"
}

auto_pipeline() {
  preflight
  auto_step tests run_tests
  # Plumbing check on ~8 utterances before anything expensive starts. Costs
  # ~30-45 min of this allocation and is far cheaper than discovering a defect
  # in E4 after P0/E1/E2 have already run. Marker-guarded: it runs once.
  # Set SKIP_SMOKE=1 to bypass.
  if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then auto_step smoke run_smoke; fi
  auto_step p0 stage_p0
  auto_step e1 stage_e1
  echo
  echo "[auto] stopped after E1. Inspect ${ARTIFACTS}/reports/e1_alignment_audit.md"
  echo "       Continue only after E1 status is passed and audit gates are complete:"
  echo "       sbatch cs_asr_e1_e5.sh e2_pilot"
  summary
}

# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------
MODE=("$@")
if [[ ${#MODE[@]} -gt 1 ]]; then
  echo "Pass one mode only; received: ${MODE[*]}"
  exit 64
fi
if [[ ${#MODE[@]} -eq 0 ]]; then MODE=(prepare); fi

case "${MODE[0]}" in
  auto|all)
    auto_pipeline
    ;;
  tests)
    preflight; run_tests
    ;;
  smoke)
    preflight; run_tests; run_smoke; summary
    ;;
  prepare)
    preflight; run_tests; stage_p0; summary
    ;;
  align)
    preflight; stage_e1; summary
    ;;
  e1_review)
    preflight; stage_e1; summary
    ;;
  pilot|confirm_dirs)
    echo "This combined mode is disabled; submit the single stages listed at the top."
    exit 64
    ;;
  *)
    preflight
    for s in "${MODE[@]}"; do
      case "${s}" in
        p0)         stage_p0 ;;
        e1)         stage_e1 ;;
        e2|e2_pilot) stage_e2_pilot ;;
        e2_full)    stage_e2_full ;;
        e3|e3_pilot) stage_e3_pilot ;;
        e3_full)    stage_e3_full ;;
        e4_pilot)   stage_e4_pilot ;;
        e4_refine)  stage_e4_refine ;;
        e4_confirm) stage_e4_confirm ;;
        e5)         stage_e5 ;;
        summary)    summary ;;
        tests)      run_tests ;;
        *) echo "unknown stage: ${s}"; exit 64 ;;
      esac
    done
    summary
    ;;
esac

banner "done"
