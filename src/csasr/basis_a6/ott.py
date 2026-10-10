"""Compact A6-OTT protocol, reuse validation, and leakage firewall.

This module is deliberately separate from :mod:`basis_a6.protocol`, which
continues to describe the superseded 70,080-cell atlas.  A6-OTT has only the
three oracle per-sample families and never imports a fixed-direction result.
The model callbacks and exact-site hooks remain in the existing A6 modules;
this module supplies the phase and resumability contract around them.
"""
from __future__ import annotations

import hashlib
import json
import ast
import math
import platform
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..utils.hashing import sha256_obj
from ..utils.provenance import code_config_snapshot_hash, resolved_config_hash, source_snapshot_hash
from .decoding import decode_config


RESULT_NAMESPACE = "results/a6_ott_upper_bound"
LEGACY_NAMESPACE = "results/basis_a6_expanded"
REUSE_MANIFEST = f"{RESULT_NAMESPACE}/migration/EXISTING_RESULT_REUSE_MANIFEST.json"
SEARCH_PHASE = "A"
CONFIRM_PHASE = "B"
TRANSFER_PHASE = "C"

COMPACT_FAMILIES = (
    ("add_unique", "encoder"),
    ("add_unique", "decoder"),
    ("conditioning_cs", "decoder"),
)
SEARCH_UTTERANCES_PER_DATASET = 20
SEARCH_DATASETS = ("cs_dialogue", "ascend")
WHISPER_RHOS = (0.5, 1.0, 2.0)
QWEN_RHOS = (0.5, 1.0, 2.0, 4.0)
SEARCH_DECODE_MODE = "greedy"

MODEL_SPECS: dict[str, dict[str, int]] = {
    "whisper": {"encoder_layers": 32, "decoder_layers": 32, "dimension": 1280},
    "qwen3_asr_1p7b": {"encoder_layers": 24, "decoder_layers": 28, "dimension": 2048},
}

_FORBIDDEN_PHASE_A_TERMS = frozenset({
    "confirm", "d-dev-confirm", "d_test", "d-test", "phase_b", "phase-c",
    "phase_c", "transfer", "seame", "test",
})
_OUTCOME_KEYS = frozenset({
    "MER", "PIER", "utility", "poi_net_utility", "corrections", "corruptions",
    "matrix_retention", "embedded_retention", "outside_harm", "metrics", "outcome",
})


class PhaseFirewallError(RuntimeError):
    """Raised before a phase loader can read an out-of-phase artifact."""


class ReuseValidationError(ValueError):
    """Raised for a malformed or scientifically incompatible reuse entry."""


def compact_sites(model: str) -> tuple[tuple[str, str, int], ...]:
    """Return only ``(family, side, layer)`` sites allowed by A6-OTT."""
    if model not in MODEL_SPECS:
        raise ValueError(f"unknown A6-OTT model: {model}")
    spec = MODEL_SPECS[model]
    rows: list[tuple[str, str, int]] = []
    for family, side in COMPACT_FAMILIES:
        n = spec[f"{side}_layers"]
        rows.extend((family, side, layer) for layer in range(n))
    return tuple(rows)


def rho_grid(model: str) -> tuple[float, ...]:
    if model == "whisper":
        return WHISPER_RHOS
    if model == "qwen3_asr_1p7b":
        return QWEN_RHOS
    raise ValueError(f"unknown A6-OTT model: {model}")


def phase_a_logical_total(model: str) -> int:
    return len(compact_sites(model)) * len(rho_grid(model)) * 2 * SEARCH_UTTERANCES_PER_DATASET


def enumerate_phase_a(model: str, *, utterance_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Enumerate Search rows without adding any legacy method or fixed source."""
    ids = list(utterance_ids) if utterance_ids is not None else [
        f"{dataset}_search_{i:02d}"
        for dataset in SEARCH_DATASETS for i in range(SEARCH_UTTERANCES_PER_DATASET)
    ]
    if len(ids) != 2 * SEARCH_UTTERANCES_PER_DATASET:
        raise ValueError("A6-OTT Phase-A requires 20 CS and 20 ASCEND IDs")
    rows = []
    for family, side, layer in compact_sites(model):
        for rho in rho_grid(model):
            for utterance_id in ids:
                dataset = "cs_dialogue" if utterance_id.startswith("cs_") else "ascend"
                rows.append({"phase": SEARCH_PHASE, "model": model, "dataset": dataset,
                             "utterance_id": utterance_id, "family": family, "side": side,
                             "layer": layer, "rho": rho, "decode_mode": SEARCH_DECODE_MODE})
    return rows


def assert_compact_row(row: Mapping[str, Any]) -> None:
    """Reject the superseded families and corpus-fixed branches at the boundary."""
    family = str(row.get("family", row.get("method", "")))
    side = str(row.get("side", ""))
    if (family, side) not in COMPACT_FAMILIES:
        raise PhaseFirewallError(f"A6-OTT row is outside compact families: {family}/{side}")
    if row.get("branch") in {"fixed", "a6_f"} or row.get("construction_source"):
        raise PhaseFirewallError("A6-OTT cannot schedule corpus-fixed steering")
    if family in {"raw", "minus_shared", "unique_minus_shared", "conditioning_all", "u-s"}:
        raise PhaseFirewallError(f"legacy A6 method is forbidden in A6-OTT: {family}")


def _safe_relative(path: Path, repo_root: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ReuseValidationError(f"artifact escapes repository: {path}") from exc


def _hash(path: Path) -> str:
    # The migration manifest uses the conventional byte SHA-256 (the legacy
    # artifact helper also has a size-mixed variant for snapshot manifests).
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _validation_hash(payload: Mapping[str, Any]) -> str:
    return "sha256:" + sha256_obj(dict(payload))


def _source_run_id(payload: Mapping[str, Any], *, fallback: str) -> str:
    provenance = payload.get("provenance") or {}
    return str(provenance.get("run_id") or (
        f"{provenance.get('scope', 'unknown')}@{provenance.get('git_commit', 'unknown')}")) or fallback


def _validate_baseline(entry: Mapping[str, Any], repo_root: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    rel = Path(str(entry.get("artifact_path", "")))
    path = repo_root / rel
    if not path.is_file():
        return {}, [f"missing artifact: {rel}"]
    actual = _hash(path)
    if actual != entry.get("artifact_sha256"):
        errors.append(f"hash mismatch: expected {entry.get('artifact_sha256')}, got {actual}")
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return {}, errors + [f"invalid JSON: {exc}"]
    parts = str(entry.get("logical_cell", "")).split("|")
    if len(parts) != 4 or parts[0] != "baseline":
        errors.append("baseline logical_cell is malformed")
    else:
        _, model, dataset, mode = parts
        for key, expected in (("model", model), ("dataset", dataset), ("decode_mode", mode)):
            if payload.get(key) != expected:
                errors.append(f"{key} mismatch: {payload.get(key)!r} != {expected!r}")
    if payload.get("schema_version") != "basis_a6_baseline_v1" or payload.get("status") != "PASS":
        errors.append("baseline is not a passing basis-a6 baseline artifact")
    provenance = payload.get("provenance") or {}
    for key in ("git_commit", "model_revision", "panel_fingerprint", "scope", "site"):
        if not provenance.get(key):
            errors.append(f"missing baseline provenance.{key}")
    if not isinstance(payload.get("rows"), dict) or not payload["rows"]:
        errors.append("baseline has no decoded rows")
    if payload.get("official_equivalent_to") and payload["official_equivalent_to"] != "greedy":
        errors.append("unexpected official-standard equivalence target")
    return payload, errors


def _validate_partial_cache(entry: Mapping[str, Any], repo_root: Path) -> tuple[dict[str, Any], list[str]]:
    """Validate direction-only reuse without reading aggregate outcome rows."""
    def read_npy(path: Path) -> tuple[str, tuple[int, ...], list[float]]:
        """Read the small 1-D numeric vectors without making preflight depend on NumPy."""
        with path.open("rb") as handle:
            if handle.read(6) != b"\x93NUMPY":
                raise ValueError("bad npy magic")
            major, minor = struct.unpack("<BB", handle.read(2))
            length_size = 2 if major == 1 else 4
            header_len = int.from_bytes(handle.read(length_size), "little")
            header = ast.literal_eval(handle.read(header_len).decode("latin1").strip())
            dtype = str(header["descr"])
            shape = tuple(int(x) for x in header["shape"])
            if header.get("fortran_order") or len(shape) != 1:
                raise ValueError("direction vector is not a C-order 1-D array")
            count = math.prod(shape)
            if dtype == "<f4":
                fmt, size = "<f", 4
            elif dtype == "<f8":
                fmt, size = "<d", 8
            else:
                raise ValueError(f"unsupported direction dtype {dtype}")
            data = handle.read(count * size)
            if len(data) != count * size:
                raise ValueError("truncated direction vector")
            return dtype, shape, [float(x[0]) for x in struct.iter_unpack(fmt, data)]
    errors: list[str] = []
    cache_path = repo_root / Path(str(entry.get("artifact_path", "")))
    if not cache_path.is_file():
        return {}, [f"missing direction cache: {cache_path}"]
    actual = _hash(cache_path)
    if actual != entry.get("artifact_sha256"):
        errors.append(f"direction-cache hash mismatch: expected {entry.get('artifact_sha256')}, got {actual}")
    try:
        payload = json.loads(cache_path.read_text())
    except Exception as exc:
        return {}, errors + [f"invalid direction cache JSON: {exc}"]
    if payload.get("schema_version") != "basis_a6_tt_direction_cache_v1":
        errors.append("unexpected direction-cache schema")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        errors.append("direction cache has no records")
        return payload, errors
    for rec in records:
        if rec.get("model") != "whisper" or rec.get("dataset") != "cs_dialogue_dev_select":
            errors.append("direction cache identity is not Whisper CS dev-select")
            break
        if rec.get("decode_analysis_mode") != "greedy" or rec.get("side") != "encoder":
            errors.append("direction cache analysis identity is not greedy encoder")
            break
        vector_path = cache_path.parent / str(rec.get("vector_file", ""))
        if not vector_path.is_file():
            errors.append(f"missing direction vector: {vector_path}")
            break
        try:
            _, _, vector = read_npy(vector_path)
            norm = math.sqrt(sum(value * value for value in vector))
            if not all(math.isfinite(value) for value in vector) or not math.isclose(norm, 1.0, abs_tol=2e-5):
                errors.append(f"invalid non-unit direction: {vector_path}")
                break
            if vector_path.stem != str(rec.get("direction_hash", "")).removeprefix("sha256:"):
                errors.append(f"direction filename/hash mismatch: {vector_path}")
                break
        except Exception as exc:
            errors.append(f"cannot read direction vector {vector_path}: {exc}")
            break
    run_manifest = entry.get("run_manifest")
    if run_manifest:
        run_path = repo_root / Path(str(run_manifest))
        if not run_path.is_file():
            errors.append(f"missing source run manifest: {run_manifest}")
        else:
            run = json.loads(run_path.read_text())
            if run.get("status") != "PASS":
                errors.append("source run manifest is not PASS")
            if Path(str(run.get("direction_cache", ""))).name != cache_path.name:
                errors.append("source run manifest does not point to the claimed cache")
            if int(run.get("direction_constructions", 0)) <= 0:
                errors.append("source run manifest has no direction constructions")
    else:
        errors.append("partial cache has no source run manifest")
    return payload, errors


def validate_reuse_manifest(manifest_path: str | Path, *, repo_root: str | Path = ".") -> dict[str, Any]:
    """Validate every manifest claim and emit a non-authoritative reuse ledger.

    ``REUSE_EXACT`` is accepted only for a hash- and provenance-valid baseline.
    A partial cache is accepted only as direction-construction reuse, never as
    an exact evaluation row.  Failed claims are downgraded in the returned
    decision without mutating the historical migration manifest.
    """
    root = Path(repo_root).resolve()
    source = Path(manifest_path)
    manifest = json.loads(source.read_text())
    decisions: list[dict[str, Any]] = []
    exact = 0
    validated_partial = 0
    downgraded = 0
    for entry in manifest.get("entries", []):
        classification = str(entry.get("classification", ""))
        if entry.get("artifact_path"):
            _safe_relative(root / Path(str(entry["artifact_path"])), root)
        payload: dict[str, Any] = {}
        errors: list[str] = []
        if classification == "REUSE_EXACT":
            payload, errors = _validate_baseline(entry, root)
        elif classification == "PARTIAL_REQUIRES_VALIDATION":
            payload, errors = _validate_partial_cache(entry, root)
        else:
            decisions.append({"logical_cell": entry.get("logical_cell"), "classification": classification,
                              "accepted": False, "reason": entry.get("reason", "not reusable")})
            continue
        accepted = not errors
        if accepted and classification == "REUSE_EXACT":
            exact += 1
        elif accepted:
            validated_partial += 1
        else:
            downgraded += 1
        artifact = str(entry.get("artifact_path", ""))
        source_hash = str(entry.get("artifact_sha256", ""))
        validation_payload = {"logical_cell": entry.get("logical_cell"), "classification": classification,
                              "accepted": accepted, "source_hash": source_hash, "errors": errors}
        decisions.append({
            "logical_cell": entry.get("logical_cell"),
            "classification": classification if accepted else "NEED_RUN",
            "accepted": accepted,
            "reused": bool(accepted),
            "source_artifact": artifact,
            "source_job_id": str(entry.get("original_job_id", "unknown")),
            "source_run_id": _source_run_id(payload, fallback=artifact),
            "source_hash": source_hash,
            "validation_hash": _validation_hash(validation_payload),
            "validation_errors": errors,
            "reuse_scope": "exact_logical_row" if classification == "REUSE_EXACT" else "direction_construction_only",
        })
    return {"schema_version": "a6_ott_reuse_validation_v1", "manifest": str(source),
            "manifest_sha256": _hash(source), "exact_reused": exact,
            "validated_partial_construction": validated_partial, "downgraded": downgraded,
            "decisions": decisions}


def _path_parts(path: Path, root: Path) -> set[str]:
    try:
        return {part.casefold() for part in path.resolve().relative_to(root.resolve()).parts}
    except ValueError as exc:
        raise PhaseFirewallError(f"phase artifact escapes result namespace: {path}") from exc


def assert_phase_artifact_allowed(path: str | Path, *, phase: str,
                                  result_root: str | Path) -> None:
    """Check path metadata before reading its contents."""
    root = Path(result_root).resolve()
    candidate = Path(path)
    parts = _path_parts(candidate, root)
    if phase == SEARCH_PHASE and parts & _FORBIDDEN_PHASE_A_TERMS:
        raise PhaseFirewallError(f"Phase-A selector cannot access confirm/transfer/test artifact: {path}")
    if phase not in {SEARCH_PHASE, CONFIRM_PHASE, TRANSFER_PHASE}:
        raise PhaseFirewallError(f"unknown A6-OTT phase: {phase}")


def assert_phase_a_selection_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    """Reject confirm/transfer rows before any outcome field is inspected."""
    for row in rows:
        phase = str(row.get("phase", SEARCH_PHASE))
        role = str(row.get("data_role", row.get("role", "search"))).casefold()
        if phase != SEARCH_PHASE or role not in {"search", "phase_a", "a"}:
            raise PhaseFirewallError("Phase-A selector received confirm/transfer data")
        if _OUTCOME_KEYS.intersection(row):
            raise PhaseFirewallError("Phase-A selector received outcome-bearing row")
        assert_compact_row(row)


def load_selection_rows(path: str | Path, *, result_root: str | Path) -> list[dict[str, Any]]:
    """Load only an outcome-free Phase-A selection manifest."""
    assert_phase_artifact_allowed(path, phase=SEARCH_PHASE, result_root=result_root)
    payload = json.loads(Path(path).read_text())
    if payload.get("phase") != SEARCH_PHASE:
        raise PhaseFirewallError("selection manifest is not Phase A")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise PhaseFirewallError("selection manifest has no rows")
    assert_phase_a_selection_rows(rows)
    return [dict(row) for row in rows]


def load_accepted_cell(path: str | Path, *, canonical_key: str,
                       phase: str, result_root: str | Path) -> dict[str, Any] | None:
    """Return a validated completed cell, or ``None`` so a caller may run it.

    This is the resumability gate used immediately before a physical decode.
    It intentionally validates the row identity and provenance before the
    caller can invoke a model callback.
    """
    assert_phase_artifact_allowed(path, phase=phase, result_root=result_root)
    candidate = Path(path)
    if not candidate.is_file():
        return None
    try:
        payload = json.loads(candidate.read_text())
    except Exception:
        return None
    if payload.get("status") != "PASS" or payload.get("canonical_key") != canonical_key:
        return None
    if not isinstance(payload.get("provenance"), Mapping):
        return None
    return {"payload": payload, "source_artifact": str(candidate),
            "source_hash": _hash(candidate), "reused": True,
            "canonical_key": canonical_key}


def freeze_search_ids(records: Iterable[Mapping[str, Any]], *, seed: int = 42,
                     target: int = SEARCH_UTTERANCES_PER_DATASET) -> dict[str, Any]:
    """Deterministically freeze IDs from metadata without touching outcomes."""
    grouped: dict[str, list[dict[str, Any]]] = {"cs_dialogue": [], "ascend": []}
    for record in records:
        dataset = str(record.get("dataset", ""))
        if dataset not in grouped or not record.get("utterance_id"):
            raise ValueError("search freeze requires dataset and utterance_id metadata")
        if _OUTCOME_KEYS.intersection(record):
            raise PhaseFirewallError("outcome-bearing metadata cannot enter Search freeze")
        grouped[dataset].append({"utterance_id": str(record["utterance_id"]),
                                 "stratum": str(record.get("stratum", record.get("dialogue_id", "")))})
    selected: dict[str, list[str]] = {}
    for dataset, values in grouped.items():
        if len(values) < target:
            raise ValueError(f"not enough {dataset} metadata rows for Search freeze")
        ordered = sorted(values, key=lambda row: hashlib.sha256(
            f"{seed}\0{dataset}\0{row['utterance_id']}".encode()).hexdigest())
        ids = [row["utterance_id"] for row in ordered[:target]]
        selected[dataset] = sorted(ids)
    payload = {"schema_version": "a6_ott_search_freeze_v1", "phase": SEARCH_PHASE,
               "seed": seed, "target_per_dataset": target, "selection_rule": "sha256(seed,dataset,utterance_id)",
               "ids": selected}
    payload["freeze_hash"] = "sha256:" + sha256_obj(payload)
    return payload


def validate_qwen_standard_equivalence(root: str | Path) -> dict[str, Any]:
    """Validate Qwen's official-standard/greedy alias from configs and rows."""
    if decode_config("qwen3_asr_1p7b", "official_standard").get("provenance_equivalent_to") != "greedy":
        return {"status": "FAIL", "reason": "decode config does not declare greedy equivalence"}
    checks = []
    base = Path(root)
    if (base / "results").is_dir():
        base = base / "results"
    for dataset in ("cs_dialogue_dev_select", "ascend_eval", "seame_dev_man", "seame_dev_sge"):
        greedy = base / "basis_a6_expanded" / "baselines" / "qwen3_asr_1p7b" / dataset / "greedy.json"
        standard = greedy.with_name("official_standard.json")
        if not greedy.is_file() or not standard.is_file():
            checks.append({"dataset": dataset, "ok": False, "reason": "missing baseline"})
            continue
        g, s = json.loads(greedy.read_text()), json.loads(standard.read_text())
        same = g.get("rows") == s.get("rows") and s.get("official_equivalent_to") == "greedy"
        checks.append({"dataset": dataset, "ok": same, "greedy_hash": _hash(greedy), "standard_hash": _hash(standard)})
    return {"status": "PASS" if checks and all(x["ok"] for x in checks) else "FAIL", "checks": checks,
            "equivalence": "official_standard == greedy"}


def runtime_estimate(root: str | Path, *, reuse: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Estimate only remaining compact work, with direction/rho reuse explicit."""
    root = Path(root)
    measured = json.loads((root / "results/basis_a6_expanded/preflight/RUNTIME_ESTIMATES.json").read_text())
    seconds = {
        model: float(spec["seconds_per_utterance"]) * float(spec["measured_tt_overhead_factor"])
        for model, spec in measured["models"].items()
    }
    # Baseline artifacts are reusable inputs, but they are not steered Search
    # rows.  Keep them in the reuse ledger while leaving Phase-A search work
    # at zero exact row reuse unless a future accepted A6-OTT row is present.
    exact = {"whisper": 0, "qwen3_asr_1p7b": 0}
    for row in (reuse or {}).get("decisions", []):
        if row.get("accepted") and row.get("reuse_scope") == "exact_logical_row":
            cell = str(row.get("logical_cell", ""))
            for model in exact:
                if cell.startswith("oracle_tt|") and f"|{model}|" in cell:
                    exact[model] += 1
    phase_a = {}
    for model in MODEL_SPECS:
        total = phase_a_logical_total(model)
        remaining = max(0, total - exact[model])
        phase_a[model] = {"logical_total": total, "exact_reused": exact[model],
                          "remaining_physical": remaining,
                          "direction_constructions": len(compact_sites(model)) * 40,
                          "seconds_per_greedy_eval": seconds[model],
                          "gpu_hours": remaining * seconds[model] / 3600.0}
    # Phase-B/C are bounded maxima from the frozen protocol.  Whisper's
    # official-standard decode is measured conservatively at 3x greedy;
    # Qwen's official-standard is the validated greedy alias.
    whisper_a = phase_a["whisper"]["gpu_hours"]
    qwen_a = phase_a["qwen3_asr_1p7b"]["gpu_hours"]
    whisper_b = 5760 * seconds["whisper"] * 2 / 3600
    whisper_c = 600 * seconds["whisper"] * 2 / 3600
    qwen_b = 5760 * seconds["qwen3_asr_1p7b"] / 3600
    qwen_c = 600 * seconds["qwen3_asr_1p7b"] / 3600
    whisper_total, qwen_total = whisper_a + whisper_b + whisper_c, qwen_a + qwen_b + qwen_c
    return {"phase_a": phase_a, "ancillary_exact_reused": int(sum(
        1 for row in (reuse or {}).get("decisions", [])
        if row.get("accepted") and row.get("reuse_scope") == "exact_logical_row")),
        "pipeline_max": {
        "whisper_gpu_hours": whisper_total, "qwen_gpu_hours": qwen_total,
        "total_gpu_hours": whisper_total + qwen_total,
        "slack_adjusted_gpu_hours": (whisper_total + qwen_total) * 1.15,
        "expected_wall_hours_at_2_gpus": (whisper_total + qwen_total) * 1.15 / 2,
    }, "reuse_note": "directions are one construction per sample/layer; rho reuses the direction; Qwen standard aliases greedy"}


def preflight_provenance(*, root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Return run identity required on every new A6-OTT manifest."""
    root = Path(root)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).splitlines()
    except Exception:
        git_commit, git_status = "unknown", []
    return {"stage": "a6_ott_preflight", "config_hash": resolved_config_hash(dict(config)),
            "source_snapshot_hash": source_snapshot_hash(root),
            "code_config_snapshot_hash": code_config_snapshot_hash(root),
            "git_commit": git_commit, "git_status": git_status,
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                             "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
                             "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES")}}


__all__ = ["COMPACT_FAMILIES", "PhaseFirewallError", "assert_compact_row", "assert_phase_a_selection_rows",
           "compact_sites", "enumerate_phase_a", "freeze_search_ids", "load_selection_rows",
           "load_accepted_cell", "phase_a_logical_total", "preflight_provenance", "rho_grid", "runtime_estimate",
           "validate_qwen_standard_equivalence", "validate_reuse_manifest"]
