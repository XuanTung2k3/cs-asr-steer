"""Running the aligner families and caching their candidate tables.

Thin wrappers over `csasr.nat5h.aligners`, which is reused unchanged. The cache
is identity-guarded: a candidate table is reused only when it was produced by
the same code, config and model, exactly as the NAT5H pipeline does.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from ...nat5h.schema import (
    RunIdentity,
    artifact_matches_identity,
    assert_schema_v2,
    empty_candidates,
)
from ...utils.hashing import sha256_obj, sha256_strings
from ...utils.logging import get_logger

log = get_logger(__name__)

SUPPORTED_FAMILIES = ("existing_ctc", "whisper_dtw", "qwen_forced_aligner")

#: Versioned because this digest is persisted in cache manifests. Changing the
#: canonicalization without changing the version would make old and new hashes
#: look comparable when they are not.
REQUEST_MANIFEST_FINGERPRINT_VERSION = "lss-aligner-request-v1"


def request_manifest_fingerprint(manifest: pd.DataFrame) -> str:
    """Fingerprint the exact ordered rows an aligner was asked to process.

    Candidate identity columns only describe the code/configuration that emitted
    a table. They do not describe *which utterances* were requested. In
    particular, two gate generations share a configuration, so accepting a cache
    on ``config_hash`` alone can return generation N's rows for generation N+1.

    All columns, their names, their order-normalized values, and row order enter
    this digest. Sorting columns makes an equivalent DataFrame construction
    stable; preserving row order binds the chunk plan as well as the universe.
    """
    if manifest is None:
        manifest = pd.DataFrame()
    columns = sorted(str(c) for c in manifest.columns)
    canonical = manifest.loc[:, columns].reset_index(drop=True) \
        if columns else pd.DataFrame()
    payload = canonical.to_json(
        orient="split", index=False, date_format="iso", date_unit="ns",
        double_precision=15, force_ascii=False, default_handler=str)
    return sha256_obj({
        "version": REQUEST_MANIFEST_FINGERPRINT_VERSION,
        "columns": columns,
        "payload": payload,
    })


def request_manifest_record(manifest: pd.DataFrame) -> dict[str, Any]:
    """Machine-readable cache key and audit summary for one aligner request."""
    utterances = (manifest["utterance_id"].astype(str).tolist()
                  if manifest is not None and "utterance_id" in manifest else [])
    return {
        "schema_version": REQUEST_MANIFEST_FINGERPRINT_VERSION,
        "sha256": request_manifest_fingerprint(manifest),
        "rows": 0 if manifest is None else int(len(manifest)),
        "unique_utterances": int(len(set(utterances))),
        "utterance_ids_sha256": sha256_strings(utterances),
    }


def _read_matching_cache(path: Path, *, cfg: Mapping[str, Any],
                         identity: RunIdentity,
                         request: Mapping[str, Any]
                         ) -> tuple[pd.DataFrame, dict[str, Any] | None, str]:
    """Authenticate a cache and bind it to the request that produced it."""
    from ...lss import manifest as manifest_mod

    frame, verdict = manifest_mod.read_verified(path, cfg=cfg,
                                                 require_identity=True)
    if not verdict["ok"]:
        return pd.DataFrame(), verdict.get("manifest"), str(verdict["verdict"])
    published = verdict.get("manifest") or {}
    recorded = published.get("request_manifest") or {}
    if not isinstance(recorded, Mapping):
        return pd.DataFrame(), published, "malformed_request_fingerprint"
    if recorded.get("schema_version") != REQUEST_MANIFEST_FINGERPRINT_VERSION:
        return pd.DataFrame(), published, "missing_request_fingerprint"
    if recorded.get("sha256") != request.get("sha256"):
        return pd.DataFrame(), published, "request_manifest_mismatch"
    try:
        row_count_matches = int(recorded.get("rows", -1)) == int(
            request.get("rows", -2))
    except (TypeError, ValueError):
        row_count_matches = False
    if not row_count_matches:
        return pd.DataFrame(), published, "request_row_count_mismatch"
    if not len(frame) or not artifact_matches_identity(frame, identity):
        return pd.DataFrame(), published, "candidate_identity_mismatch"
    return frame, published, "ok"


def _cache_name(family: str, pred_start_offset: int | None) -> str:
    """Cache filename for one family's candidates.

    Two Whisper decoder-query conventions resolve to the same `cfg`, so
    `artifact_matches_identity` cannot tell their tables apart -- a sweep would
    reuse the first offset's cache for the second and score one convention twice.
    The convention is therefore part of the filename.
    """
    from ...nat5h.aligners import DEFAULT_PRED_START_OFFSET

    if family != "whisper_dtw" or pred_start_offset is None \
            or int(pred_start_offset) == DEFAULT_PRED_START_OFFSET:
        return f"candidates_{family}.parquet"
    return f"candidates_{family}_pred{int(pred_start_offset)}.parquet"


def load_or_run(path: str | Path, builder: Callable[[], pd.DataFrame],
                identity: RunIdentity, *, cfg: Mapping[str, Any],
                request_manifest: pd.DataFrame, overwrite: bool = False,
                stage: str = "l1b_valid", run_dir: str | Path | None = None,
                parents: Sequence[Mapping[str, Any]] = (),
                taint_reasons: Sequence[str] = (), purpose: str = "natural"
                ) -> tuple[pd.DataFrame, dict[str, Any], str]:
    """Reuse only an authenticated cache for this exact aligner request."""
    from ...lss import manifest as manifest_mod

    path = Path(path)
    request = request_manifest_record(request_manifest)
    if path.is_file() and not overwrite:
        cached, published, verdict = _read_matching_cache(
            path, cfg=cfg, identity=identity, request=request)
        if verdict == "ok":
            log.info("reusing cached candidates: %s (%d rows)", path, len(cached))
            return cached, dict(published or {}), "cached_authenticated_request"
        log.info("cached candidates at %s are not reusable (%s); rebuilding",
                 path, verdict)
    table = builder()
    published = manifest_mod.publish_frame(
        path, table, stage=stage, cfg=cfg, run_dir=run_dir,
        parents=[p for p in parents if p], taint_reasons=list(taint_reasons),
        key_columns=("utterance_id", "reference_unit_index", "aligner_family",
                     "aligner_variant"), schema="nat5h_candidates_v2",
        extra={"request_manifest": request, "candidate_cache": {
            "purpose": str(purpose), "family": (
                str(table["aligner_family"].iloc[0]) if len(table)
                and "aligner_family" in table else "unknown")}})
    return table, published, "rebuilt"


def run_families(manifest: pd.DataFrame, cfg: dict, geometry, identity: RunIdentity,
                 families: Sequence[str], *, bundle=None,
                 out_dir: Path | None = None,
                 overwrite: bool = False,
                 qwen_runner: Callable[..., Any] | None = None,
                 pred_start_offset: int | None = None,
                 stage: str = "l1b_valid", run_dir: str | Path | None = None,
                 parents: Sequence[Mapping[str, Any]] = (),
                 taint_reasons: Sequence[str] = (), purpose: str = "natural"
                 ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run each configured aligner family; a family that fails is recorded.

    ``qwen_runner`` is the production Qwen path
    (`csasr.lss.align.qwen_prod.production_runner`). Without it the family is
    still never run inline -- a hang here would take the whole sweep with it, and
    only a process-group deadline can stop a checkpoint stuck in a native call --
    so this function falls back to reusing an identity-matching table if one is
    already on disk, and otherwise records the family as deferred.
    """
    from ...nat5h.aligners import run_existing_ctc, run_whisper_dtw

    frames: list[pd.DataFrame] = []
    request = request_manifest_record(manifest)
    report: dict[str, Any] = {"families": {}, "active": [],
                              "request_manifest": request,
                              "cache_manifests": {}}

    for family in families:
        if family not in SUPPORTED_FAMILIES:
            report["families"][family] = {"state": "unsupported"}
            continue
        if family == "qwen_forced_aligner":
            cached = (Path(out_dir) / f"candidates_{family}.parquet"
                      if out_dir is not None else None)
            if cached is not None and cached.is_file() and not overwrite:
                table, published, verdict = _read_matching_cache(
                    cached, cfg=cfg, identity=identity, request=request)
                if verdict == "ok":
                    frames.append(table)
                    report["families"][family] = {
                        "state": "ok", "rows": int(len(table)),
                        "valid_rows": int(table["is_valid"].sum())
                        if "is_valid" in table else None,
                        "source": "cached_authenticated_request"}
                    report["cache_manifests"][family] = published
                    report["active"].append(family)
                    continue
                log.info("cached Qwen candidates at %s are not reusable (%s); "
                         "the production runner decides what happens next",
                         cached, verdict)
            if qwen_runner is None:
                report["families"][family] = {"state": "deferred_to_probe"}
                continue
            outcome = qwen_runner(manifest, out_dir=out_dir)
            record = outcome.to_dict() if hasattr(outcome, "to_dict") else dict(outcome)
            if record.get("state") == "ok" and cached is not None and cached.is_file():
                table, published, verdict = _read_matching_cache(
                    cached, cfg=cfg, identity=identity, request=request)
                if verdict == "ok":
                    frames.append(table)
                    report["families"][family] = {
                        "state": "ok", "rows": int(len(table)),
                        "valid_rows": int(table["is_valid"].sum())
                        if "is_valid" in table else None,
                        "source": "production_runner", "production": record}
                    report["cache_manifests"][family] = published
                    report["active"].append(family)
                else:
                    report["families"][family] = {
                        "state": "failed",
                        "error": ("production runner published a cache that does "
                                  f"not match its request: {verdict}"),
                        "production": record}
            else:
                report["families"][family] = {
                    "state": str(record.get("state", "unavailable")),
                    "error": str(record.get("reason") or "production run did not "
                                 "publish a table"),
                    "production": record}
            continue

        def builder(family=family):
            if family == "existing_ctc":
                return run_existing_ctc(manifest, cfg, geometry, identity)
            if bundle is None:
                raise RuntimeError("whisper_dtw requires a loaded model bundle")
            # the decoder-query convention is passed, not injected into cfg:
            # mutating cfg would change `resolved_config_hash` and every artifact
            # identity derived from it, so a frozen selection would look like a
            # different configuration to the manifest checks
            table, meta = run_whisper_dtw(bundle, manifest, cfg, identity,
                                          pred_start_offset=pred_start_offset)
            report["families"].setdefault(family, {})["mapping"] = meta
            return table

        try:
            if out_dir is not None:
                table, published, source = load_or_run(
                    Path(out_dir) / _cache_name(family, pred_start_offset),
                    builder, identity, cfg=cfg, request_manifest=manifest,
                    overwrite=overwrite, stage=stage, run_dir=run_dir,
                    parents=parents, taint_reasons=taint_reasons,
                    purpose=purpose)
                report["cache_manifests"][family] = published
            else:
                table = builder()
                source = "uncached"
            frames.append(table)
            report["families"].setdefault(family, {}).update({
                "state": "ok", "rows": int(len(table)),
                "valid_rows": int(table["is_valid"].sum()) if "is_valid" in table else None,
                "source": source,
            })
            report["active"].append(family)
        except Exception as exc:
            log.warning("aligner family %s unavailable: %s", family, exc)
            report["families"].setdefault(family, {}).update({
                "state": "unavailable", "error": repr(exc)})

    if not frames:
        return empty_candidates(), report
    out = pd.concat(frames, ignore_index=True)
    assert_schema_v2(out, artifact="lss candidate table")
    return out, report


# The decoder-query convention is now selected, not just swept.
# `csasr.lss.align.devselect.whisper_variant_sweep` runs each configured offset
# over the synthetic development set through `run_whisper_dtw` -- the same
# function production alignment uses -- and scores it against known boundaries.
# `run_whisper_dtw` threads `pred_start_offset` and puts it in the variant label,
# defaulting to the historical -1 so every artifact recorded before the sweep
# existed keeps its meaning. L1b passes the frozen selection through
# `run_families(pred_start_offset=...)` rather than editing `cfg`, because
# mutating `cfg` would change every artifact identity derived from it.
