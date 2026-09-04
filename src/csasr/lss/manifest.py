"""Immutable per-artifact manifests: lineage, identity and taint.

A status file says what a *stage* concluded. It cannot say whether the parquet a
later stage is about to read was produced by that run, by a forced diagnostic
three days ago, or by a job that died halfway through writing it. That gap is
what lets an `--overwrite` clear a stage's status taint while leaving tainted
bytes on disk for a downstream gate to consume.

So every artifact that Gate A depends on is published with a sidecar manifest,
`<artifact>.manifest.json`, written atomically *after* the artifact itself:

    {
      "manifest_version": "lss-artifact-manifest-v1",
      "path": "alignments/candidates_all.parquet",
      "sha256": "...",                     # of the artifact, not the manifest
      "rows": 12043,
      "complete": true,                    # false until the write finishes
      "producing_run_id": "l1b_valid/2026-08-11T02:14:41Z",
      "producing_stage": "l1b_valid",
      "created_at": "...",
      "identity": {
        "code_config_sha256": ...,           # gated
        "test_sha256": ...,                  # recorded, not gated
        "config_sha256": ..., "data_sha256": ..., "model_id": ...
      },
      "expected_keys_sha256": "...",       # what the producer promised to write
      "parent_artifacts": [{"path": ..., "sha256": ..., "run_id": ...}],
      "parent_run_ids": [...],
      "taint_reasons": [...],
      "diagnostic_only": false
    }

Three properties follow, and they are the point of the file:

* **Taint is bound to bytes.** `merge_taint` unions the taint of every parent
  manifest into the child's. Reusing a cached artifact re-reads its manifest, so
  a tainted input taints its consumer no matter what any status file says.
* **`--overwrite` cannot launder.** It only clears taint for artifacts this run
  actually rebuilt; anything reused keeps the manifest it was published with.
* **Truncation is detectable.** `complete` is written only after a successful
  atomic publish, and `expected_keys_sha256` lets a reader check that the rows
  present are the rows the producer intended, not a prefix of them.

`verify` returns a structured verdict rather than raising, because a stage must
be able to report "the evidence I was given is not authentic" as a gate outcome
instead of a crash.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from ..nat5h.statusing import atomic_output_path, atomic_write_json
from ..utils.hashing import sha256_file, sha256_obj, sha256_strings
from ..utils.provenance import (
    code_config_snapshot_hash,
    dataset_manifest_hash,
    resolved_config_hash,
    source_snapshot_hash,
    test_snapshot_hash,
)

MANIFEST_VERSION = "lss-artifact-manifest-v1"
MANIFEST_SUFFIX = ".manifest.json"

#: verdicts `verify` can return
OK = "ok"
MISSING_ARTIFACT = "missing_artifact"
MISSING_MANIFEST = "missing_manifest"
SHA_MISMATCH = "sha256_mismatch"
INCOMPLETE = "incomplete"
IDENTITY_MISMATCH = "identity_mismatch"
KEY_MISMATCH = "expected_keys_mismatch"
BAD_MANIFEST = "unreadable_manifest"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def manifest_path(artifact: str | Path) -> Path:
    return Path(str(artifact) + MANIFEST_SUFFIX)


def run_id(stage: str, run_dir: str | Path | None = None) -> str:
    """A stable identifier for one execution of one stage."""
    if run_dir:
        return f"{stage}/{Path(run_dir).name}"
    return f"{stage}/{os.environ.get('SLURM_JOB_ID') or _now()}"


def _common_identity(cfg: Mapping[str, Any]) -> dict[str, Any]:
    model = dict(cfg.get("model") or {})
    return {
        "config_sha256": resolved_config_hash(dict(cfg)),
        "data_sha256": dataset_manifest_hash(dict(cfg)),
        "model_id": model.get("hub_id") or model.get("id"),
        "model_revision": model.get("revision"),
    }


def identity(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """What must match for a cached artifact to be reusable.

    `nat5h.RunIdentity` defaults `code_commit` to "unknown", which makes its
    cache check blind to source changes. Production code/configuration and the
    test suite are both recorded, but only the former is gated. This prevents a
    test-only change from invalidating scientific bytes while preserving the
    full provenance record.
    """
    return {
        "code_config_sha256": code_config_snapshot_hash(),
        "test_sha256": test_snapshot_hash(),
        **_common_identity(cfg),
    }


def _legacy_identity(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Identity expected from an immutable pre-split artifact manifest."""
    return {
        "source_sha256": source_snapshot_hash(),
        **_common_identity(cfg),
    }


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _identity_format(have: Mapping[str, Any]) -> tuple[str | None, str]:
    """Classify the mutually exclusive legacy and split identity schemas."""
    legacy = "source_sha256" in have
    code = "code_config_sha256" in have
    tests = "test_sha256" in have
    if legacy and not code and not tests:
        if _is_sha256(have.get("source_sha256")):
            return "legacy", ""
        return None, "legacy source_sha256 is not a lowercase SHA-256 digest"
    if not legacy and code and tests:
        bad = [name for name in ("code_config_sha256", "test_sha256")
               if not _is_sha256(have.get(name))]
        if not bad:
            return "split", ""
        return None, f"split identity has invalid digest fields {bad}"
    present = sorted(name for name, yes in (
        ("source_sha256", legacy),
        ("code_config_sha256", code),
        ("test_sha256", tests),
    ) if yes)
    return None, ("expected exactly source_sha256 (legacy) or both "
                  "code_config_sha256 and test_sha256 (split); found "
                  f"{present}")


def expected_keys_hash(frame: pd.DataFrame,
                       key_columns: Sequence[str] = ()) -> str | None:
    """Hash of the exact row keys an artifact contains.

    A truncated parquet is still a valid parquet with fewer rows. Recording the
    key set is what turns "nonempty" into "complete".
    """
    if not len(key_columns) or frame is None or not len(frame):
        return None
    present = [c for c in key_columns if c in frame.columns]
    if len(present) != len(key_columns):
        return None
    keys = (frame[list(present)].astype(str)
            .agg("␟".join, axis=1).sort_values().tolist())
    return sha256_strings(keys)


def merge_taint(parents: Iterable[Mapping[str, Any]], *,
                extra_reasons: Iterable[str] = ()) -> dict[str, Any]:
    """Union the taint of every parent manifest with this run's own reasons."""
    reasons: set[str] = {str(r) for r in extra_reasons}
    run_ids: set[str] = set()
    for parent in parents:
        if not parent:
            continue
        reasons.update(str(r) for r in (parent.get("taint_reasons") or []))
        run_ids.update(str(r) for r in (parent.get("parent_run_ids") or []))
        if parent.get("diagnostic_only") and parent.get("producing_run_id"):
            run_ids.add(str(parent["producing_run_id"]))
    return {"diagnostic_only": bool(reasons), "taint_reasons": sorted(reasons),
            "parent_run_ids": sorted(run_ids)}


def publish(artifact: str | Path, frame: pd.DataFrame | None = None, *,
            stage: str, cfg: Mapping[str, Any], run_dir: str | Path | None = None,
            parents: Sequence[Mapping[str, Any]] = (),
            taint_reasons: Iterable[str] = (),
            key_columns: Sequence[str] = (),
            schema: str | None = None,
            extra: Mapping[str, Any] | None = None,
            logical_path: str | Path | None = None) -> dict[str, Any]:
    """Write the manifest for an artifact that has just been written.

    Call this *after* the artifact exists; the manifest hashes it, so a manifest
    can never describe bytes that were never written.
    """
    path = Path(artifact)
    if not path.is_file():
        raise FileNotFoundError(f"cannot publish a manifest for a missing artifact: {path}")
    taint = merge_taint(parents, extra_reasons=taint_reasons)
    payload = {
        "manifest_version": MANIFEST_VERSION,
        # Generation-atomic publishers write into an attempt-private directory
        # and rename the whole directory only after its complete manifest is
        # durable. In that case the bytes are hashed at `path`, while consumers
        # must see the immutable final location recorded here.
        "path": str(Path(logical_path)) if logical_path is not None else str(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "rows": None if frame is None else int(len(frame)),
        "complete": True,
        "producing_stage": stage,
        "producing_run_id": run_id(stage, run_dir),
        "created_at": _now(),
        "identity": identity(cfg),
        "expected_keys_sha256": expected_keys_hash(frame, key_columns)
        if frame is not None else None,
        "key_columns": list(key_columns),
        "schema": schema,
        "parent_artifacts": [
            {"path": p.get("path"), "sha256": p.get("sha256"),
             "run_id": p.get("producing_run_id")}
            for p in parents if p
        ],
        **taint,
        **(dict(extra) if extra else {}),
    }
    atomic_write_json(manifest_path(path), payload)
    return payload


def publish_frame(artifact: str | Path, frame: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
    """Atomically write a parquet and its manifest together."""
    path = Path(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path(path).unlink(missing_ok=True)     # no stale manifest mid-write
    with atomic_output_path(path) as tmp:
        frame.to_parquet(tmp, index=False)
    return publish(path, frame, **kwargs)


def load(artifact: str | Path) -> dict[str, Any] | None:
    path = manifest_path(artifact)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def verify(artifact: str | Path, *, cfg: Mapping[str, Any] | None = None,
           require_identity: bool = False,
           frame: pd.DataFrame | None = None) -> dict[str, Any]:
    """Is this artifact authentic, complete, and produced by this configuration?

    Returns ``{"ok": bool, "verdict": str, "manifest": dict | None, ...}``. The
    caller decides whether a failure is `blocked` (unauthenticated evidence) or
    `failed` (corrupt artifact); this function never guesses.
    """
    path = Path(artifact)
    out: dict[str, Any] = {"path": str(path), "ok": False, "verdict": OK,
                           "manifest": None, "detail": ""}
    if not path.is_file():
        out["verdict"] = MISSING_ARTIFACT
        return out
    payload = load(path)
    if payload is None:
        out["verdict"] = MISSING_MANIFEST if not manifest_path(path).exists() \
            else BAD_MANIFEST
        out["detail"] = (f"{path.name} has no authenticated provenance; it may be "
                         "a leftover from another configuration or a partial write")
        return out
    out["manifest"] = payload
    if not payload.get("complete", False):
        out["verdict"] = INCOMPLETE
        return out
    actual = sha256_file(path)
    if actual != payload.get("sha256"):
        out["verdict"] = SHA_MISMATCH
        out["detail"] = f"expected {payload.get('sha256')}, found {actual}"
        return out
    if require_identity and cfg is not None:
        raw_identity = payload.get("identity")
        have = dict(raw_identity) if isinstance(raw_identity, Mapping) else {}
        identity_format, format_problem = _identity_format(have)
        if identity_format is None:
            out["verdict"] = IDENTITY_MISMATCH
            out["detail"] = f"malformed source identity: {format_problem}"
            return out
        if identity_format == "legacy":
            want = _legacy_identity(cfg)
            gated_keys = tuple(want)
        else:
            want = identity(cfg)
            # test_sha256 remains recorded in the immutable manifest, but it is
            # deliberately not an execution gate.
            gated_keys = tuple(k for k in want if k != "test_sha256")
        differing = sorted(k for k in gated_keys if want[k] != have.get(k))
        if differing:
            out["verdict"] = IDENTITY_MISMATCH
            out["detail"] = f"differs in {differing}"
            return out
    if frame is not None and payload.get("expected_keys_sha256"):
        recomputed = expected_keys_hash(frame, payload.get("key_columns") or ())
        if recomputed != payload["expected_keys_sha256"]:
            out["verdict"] = KEY_MISMATCH
            out["detail"] = ("the rows present are not the rows the producer "
                             "recorded; the artifact is truncated or edited")
            return out
    out["ok"] = True
    return out


def read_verified(artifact: str | Path, *, cfg: Mapping[str, Any] | None = None,
                  require_identity: bool = False
                  ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a parquet only if its manifest authenticates it.

    Returns an empty frame with the failing verdict rather than raising, so a
    stage can turn unauthenticated evidence into a gate outcome.
    """
    path = Path(artifact)
    verdict = verify(path, cfg=cfg, require_identity=require_identity)
    if not verdict["ok"]:
        return pd.DataFrame(), verdict
    frame = pd.read_parquet(path)
    verdict = verify(path, cfg=cfg, require_identity=require_identity, frame=frame)
    if not verdict["ok"]:
        return pd.DataFrame(), verdict
    return frame, verdict


def taint_of_inputs(manifests: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    """Taint a consumer inherits purely from the artifacts it read."""
    return merge_taint([m for m in manifests if m])


def rebuilt_in_this_run(manifests: Sequence[Mapping[str, Any] | None],
                        this_run_id: str) -> bool:
    """Did this run produce every artifact it is about to rely on?

    `--overwrite` may clear a stage's historical taint only when this is true;
    otherwise some of the bytes predate the recompute.
    """
    present = [m for m in manifests if m]
    if not present:
        return False
    return all(str(m.get("producing_run_id")) == this_run_id for m in present)


def sha256_of(payload: Any) -> str:
    return sha256_obj(payload)
