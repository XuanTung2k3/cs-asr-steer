"""Seal the dialogue-atomic v2 spec in its isolated namespace.

This command does not run L0. It reads the immutable v1 freeze only to record a
formal supersession pointer, authenticates the already-published v2 dialogue
roles, and atomically publishes a new v2 freeze plus artifact manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from ..lss import manifest as manifest_mod
from ..lss.artifacts import artifact_ref
from ..lss.specfreeze import load as load_spec
from ..lss.specfreeze import supersede_at, verify
from ..lss.v2_freeze import build_v2_spec
from ..lss.v2_namespace import configured_paths, validate_namespace
from ..utils.config import load_config

STAGE = "lss_v2_spec_freeze"
DEFAULT_CONFIG = "lss/l1b_candidates_dialogue_v2.yaml"
SUPERSESSION_REASON = (
    "Dialogue-atomic repartition corrects the independent cluster from a "
    "single speaker channel (conversation_id) to the two-party session "
    "(dialogue_id). Balance thresholds remain unchanged; gated balance now "
    "acts as a feasibility constraint. All derived artifacts move to an "
    "isolated v2 namespace."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    isolation = validate_namespace(cfg, require_freeze=False)
    paths = configured_paths(cfg)

    final_dir = paths.freeze_path.parent
    if final_dir.exists() or manifest_mod.manifest_path(paths.freeze_path).exists():
        raise SystemExit(
            f"refusing to overwrite the immutable v2 freeze generation: {final_dir}")
    paths.namespace_root.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(
        prefix=".freeze-v2-attempt-", dir=str(paths.namespace_root)))
    promoted = False
    try:
        attempt_freeze = attempt / paths.freeze_path.name
        spec = build_v2_spec(cfg)
        supersede_at(
            paths.superseded_freeze, spec, attempt_freeze,
            reason=SUPERSESSION_REASON)
        verification = verify(attempt_freeze)
        if not verification["ok"]:
            raise SystemExit(
                "new v2 freeze did not self-verify; nothing was promoted: "
                + json.dumps(verification, default=str))

        old_parent = artifact_ref(
            paths.superseded_freeze, schema="lss_spec_freeze_v1")
        evidence_parents = [old_parent]
        for artifact in (
            paths.namespace_root / "manifests" /
                "cs_dialogue_with_dialogue_id.parquet",
            paths.role_root / "dialogue_role_report.json",
        ):
            published = manifest_mod.load(artifact)
            if not published:
                raise SystemExit(f"required parent has no artifact manifest: {artifact}")
            evidence_parents.append(published)
        manifest_mod.publish(
            attempt_freeze, stage=STAGE, cfg=cfg, parents=evidence_parents,
            schema="lss_spec_freeze_v2", logical_path=paths.freeze_path,
            extra={
                "namespace_version": "lss-dialogue-v2",
                "supersedes": old_parent,
                "dialogue_derivation_fingerprint": spec.get(
                    "dialogue_derivation", "fingerprint"),
                "role_partition_fingerprint": spec.get(
                    "roles", "role_partition_fingerprint"),
            })
        # Both the freeze and its sidecar become visible together. The v1 tree
        # is only a parent and is never a rename/write destination.
        os.rename(attempt, final_dir)
        promoted = True
    finally:
        if not promoted and attempt.exists():
            shutil.rmtree(attempt)

    sealed = load_spec(paths.freeze_path)
    result = {
        "state": "completed",
        "freeze": str(paths.freeze_path),
        "sha256": sealed.sha256,
        "supersedes": sealed.get("supersedes"),
        "isolation": isolation,
        "verification": verify(paths.freeze_path),
    }
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
