"""Isolation contract for the dialogue-atomic LSS v2 namespace.

The v1 artifacts are an immutable record of the conversation-clustered run.
Dialogue-atomic roles and every artifact derived from them therefore live under
another namespace and candidate generations use a versioned, write-once root.
This module contains only path/config validation; it never writes an artifact.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class NamespaceIsolationError(RuntimeError):
    """A configured v2 path could alias, overwrite, or mix with v1."""


@dataclass(frozen=True)
class V2Paths:
    namespace_root: Path
    protected_v1_root: Path
    superseded_freeze: Path
    role_root: Path
    candidate_generation_root: Path
    candidate_path: Path
    freeze_path: Path

    def to_dict(self) -> dict[str, str]:
        return {name: str(value) for name, value in self.__dict__.items()}


def _absolute_unaliased(value: Any, *, field: str) -> Path:
    raw = Path(str(value or ""))
    if not raw.is_absolute():
        raise NamespaceIsolationError(f"{field} must be an absolute path: {raw}")
    absolute = raw.absolute()
    resolved = raw.resolve(strict=False)
    # Refuse a symlink in any existing component.  Comparing the normalized
    # absolute spelling to resolve() closes both a direct symlink and a symlinked
    # parent; otherwise a path that looks outside v1 can resolve inside it.
    if absolute != resolved:
        raise NamespaceIsolationError(
            f"{field} contains a symlink or alias: {absolute} -> {resolved}")
    return resolved


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def configured_paths(cfg: Mapping[str, Any]) -> V2Paths:
    block = dict(cfg.get("v2_namespace") or {})
    if str(block.get("version") or "") != "lss-dialogue-v2":
        raise NamespaceIsolationError(
            "v2_namespace.version must be exactly 'lss-dialogue-v2'")

    paths = V2Paths(
        namespace_root=_absolute_unaliased(block.get("root"), field="v2_namespace.root"),
        protected_v1_root=_absolute_unaliased(
            block.get("protected_v1_root"), field="v2_namespace.protected_v1_root"),
        superseded_freeze=_absolute_unaliased(
            block.get("superseded_freeze"), field="v2_namespace.superseded_freeze"),
        role_root=_absolute_unaliased(
            block.get("role_root"), field="v2_namespace.role_root"),
        candidate_generation_root=_absolute_unaliased(
            block.get("candidate_generation_root"),
            field="v2_namespace.candidate_generation_root"),
        candidate_path=_absolute_unaliased(
            block.get("candidate_path"), field="v2_namespace.candidate_path"),
        freeze_path=_absolute_unaliased(
            block.get("freeze_path"), field="v2_namespace.freeze_path"),
    )
    expected_roles = paths.namespace_root / "manifests" / "roles"
    expected_generations = paths.namespace_root / "candidate_generations"
    expected_candidate = (paths.candidate_generation_root / "alignments" /
                          "candidates_all.parquet")
    expected_freeze = paths.namespace_root / "freeze" / "spec_freeze_v2.json"
    configured_output = _absolute_unaliased(
        (cfg.get("experiment") or {}).get("output_root"),
        field="experiment.output_root")
    configured_roles = _absolute_unaliased(
        (cfg.get("experiment") or {}).get("role_manifests_root"),
        field="experiment.role_manifests_root")

    errors: list[str] = []
    if _is_within(paths.namespace_root, paths.protected_v1_root) or \
            _is_within(paths.protected_v1_root, paths.namespace_root):
        errors.append("v1 and v2 roots overlap")
    expected_v1_freeze = (paths.protected_v1_root / "freeze" /
                          "spec_freeze_v1.json")
    if paths.superseded_freeze != expected_v1_freeze:
        errors.append(f"superseded_freeze must be {expected_v1_freeze}")
    if paths.role_root != expected_roles:
        errors.append(f"role_root must be {expected_roles}")
    if paths.candidate_generation_root.parent != expected_generations:
        errors.append(
            f"candidate_generation_root must be a direct child of {expected_generations}")
    if paths.candidate_path != expected_candidate:
        errors.append(f"candidate_path must be {expected_candidate}")
    if paths.freeze_path != expected_freeze:
        errors.append(f"freeze_path must be {expected_freeze}")
    if configured_output != paths.candidate_generation_root:
        errors.append("experiment.output_root must equal candidate_generation_root")
    if configured_roles != paths.role_root:
        errors.append("experiment.role_manifests_root must equal role_root")
    for label, path in paths.to_dict().items():
        candidate = Path(path)
        if label not in {"protected_v1_root", "superseded_freeze"} and _is_within(
                candidate, paths.protected_v1_root):
            errors.append(f"{label} resolves inside the protected v1 root")
    if errors:
        raise NamespaceIsolationError("; ".join(errors))
    return paths


def validate_namespace(cfg: Mapping[str, Any], *, for_execution: bool = False,
                       require_freeze: bool = True) -> dict[str, Any]:
    """Validate the immutable role input and write-once candidate destination."""
    paths = configured_paths(cfg)
    role_generation = paths.role_root / "generation_complete.manifest.json"
    if not role_generation.is_file():
        raise NamespaceIsolationError(
            f"v2 role generation is incomplete: {role_generation} is missing")
    try:
        generation = json.loads(role_generation.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise NamespaceIsolationError(
            f"v2 role generation manifest is unreadable: {role_generation}") from exc
    if not generation.get("complete") or generation.get("state") != "completed":
        raise NamespaceIsolationError(
            f"v2 role generation is not complete: {role_generation}")
    if require_freeze and not paths.freeze_path.is_file():
        raise NamespaceIsolationError(f"v2 spec freeze is missing: {paths.freeze_path}")
    if for_execution:
        # The generation root itself is the write-once boundary.  A partial
        # interrupted attempt is never resumed or treated as a valid cache.
        if paths.candidate_generation_root.exists():
            raise NamespaceIsolationError(
                "candidate generation destination already exists; it is immutable "
                f"even when incomplete: {paths.candidate_generation_root}")
        sidecar = Path(str(paths.candidate_path) + ".manifest.json")
        if paths.candidate_path.exists() or sidecar.exists():
            raise NamespaceIsolationError(
                f"candidate artifact or sidecar already exists: {paths.candidate_path}")
    return {
        "ok": True,
        "namespace_version": "lss-dialogue-v2",
        "paths": paths.to_dict(),
        "role_generation": {
            "path": str(role_generation),
            "role_assignment_hash": generation.get("role_assignment_hash"),
            "role_partition_fingerprint": generation.get(
                "role_partition_fingerprint"),
        },
        "candidate_generation_destination_exists":
            paths.candidate_generation_root.exists(),
    }
