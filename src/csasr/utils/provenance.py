"""Small provenance helpers for scientific stage artifacts."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from .config import REPO_ROOT
from .hashing import sha256_file, sha256_obj, sha256_strings

SOURCE_ROOTS = ("src", "tests", "configs")
# Launchers set the environment a run executed in (library paths, offline flags,
# thread counts), so they define the run as much as the code does. The
# environment lock is written by `cs_asr_lss.sh preflight`; it is absent on a
# fresh checkout and simply skipped.
SOURCE_FILES = ("cs_asr_e1_e5.sh", "cs_asr_nat5h.sh", "cs_asr_lss.sh",
                "pyproject.toml", "environment.lock.txt")
PROVENANCE_VERSION = "artifact-provenance-v1"


def source_snapshot_hash(repo_root: str | Path = REPO_ROOT) -> str:
    """Hash source/config/launcher files that define a scientific run."""
    root = Path(repo_root)
    entries: list[str] = []
    for rel in SOURCE_ROOTS:
        base = root / rel
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                entries.append(f"{p.relative_to(root)}={sha256_file(p)}")
    for rel in SOURCE_FILES:
        p = root / rel
        if p.exists():
            entries.append(f"{p.relative_to(root)}={sha256_file(p)}")
    return sha256_strings(entries)


def resolved_config_hash(cfg: dict[str, Any]) -> str:
    """Hash the resolved config, excluding volatile absolute run metadata only."""
    return sha256_obj(cfg)


def dataset_manifest_hash(cfg: dict[str, Any]) -> str | None:
    path = Path(str(cfg.get("data", {}).get("manifest", "")))
    return sha256_file(path) if path.is_file() else None


def production_artifact_flag(cfg: dict[str, Any]) -> bool:
    root = Path(str(cfg.get("experiment", {}).get("output_root", "")))
    return "smoke" not in {part.lower() for part in root.parts}


def stage_provenance(cfg: dict[str, Any], stage: str,
                     upstream: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Metadata required to decide whether a downstream artifact is reusable."""
    model = cfg.get("model", {}) or {}
    exp = cfg.get("experiment", {}) or {}
    return {
        "provenance_version": PROVENANCE_VERSION,
        "stage": stage,
        "phase": stage.split("_", 1)[1] if "_" in stage else None,
        "model_id": model.get("id") or model.get("hub_id"),
        "model_revision": model.get("revision"),
        "dataset_manifest_hash": dataset_manifest_hash(cfg),
        "resolved_config_hash": resolved_config_hash(cfg),
        "source_snapshot_hash": source_snapshot_hash(),
        "normalization_version": exp.get("normalization_version"),
        "upstream_artifacts": upstream or [],
        "seed": exp.get("seed"),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "production_artifact": production_artifact_flag(cfg),
    }


def assert_production_artifact(payload: dict[str, Any], stage: str) -> None:
    """Reject stale/smoke upstream status payloads before downstream loading."""
    if payload.get("status") != "passed":
        raise RuntimeError(f"upstream {stage} is not passed: {payload.get('status')}")
    prov = payload.get("provenance") or {}
    if prov.get("production_artifact") is False:
        raise RuntimeError(f"upstream {stage} is a smoke artifact, not production")
