"""Config loading with `include:` merging, CLI overrides and path helpers."""
from __future__ import annotations

import copy
import datetime as dt
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config, resolving `include:` lists relative to configs/.

    Includes are merged first (in order), then the file's own keys win.
    ``overrides`` are ``dotted.key=value`` strings parsed as YAML scalars.
    """
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_ROOT / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    merged: dict[str, Any] = {}
    for inc in raw.pop("include", []) or []:
        inc_path = Path(inc)
        if not inc_path.is_absolute():
            inc_path = CONFIG_ROOT / inc
        merged = _deep_merge(merged, load_config(inc_path))
    merged = _deep_merge(merged, raw)
    merged.pop("defaults", None)

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        key, value = ov.split("=", 1)
        node = merged
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)

    merged["_config_path"] = str(path)   # the entry point, not the last include
    return merged


def artifacts_root(cfg: dict) -> Path:
    return Path(cfg["experiment"]["output_root"])


def art(cfg: dict, *parts: str) -> Path:
    """Path inside the artifacts tree, parent directories created."""
    p = artifacts_root(cfg).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def run_dir(cfg: dict, stage: str, tag: str | None = None) -> Path:
    stage_name = stage if tag is None else f"{stage}_{tag}"
    invocation = os.environ.get("SLURM_JOB_ID")
    if not invocation:
        invocation = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    p = artifacts_root(cfg) / "runs" / stage_name / invocation
    p.mkdir(parents=True, exist_ok=True)
    return p
