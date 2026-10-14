"""Paths, config and write-once I/O shared by every ReFT P0/P1 command."""
from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/reft/p0p1.yaml"


def cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def art(*parts) -> Path:
    """Artifact path under the local (never versioned) artifact root."""
    return Path(cfg()["artifact_root"]).joinpath(*parts)


def write_json(path: Path, obj, overwrite: bool = False) -> Path:
    """Deterministic JSON. Write-once unless ``overwrite``: sealed artifacts are never silently replaced."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_jsonl(path: Path, rows) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def jload_gz(path: Path):
    with gzip.open(path, "rb") as g:
        return json.loads(g.read())


def audio_hash_ok(path: str, expected: str) -> bool:
    """Audio identity in the role manifest's convention: sha256(str(size) + first ``audio_hash_max_bytes`` bytes)."""
    from csasr.utils.hashing import sha256_file
    got = sha256_file(path, max_bytes=cfg()["data"]["audio_hash_max_bytes"])
    return got.removeprefix("sha256:") == str(expected).removeprefix("sha256:")


def fillers() -> set[str]:
    f = cfg()["evaluation"]["fillers"]
    return set(f["ZH"]) | {w.lower() for w in f["EN"]}


# ------------------------------------------------------------------ reference firewall

_FIREWALL = {"on": False, "patterns": ()}


def firewall(patterns) -> None:
    """Refuse to open any path containing one of ``patterns`` while the firewall is on (decode phases).
    The audit hook is permanent for the process; ``firewall_off`` only lowers it for a later label-using phase."""
    _FIREWALL["patterns"] = tuple(patterns)
    if not _FIREWALL.get("installed"):
        def audit(ev, args):
            if _FIREWALL["on"] and ev == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
                s = os.fsdecode(args[0])
                if any(p in s for p in _FIREWALL["patterns"]):
                    raise RuntimeError("reference firewall: " + s)
        sys.addaudithook(audit)
        _FIREWALL["installed"] = True
    _FIREWALL["on"] = True


def firewall_off() -> None:
    _FIREWALL["on"] = False
