"""Run logging: console + per-run log file, plus environment capture."""
from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

_CONFIGURED = False


def get_logger(name: str = "csasr") -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(run_dir: str | Path | None = None, level: str = "INFO") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger("csasr")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    if not _CONFIGURED:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        _CONFIGURED = True
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(run_dir / "run.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "executable": sys.executable,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_gb"] = round(props.total_memory / 1e9, 2)
    except ImportError:
        info["torch"] = None
    for pkg in ("transformers", "numpy", "pandas", "scipy", "sklearn", "soundfile"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[pkg] = None
    return info


def git_state(repo_dir: str | Path) -> dict[str, Any]:
    repo_dir = str(repo_dir)

    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                args, cwd=repo_dir, capture_output=True, text=True, timeout=15
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "is_git_repo": commit is not None,
        "commit": commit,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines() if status else [],
    }


def write_run_metadata(run_dir: str | Path, config: dict, repo_dir: str | Path,
                       extra: dict | None = None) -> None:
    """Write the reproducibility bundle required by the guide (section 6.3)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    import yaml

    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps(environment_info(), indent=2, default=str), encoding="utf-8"
    )
    (run_dir / "git_state.json").write_text(
        json.dumps(git_state(repo_dir), indent=2), encoding="utf-8"
    )
    if extra:
        for name, payload in extra.items():
            (run_dir / f"{name}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
