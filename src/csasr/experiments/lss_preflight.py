"""Assert the environment before any stage spends GPU time.

Every check here corresponds to something that has actually broken this project
once: torchaudio failing to load against the installed torch, scipy missing a
CXXABI symbol when the system libstdc++ wins, a source artifact silently
missing, or a stage writing into the wrong artifacts root.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..lss.artifacts import write_json
from ..utils.config import artifacts_root, load_config


def _check_imports() -> dict:
    out: dict[str, object] = {}
    try:
        import scipy  # noqa: F401
        out["scipy"] = "ok"
    except Exception as exc:                                   # pragma: no cover
        out["scipy"] = f"FAILED: {exc!r} (is LD_LIBRARY_PATH set to $CONDA_PREFIX/lib?)"
    for name in ("torch", "transformers", "sklearn", "pandas", "numpy", "soundfile"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:                               # pragma: no cover
            out[name] = f"FAILED: {exc!r}"
    try:                                    # known broken; must never be a dependency
        import torchaudio  # noqa: F401
        out["torchaudio"] = "importable"
    except Exception as exc:
        out["torchaudio"] = f"unavailable ({type(exc).__name__}); no LSS code may use it"
    return out


def _check_paths(cfg: dict) -> tuple[dict, list[str]]:
    experiment = cfg.get("experiment") or {}
    data = cfg.get("data") or {}
    source = Path(str(experiment.get("source_artifacts_root", "")))
    required = {
        "corpus_manifest": Path(str(data.get("manifest", ""))),
        "train_manifest": source / "manifests" / "train.parquet",
        "dev_select_manifest": source / "manifests" / "dev_select.parquet",
        "dev_confirm_manifest": source / "manifests" / "dev_confirm.parquet",
        "primary_baseline": Path(str(data.get("primary_baseline", ""))),
        "whisper_model": Path(str((cfg.get("model") or {}).get("id", ""))),
        "audio_root": Path(str(data.get("audio_root", ""))),
    }
    report, problems = {}, []
    for name, path in required.items():
        ok = path.exists()
        report[name] = {"path": str(path), "exists": bool(ok)}
        if not ok:
            problems.append(f"{name} missing at {path}")
    return report, problems


def _check_gpu() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        return {
            "cuda_available": True,
            "device": torch.cuda.get_device_name(0),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1),
        }
    except Exception as exc:                                   # pragma: no cover
        return {"cuda_available": False, "error": repr(exc)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LSS preflight checks")
    parser.add_argument("--config", default="lss/base.yaml")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    root = artifacts_root(cfg)
    root.mkdir(parents=True, exist_ok=True)

    paths, problems = _check_paths(cfg)
    gpu = _check_gpu()
    if args.require_gpu and not gpu.get("cuda_available"):
        problems.append("CUDA requested but unavailable")

    report = {
        "artifacts_root": str(root),
        "config": cfg.get("_config_path"),
        "imports": _check_imports(),
        "paths": paths,
        "gpu": gpu,
        "environment": {k: os.environ.get(k) for k in
                        ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHASHSEED", "TMPDIR",
                         "HF_HUB_OFFLINE", "MPLCONFIGDIR", "SLURM_JOB_ID")},
        "problems": problems,
        "ok": not problems,
    }
    write_json(report, root / "diagnostics" / "preflight.json")
    print(json.dumps(report, indent=2, default=str))
    if problems:
        print("\nPREFLIGHT FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
