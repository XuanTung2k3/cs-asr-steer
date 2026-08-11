"""Stage optional checkpoints into the persistent cache. Login node only.

Compute nodes run with HF_HUB_OFFLINE=1, so anything not already on disk has to
be fetched here first. Failures are recorded and returned as a non-zero exit
code rather than raised, so a missing optional model degrades the pipeline
instead of crashing a stage six hours in.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..lss.artifacts import write_json
from ..nat5h.model_cache import ensure_qwen_checkpoint
from ..utils.config import artifacts_root, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="lss/l1a_diag.yaml")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    root = artifacts_root(cfg)
    cache = Path("/mnt/data/tungnx/cs-asr-steer/model_cache")
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache / "huggingface" / "hub"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))

    qwen_cfg = ((cfg.get("alignment") or {}).get("qwen") or {})
    results: dict[str, dict] = {}
    ok = True

    if qwen_cfg.get("model_id"):
        marker = root / "models" / "qwen3_forced_aligner_complete.json"
        try:
            results["qwen_forced_aligner"] = {
                "state": "ok",
                "detail": ensure_qwen_checkpoint(
                    model_id=qwen_cfg["model_id"],
                    local_model_dir=qwen_cfg["local_model_dir"],
                    cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
                    marker_path=marker,
                    allow_download=not args.no_download,
                ),
            }
        except Exception as exc:
            ok = False
            results["qwen_forced_aligner"] = {
                "state": "blocked", "error": repr(exc),
                "expected_local_model_dir": qwen_cfg.get("local_model_dir"),
                "note": "the alignment consensus falls back to two families",
            }

    ctc_cfg = ((cfg.get("alignment") or {}).get("ctc") or {})
    ctc_path = Path(str(ctc_cfg.get("model_id", "")))
    results["existing_ctc"] = {
        "state": "ok" if ctc_path.exists() else "missing",
        "path": str(ctc_path),
        "note": "MMS-FA wav2vec2 CTC driven by uroman romanization",
    }
    if not ctc_path.exists():
        ok = False

    payload = {"ok": ok, "results": results,
               "hf_home": os.environ.get("HF_HOME")}
    write_json(payload, root / "models" / "lss_model_acquisition.json")
    print(json.dumps(payload, indent=2, default=str))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
