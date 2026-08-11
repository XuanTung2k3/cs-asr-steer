"""Ensure NAT5H optional model checkpoints exist in persistent caches."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from csasr.nat5h.model_cache import ensure_qwen_checkpoint
from csasr.nat5h.statusing import atomic_write_json
from csasr.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nat5h.yaml")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cache = cfg["model_cache"]
    os.environ.setdefault("HF_HOME", cache["hf_home"])
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache["hf_hub_cache"])
    os.environ.setdefault("TORCH_HOME", cache["torch_home"])

    out_root = Path(cfg["experiment"]["output_root"])
    marker = out_root / "models" / "qwen3_forced_aligner_complete.json"
    try:
        result = ensure_qwen_checkpoint(
            model_id=cfg["qwen_aligner"]["model_id"],
            local_model_dir=cfg["qwen_aligner"]["local_model_dir"],
            cache_dir=cache["hf_hub_cache"],
            marker_path=marker,
            allow_download=not args.no_download,
            lock_timeout_seconds=int(cfg["qwen_aligner"].get("lock_timeout_seconds", 7200)),
        )
    except Exception as exc:
        payload = {
            "state": "blocked",
            "reason": "qwen_checkpoint_unavailable",
            "error": repr(exc),
            "model_id": cfg["qwen_aligner"]["model_id"],
            "expected_local_model_dir": cfg["qwen_aligner"]["local_model_dir"],
            "hf_home": os.environ.get("HF_HOME"),
            "huggingface_hub_cache": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        }
        atomic_write_json(out_root / "status" / "n0_model_acquisition_blocked.json", payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 2

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
