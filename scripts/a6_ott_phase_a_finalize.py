#!/usr/bin/env python3
"""Validate and record the completed A6-OTT Phase-A physical search."""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
from collections import Counter
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "a6_ott_upper_bound"
ROWS = RESULTS / "phase_a" / "rows"


def _sha(obj):
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _elapsed(value: str) -> int:
    parts = value.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2]))
    if len(parts) == 4:
        return (int(parts[0]) * 86400 + int(parts[1]) * 3600 +
                int(parts[2]) * 60 + int(float(parts[3])))
    return 0


def _expected():
    freeze = json.loads((RESULTS / "manifests" / "v2" / "SEARCH_FREEZE.json").read_text())
    models = {"whisper": (32, 32, (0.5, 1.0, 2.0)),
              "qwen3_asr_1p7b": (24, 28, (0.5, 1.0, 2.0, 4.0))}
    out = set()
    for model, (enc, dec, rhos) in models.items():
        for dataset, ids in freeze["ids"].items():
            for uid in ids:
                for side, n in (("encoder", enc), ("decoder", dec)):
                    families = ("add_unique",) if side == "encoder" else ("add_unique", "conditioning_cs")
                    site = ("encoder_post_self_attn_residual_pre_ffn" if side == "encoder"
                            else "decoder_post_cross_attn_residual")
                    for family in families:
                        for layer in range(n):
                            for rho in rhos:
                                out.add("|".join(map(str, ("A", model, dataset, uid, family,
                                    side, layer, float(rho), "greedy", site))))
    return out


def main():
    expected = _expected()
    seen = {}
    invalid = []
    for path in ROWS.glob("**/*.json"):
        try:
            row = json.loads(path.read_text())
        except Exception as exc:
            invalid.append({"path": str(path), "error": str(exc)})
            continue
        key = row.get("canonical_key")
        if key not in expected:
            continue
        if key in seen:
            invalid.append({"path": str(path), "error": "duplicate canonical key"})
            continue
        if row.get("status") != "PASS" or row.get("phase") != "A" or row.get("regime") != "oracle_tt":
            invalid.append({"path": str(path), "error": "non-PASS or wrong phase/regime"})
        if (row.get("provenance") or {}).get("fixed_direction_artifact") is not None:
            invalid.append({"path": str(path), "error": "fixed direction provenance"})
        seen[key] = row
    missing = sorted(expected - set(seen))
    reuse_runs = {"53468", "53469"}
    reused = [row for row in seen.values()
              if str((row.get("provenance") or {}).get("source_run_id")) in reuse_runs]
    counts = Counter(row["model"] for row in seen.values())
    if missing or invalid or len(seen) != len(expected):
        raise SystemExit(json.dumps({"missing": len(missing), "invalid": len(invalid)}, indent=2))

    # Batch elapsed excludes queue time and is the GPU-work accounting surface.
    jobs = {"whisper": ["53470_0", "53470_1", "53470_2", "53470_3"],
            "qwen3_asr_1p7b": ["53471_0", "53471_1", "53471_2", "53471_3"]}
    gpu_seconds = {}
    for model, ids in jobs.items():
        raw = subprocess.check_output(["sacct", "-D", "-j", ",".join(ids),
            "--format=JobID,Elapsed", "-n", "-P"], text=True)
        seconds = 0
        for line in raw.splitlines():
            job, elapsed = line.split("|", 1)
            if job.endswith(".batch"):
                seconds += _elapsed(elapsed)
        gpu_seconds[model] = seconds
    payload = {
        "schema_version": "a6_ott_phase_a_completion_v1",
        "status": "PASS",
        "phase": "A",
        "manifest": "results/a6_ott_upper_bound/manifests/v2/SEARCH_FREEZE.json",
        "manifest_freeze_hash": json.loads((RESULTS / "manifests/v2/SEARCH_FREEZE.json").read_text())["freeze_hash"],
        "logical": {"whisper": 11520, "qwen3_asr_1p7b": 12800},
        "resolved": {"whisper": counts["whisper"], "qwen3_asr_1p7b": counts["qwen3_asr_1p7b"]},
        "exact_reused": {"whisper": sum(r["model"] == "whisper" for r in reused),
                         "qwen3_asr_1p7b": sum(r["model"] == "qwen3_asr_1p7b" for r in reused)},
        "computed_new": {"whisper": counts["whisper"] - sum(r["model"] == "whisper" for r in reused),
                         "qwen3_asr_1p7b": counts["qwen3_asr_1p7b"] - sum(r["model"] == "qwen3_asr_1p7b" for r in reused)},
        "preserved_out_of_scope_rows": 63,
        "validation": {"missing": len(missing), "invalid": len(invalid),
                        "expected_key_hash": _sha(sorted(expected)),
                        "accepted_key_hash": _sha(sorted(seen)),
                        "confirmation_and_seame_outcomes_loaded": False},
        "jobs": [53470, 53471],
        "gpu_seconds_including_failed_pre_fix_attempts": gpu_seconds,
        "gpu_hours_including_failed_pre_fix_attempts": {k: v / 3600 for k, v in gpu_seconds.items()},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out = RESULTS / "search" / "PHASE_A_COMPLETION.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
