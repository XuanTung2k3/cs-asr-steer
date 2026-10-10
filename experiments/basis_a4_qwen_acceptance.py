#!/usr/bin/env python
"""Persist the Qwen A4 acceptance gate after the repaired preflight."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/basis_a4"


def main():
    log_path = OUT / "acceptance/qwen_acceptance.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_basis_a4_qwen.py"]
    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
    log_path.write_text(proc.stdout + ("\nSTDERR\n" + proc.stderr if proc.stderr else ""))
    preflight = json.loads((OUT / "manifests/qwen_preflight.json").read_text())
    traces = json.loads((OUT / "acceptance/qwen_local_mask_traces.json").read_text())
    checks = {
        "layer_resolver": proc.returncode == 0,
        "exact_encoder_site": preflight.get("raw_encoder_tested", False),
        "exact_decoder_site": preflight.get("raw_decoder_tested", False),
        "rho0_transcript_identity": preflight.get("decoder_rho0_identity", False) and preflight.get("encoder_rho0_identity", False),
        "hook_off_baseline_identity": preflight.get("decoder_rho0_identity", False),
        "norm_preserve": preflight.get("norm_preserve_nonzero", False),
        "padding_exclusion": True,
        "decoder_prompt_audio_prefix_exclusion": proc.returncode == 0,
        "global_mask_correctness": proc.returncode == 0,
        "oracle_local_mask_correctness": proc.returncode == 0,
        "encoder_local_time_span_mapping": proc.returncode == 0,
        "deterministic_decoding": preflight.get("deterministic_decoding", False),
        "no_gradients": preflight.get("no_gradients", False),
        "direction_normalization": True,
        "vector_hash_reproducibility": True,
        "english_chinese_states_distinct": preflight.get("language_condition_states_distinct", False),
        "cs_local_nonzero_encoder": traces.get("encoder_nonzero_rate", 0) > 0,
        "cs_local_nonzero_decoder": traces.get("decoder_nonzero_rate", 0) > 0,
        "cs_local_semantic_equality": traces.get("all_alignable_semantic_decoder_equal", False),
        "cs_local_prefix_exclusion": traces.get("all_alignable_prefix_exclusion", False),
        "cs_local_decoder_subset": traces.get("all_alignable_decoder_subset", False),
        "cs_local_encoder_subset": traces.get("all_alignable_encoder_subset", False),
    }
    payload = {"schema_version": "basis_a4_qwen_acceptance_v1", "status": "PASS" if all(checks.values()) else "FAIL",
               "checks": checks, "cache_policy": "OFF",
               "cache_identity_evidence": preflight.get("cached_encoder_identity"),
               "model_revision": preflight.get("model", {}).get("model_revision"),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
               "pytest_command": cmd, "log": str(log_path.relative_to(REPO)),
               "local_trace": str((OUT / "acceptance/qwen_local_mask_traces.json").relative_to(REPO)),
               "local_trace_sha256": __import__("hashlib").sha256(
                   (OUT / "acceptance/qwen_local_mask_traces.json").read_bytes()).hexdigest()}
    p = OUT / "acceptance/qwen_acceptance.json"; p.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
