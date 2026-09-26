#!/usr/bin/env python3
"""Audit accepted A4 Oracle-local site/mask artifacts for A6 reuse."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/basis_a6_expanded/preflight/LOCAL_MASK_ACCEPTANCE.json"


def main() -> int:
    expected = {
        ("whisper", "encoder"): (REPO / "results/basis_a4/whisper/raw/encoder/cs_dialogue/L16/Raw_encoder_L16_oracle_local_rho0.5.json",
                                   "encoder_post_self_attn_residual_pre_ffn"),
        ("whisper", "decoder"): (REPO / "results/basis_a4/whisper/conditioning/cs_dialogue/L16/conditioning_decoder_L16_oracle_local_rho0.5.json",
                                   "decoder_post_cross_attn_residual"),
        ("qwen3_asr_1p7b", "encoder"): (REPO / "results/basis_a4/qwen3_asr_1p7b/raw/cs_dialogue/L16/Raw_encoder_L16_oracle_local_rho0.5.json",
                                           "qwen_audio_encoder_post_self_attn_residual_pre_ffn"),
        ("qwen3_asr_1p7b", "decoder"): (REPO / "results/basis_a4/qwen3_asr_1p7b/conditioning/cs_dialogue/L16/Conditioning_decoder_L16_oracle_local_rho0.5.json",
                                           "qwen_text_decoder_post_self_attn_residual_pre_mlp"),
    }
    rows, failures = [], []
    for (model, side), (path, site) in expected.items():
        if not path.is_file():
            failures.append(f"missing {path}"); continue
        value = json.loads(path.read_text())
        checks = {"scope_oracle_local": value.get("scope") == "oracle_local",
                  "site_exact": value.get("site") == site,
                  "site_hash_present": bool(value.get("site_hash")),
                  "mask_hash_present": bool(value.get("mask_hash")),
                  "no_global": value.get("scope") != "global"}
        if not all(checks.values()): failures.append(f"{model}/{side}: {checks}")
        rows.append({"model": model, "side": side, "source_artifact": str(path.relative_to(REPO)),
                     "site": value.get("site"), "site_hash": value.get("site_hash"),
                     "mask_hash": value.get("mask_hash"), "checks": checks,
                     "qwen_repaired_mask_evidence": model != "qwen3_asr_1p7b" or "results/basis_a4/acceptance/qwen_acceptance.json"})
    payload = {"schema_version": "basis_a6_local_mask_acceptance_v1", "status": "PASS" if not failures else "FAIL",
               "scope": "oracle_local_only", "rows": rows, "failures": failures,
               "qwen_acceptance": "results/basis_a4/acceptance/qwen_acceptance.json",
               "qwen_local_trace": "results/basis_a4/acceptance/qwen_local_mask_traces.json"}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
