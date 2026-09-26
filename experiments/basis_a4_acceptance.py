#!/usr/bin/env python
"""CPU acceptance gate for the additive BASIS-A4 implementation."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/basis_a4"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def main():
    freeze = json.loads((OUT / "manifests/a4_protocol_freeze.json").read_text())
    reuse = json.loads((OUT / "manifests/whisper_reuse_manifest.json").read_text())
    qconfig = json.loads(Path("/mnt/data/tungnx/Qwen3-ASR-1.7B/config.json").read_text())
    nested = qconfig["thinker_config"]
    checks = {
        "protocol_frozen": freeze["status"] == "FROZEN_PRE_RUN",
        "directions_are_raw_conditioning_only": freeze["directions"] == ["Raw", "Conditioning"],
        "conditioning_avg_not_materialized": not list(OUT.glob("qwen/**/*conditioning_avg*")),
        "whisper_reuse_count": len(reuse["accepted"]) == 408,
        "qwen_official_model_type": nested["model_type"] == "qwen3_asr",
        "qwen_architecture": nested["architectures"] == ["Qwen3ASRForConditionalGeneration"],
        "qwen_dimensions": (nested["audio_config"]["d_model"] == 1024 and
                             nested["audio_config"]["num_hidden_layers"] == 24 and
                             nested["text_config"]["hidden_size"] == 2048 and
                             nested["text_config"]["num_hidden_layers"] == 28),
        "qwen_sites_frozen": (freeze["qwen"]["encoder_site"] == "qwen_audio_encoder_post_self_attn_residual_pre_ffn" and
                              freeze["qwen"]["decoder_site"] == "qwen_text_decoder_post_self_attn_residual_pre_mlp"),
    }
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_basis_a4_qwen.py",
           "tests/test_lss_sites.py", "tests/test_dg02_site.py"]
    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
    checks["focused_cpu_tests"] = proc.returncode == 0
    report = {"schema_version": "basis_a4_acceptance_v1",
              "status": "PASS" if all(checks.values()) else "FAIL",
              "checks": checks, "pytest_command": cmd,
              "pytest_stdout": proc.stdout[-12000:], "pytest_stderr": proc.stderr[-4000:],
              "required_before_gpu": True}
    write(OUT / "manifests/acceptance_cpu.json", report)
    if report["status"] != "PASS":
        raise SystemExit(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
