"""Two-track steering study harness for Mandarin-English CS-ASR.

Track A  inference-only steering sweep on a frozen Whisper-large-v3.
Track B  learned interventions (LoReFT-style, LoRA, reimplemented AGA).

Everything here is DEVELOPMENT-ONLY DIAGNOSTIC. Nothing in this package
evaluates production Gate A, freezes spans, allocates a held-out gate
generation, or opens `D-dev-confirm` / `D-test`.

The package reuses the existing repository stack rather than reimplementing it:

  csasr.models.whisper      model loading, frame geometry, batched features
  csasr.models.generation   deterministic decoding, teacher forcing
  csasr.evaluation.pier     PIER, per-unit status, POI taxonomy
  csasr.evaluation.mer      MER / WER
  csasr.evaluation.bootstrap cluster bootstrap
  csasr.experiments.v2r3_*  the frozen v2r3 target construction
"""
from __future__ import annotations

__all__ = ["config"]
TAINT = "development_only_diagnostic"
