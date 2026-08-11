"""Measured throughput and storage, so later stages are sized from evidence.

Plan v1 asserted that generating intervention-utility labels would be the
dominant compute item. Measured free decoding on this cluster is 6.0-6.5 utt/s
at batch 16, which puts label generation at well under an hour and makes
alignment the real schedule risk -- but the alignment families' throughput had
never been measured at all. Both are measured here before either is budgeted.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class PilotResult:
    mode: str
    n: int
    seconds: float
    rate: float                      # items per second
    peak_gb: float = float("nan")
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _peak_gb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return float("nan")
        return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
    except Exception:                                          # pragma: no cover
        return float("nan")


def _reset_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:                                          # pragma: no cover
        pass


def measure(mode: str, n: int, fn: Callable[[], Any], *,
            detail: dict[str, Any] | None = None) -> PilotResult:
    """Time one measurement, recording peak GPU memory for it alone."""
    _reset_peak()
    start = time.monotonic()
    extra = fn()
    seconds = time.monotonic() - start
    result = PilotResult(
        mode=mode, n=int(n), seconds=float(seconds),
        rate=(n / seconds) if seconds > 0 else float("nan"),
        peak_gb=_peak_gb(),
        detail={**(detail or {}), **(extra if isinstance(extra, dict) else {})},
    )
    log.info("pilot %s: %d items in %.1f s (%.2f/s, peak %.2f GB)",
             mode, result.n, result.seconds, result.rate, result.peak_gb)
    return result


def pilot_free_decode(bundle, manifest: pd.DataFrame, cfg: Mapping[str, Any],
                      batch_sizes: Sequence[int] = (8, 16)) -> list[PilotResult]:
    """Free decoding, the cost that dominates second-pass and label generation."""
    from ..models.generation import decode_batch

    results = []
    for bs in batch_sizes:
        n_batches = math.ceil(len(manifest) / bs)

        def run(bs=bs, n_batches=n_batches):
            for bi in range(n_batches):
                batch = manifest.iloc[bi * bs: (bi + 1) * bs]
                decode_batch(bundle, batch["audio_path"].tolist(), dict(cfg),
                             language=None)
            return {"batch_size": bs}

        results.append(measure(f"free_decode_bs{bs}", len(manifest), run))
    return results


def pilot_encoder_only(bundle, manifest: pd.DataFrame,
                       batch_size: int = 8) -> PilotResult:
    """Encoder forward alone: the cost of building the activation cache."""
    import torch

    from ..models.whisper import batch_features

    n_batches = math.ceil(len(manifest) / batch_size)

    def run():
        with torch.inference_mode():
            for bi in range(n_batches):
                batch = manifest.iloc[bi * batch_size: (bi + 1) * batch_size]
                features = batch_features(bundle, batch["audio_path"].tolist())
                bundle.model.model.encoder(features)
        return {"batch_size": batch_size}

    return measure("encoder_only", len(manifest), run)


def pilot_teacher_forced(bundle, manifest: pd.DataFrame, layers: Sequence[int],
                         batch_size: int = 8) -> PilotResult:
    """Teacher-forced forward with recorders: direction construction's cost."""
    import torch

    from ..models.generation import teacher_forced_forward
    from ..models.hooks import ActivationRecorder

    tok = bundle.processor.tokenizer
    from ..data.alignment import build_prefix
    from ..data.normalize import normalize_text

    prefix = build_prefix(bundle.processor)
    n_batches = math.ceil(len(manifest) / batch_size)

    def run():
        with torch.inference_mode():
            for bi in range(n_batches):
                batch = manifest.iloc[bi * batch_size: (bi + 1) * batch_size]
                seqs = [prefix + tok.encode(normalize_text(t), add_special_tokens=False)[:200]
                        for t in batch["transcript_raw"]]
                with ActivationRecorder(bundle, list(layers), module="encoder"):
                    teacher_forced_forward(bundle, batch["audio_path"].tolist(), seqs)
        return {"batch_size": batch_size, "layers": list(layers)}

    return measure("teacher_forced_encoder_cache", len(manifest), run)


def pilot_decoder_site_capture(bundle, manifest: pd.DataFrame,
                               layers: Sequence[int],
                               batch_size: int = 8) -> PilotResult:
    """Overhead of the dual-hook post-cross-attention capture."""
    import torch

    from ..data.alignment import build_prefix
    from ..data.normalize import normalize_text
    from ..models.generation import teacher_forced_forward
    from .sites import DecoderPostCrossAttnRecorder

    tok = bundle.processor.tokenizer
    prefix = build_prefix(bundle.processor)
    n_batches = math.ceil(len(manifest) / batch_size)

    def run():
        with torch.inference_mode():
            for bi in range(n_batches):
                batch = manifest.iloc[bi * batch_size: (bi + 1) * batch_size]
                seqs = [prefix + tok.encode(normalize_text(t), add_special_tokens=False)[:200]
                        for t in batch["transcript_raw"]]
                with DecoderPostCrossAttnRecorder(bundle, list(layers),
                                                  keep_attention=True):
                    teacher_forced_forward(bundle, batch["audio_path"].tolist(), seqs)
        return {"batch_size": batch_size, "layers": list(layers)}

    return measure("decoder_site_capture", len(manifest), run)


def pilot_steered_decode(bundle, manifest: pd.DataFrame, cfg: Mapping[str, Any],
                         layer: int, batch_size: int = 16) -> PilotResult:
    """Free decoding with an encoder steering hook: the second-pass cost."""
    import numpy as np
    import torch

    from ..models.generation import decode_batch
    from ..steering.encoder_hook import EncoderLocalSteering
    from ..steering.masks import MaskSpec

    direction = torch.zeros(bundle.d_model)
    direction[0] = 1.0
    n_batches = math.ceil(len(manifest) / batch_size)

    def run():
        for bi in range(n_batches):
            batch = manifest.iloc[bi * batch_size: (bi + 1) * batch_size]
            spans = {}
            for _, row in batch.iterrows():
                n_valid = bundle.valid_frames(row["duration_sec"])
                mid = max(1, n_valid // 2)
                spans[row["utterance_id"]] = (max(0, mid - 10), min(n_valid, mid + 10))
            builder = EncoderLocalSteering(
                bundle, layer, direction, alpha=0.0, scale=1.0,
                spec=MaskSpec(kind="EXACT_TAPER", shoulder_frames=5), spans=spans)
            decode_batch(bundle, batch["audio_path"].tolist(), dict(cfg),
                         language=None, hooks=list(builder(batch)))
        return {"batch_size": batch_size, "layer": int(layer)}

    return measure("steered_decode", len(manifest), run)


def pilot_aligners(manifest: pd.DataFrame, cfg: Mapping[str, Any], geometry,
                   identity, families: Sequence[str], bundle=None) -> list[PilotResult]:
    """Per-family alignment throughput -- the number plan v1 never measured."""
    from ..nat5h.aligners import run_existing_ctc, run_whisper_dtw

    results: list[PilotResult] = []
    for family in families:
        if family == "existing_ctc":
            def run():
                table = run_existing_ctc(manifest, dict(cfg), geometry, identity)
                return {"rows": int(len(table))}
        elif family == "whisper_dtw":
            if bundle is None:
                continue

            def run():
                table, meta = run_whisper_dtw(bundle, manifest, dict(cfg), identity)
                return {"rows": int(len(table)), "mapping": meta.get("unit_coverage")}
        else:
            continue
        try:
            results.append(measure(f"align_{family}", len(manifest), run))
        except Exception as exc:                               # diagnostics only
            log.warning("aligner pilot %s failed: %s", family, exc)
            results.append(PilotResult(mode=f"align_{family}", n=0, seconds=0.0,
                                       rate=float("nan"),
                                       detail={"error": repr(exc)}))
    return results


def pilot_storage(bundle, manifest: pd.DataFrame, layers: Sequence[int]) -> dict[str, Any]:
    """Bytes per utterance for the activation cache, from real durations."""
    step = float(bundle.encoder_step_sec)
    d_model = int(bundle.d_model)
    frames = [bundle.valid_frames(d) for d in manifest["duration_sec"]]
    mean_frames = float(sum(frames)) / max(1, len(frames))
    per_layer_fp16 = mean_frames * d_model * 2
    return {
        "mean_valid_frames": mean_frames,
        "encoder_step_sec": step,
        "d_model": d_model,
        "encoder_state_bytes_per_utt_per_layer_fp16": int(per_layer_fp16),
        "encoder_state_bytes_per_utt_fp16": int(per_layer_fp16 * len(layers)),
        "layers": list(layers),
    }


def extrapolate(results: Sequence[PilotResult], storage: Mapping[str, Any],
                workload: Mapping[str, int]) -> dict[str, Any]:
    """Project per-stage GPU hours and storage from the measured rates."""
    rates = {r.mode: r.rate for r in results if r.rate and r.rate == r.rate}
    decode_rate = max((v for k, v in rates.items() if k.startswith("free_decode")),
                      default=float("nan"))
    steered_rate = rates.get("steered_decode", decode_rate)
    align_rates = {k: v for k, v in rates.items() if k.startswith("align_")}
    align_rate = min(align_rates.values()) if align_rates else float("nan")

    def hours(count: int, rate: float) -> float:
        return float(count) / rate / 3600.0 if rate and rate == rate else float("nan")

    n_align = int(workload.get("alignment_utterances", 0))
    n_families = int(workload.get("alignment_families", 2))
    n_labels = int(workload.get("utility_candidates", 0))
    n_prompts = int(workload.get("prompt_decodes", 0))

    projected = {
        "decode_rate_utt_per_s": decode_rate,
        "steered_decode_rate_utt_per_s": steered_rate,
        "alignment_rate_utt_per_s": align_rate,
        "l1_alignment_gpu_hours": hours(n_align * n_families, align_rate),
        "l2a_prompt_gpu_hours": hours(n_prompts, decode_rate),
        "l7a_label_gpu_hours": hours(n_labels, steered_rate),
        "storage_gb": (storage.get("encoder_state_bytes_per_utt_fp16", 0)
                       * int(workload.get("cached_utterances", 0)) / 1024 ** 3),
        "workload": dict(workload),
    }
    return projected


def results_frame(results: Sequence[PilotResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in results])
