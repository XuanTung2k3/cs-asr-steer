"""Frozen Whisper-large-v3 loading, module inspection and frame geometry.

Nothing in this package ever unfreezes a parameter or computes a gradient:
every entry point runs under ``torch.inference_mode()``.
"""
from __future__ import annotations
import hashlib

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..utils.logging import get_logger

log = get_logger(__name__)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class WhisperBundle:
    """Model + processor + everything derived from the actual config."""

    model: Any
    processor: Any
    config: Any
    device: str
    dtype: torch.dtype
    model_id: str
    revision: str | None
    encoder_step_sec: float
    max_encoder_frames: int
    num_encoder_layers: int
    num_decoder_layers: int
    d_model: int
    sample_rate: int
    chunk_sec: float
    module_report: dict = field(default_factory=dict)

    tokenizer_revision: str | None = None
    # ---- frame geometry -------------------------------------------------
    def sec_to_frames(self, start_sec: float, end_sec: float,
                      num_valid: int | None = None) -> tuple[int, int]:
        """Half-open [start_frame, end_frame) with clamping (guide section 12)."""
        limit = self.max_encoder_frames if num_valid is None else num_valid
        start = int(np.floor(start_sec / self.encoder_step_sec))
        end = int(np.ceil(end_sec / self.encoder_step_sec))
        start = max(0, min(start, limit - 1))
        end = max(start + 1, min(end, limit))
        return start, end

    def valid_frames(self, duration_sec: float) -> int:
        """Number of non-padding encoder frames for an utterance."""
        n = int(np.ceil(float(duration_sec) / self.encoder_step_sec))
        return int(max(1, min(n, self.max_encoder_frames)))

    def encoder_layer(self, index: int):
        return self.model.model.encoder.layers[index]

    def decoder_layer(self, index: int):
        return self.model.model.decoder.layers[index]

    def metadata(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_revision": self.revision,
            "tokenizer_revision": self.tokenizer_revision,
            "dtype": str(self.dtype),
            "device": self.device,
            "encoder_step_sec": self.encoder_step_sec,
            "max_encoder_frames": self.max_encoder_frames,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "d_model": self.d_model,
            "sample_rate": self.sample_rate,
            "chunk_sec": self.chunk_sec,
            "modules": self.module_report,
        }


def _file_revision(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _resolve_revision(model_path: str) -> str | None:
    """Content-address the exact local weights used by from_pretrained."""
    p = Path(model_path)
    for name in ("model.safetensors", "pytorch_model.bin"):
        revision = _file_revision(p / name)
        if revision:
            return revision
    return None


def inspect_modules(model) -> dict:
    """Record the exact hook targets (guide section 18: never hook blindly)."""
    enc0 = model.model.encoder.layers[0]
    dec0 = model.model.decoder.layers[0]
    return {
        "encoder_layer_module_path": "model.model.encoder.layers[l]",
        "encoder_layer_class": type(enc0).__name__,
        "encoder_hook_location": "residual stream output of encoder block l (before block l+1)",
        "encoder_output_is_tuple": True,
        "encoder_layer_indexing": "zero-indexed",
        "decoder_layer_module_path": "model.model.decoder.layers[k]",
        "decoder_layer_class": type(dec0).__name__,
        "decoder_hook_location": "residual stream output of decoder block k",
        "encoder_final_layer_norm": "model.model.encoder.layer_norm (applied after last block)",
        "num_encoder_layers": len(model.model.encoder.layers),
        "num_decoder_layers": len(model.model.decoder.layers),
    }


def load_whisper(cfg: dict) -> WhisperBundle:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    mcfg = cfg["model"]
    model_id = mcfg["id"]
    dtype = DTYPES[str(mcfg.get("dtype", "bfloat16"))]
    device = mcfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU (float32)")
        device, dtype = "cpu", torch.float32

    processor = WhisperProcessor.from_pretrained(
        model_id, local_files_only=mcfg.get("local_files_only", True)
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        local_files_only=mcfg.get("local_files_only", True),
        attn_implementation=mcfg.get("attn_implementation", "eager"),
    )
    model.to(device)
    model.eval()
    if mcfg.get("freeze_backbone", True):
        for p in model.parameters():
            p.requires_grad_(False)

    config = model.config
    fe = processor.feature_extractor
    sample_rate = int(fe.sampling_rate)
    hop = int(fe.hop_length)
    chunk_sec = float(fe.chunk_length)
    # Whisper's two conv layers downsample the mel sequence by 2.
    conv_stride = 2
    encoder_step_sec = hop / sample_rate * conv_stride
    max_frames = int(config.max_source_positions)
    derived = chunk_sec / max_frames
    if abs(derived - encoder_step_sec) > 1e-9:
        log.warning("encoder step mismatch: derived %.6f vs config %.6f", encoder_step_sec, derived)

    bundle = WhisperBundle(
        model=model,
        processor=processor,
        config=config,
        device=device,
        dtype=dtype,
        model_id=mcfg.get("hub_id", model_id),
        revision=mcfg.get("revision") or _resolve_revision(model_id),
        tokenizer_revision=_file_revision(Path(model_id) / "tokenizer.json"),
        encoder_step_sec=encoder_step_sec,
        max_encoder_frames=max_frames,
        num_encoder_layers=len(model.model.encoder.layers),
        num_decoder_layers=len(model.model.decoder.layers),
        d_model=int(config.d_model),
        sample_rate=sample_rate,
        chunk_sec=chunk_sec,
        module_report=inspect_modules(model),
    )
    log.info(
        "loaded %s (%s) on %s: %d enc / %d dec layers, d_model=%d, step=%.3f s, max_frames=%d",
        bundle.model_id, bundle.dtype, bundle.device, bundle.num_encoder_layers,
        bundle.num_decoder_layers, bundle.d_model, bundle.encoder_step_sec,
        bundle.max_encoder_frames,
    )
    return bundle


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import scipy.signal as sps

        n = int(round(len(audio) * target_sr / sr))
        audio = sps.resample(audio, n).astype("float32")
    return audio


_AUDIO_POOL: Any = None


def _audio_pool():
    """Lazily-built thread pool for decoding audio files concurrently.

    The dataset lives on NFS and ``soundfile`` releases the GIL while reading,
    so loading a batch serially leaves the GPU idle on network latency. Threads
    (not processes) keep this a drop-in change with no serialisation cost.
    """
    global _AUDIO_POOL
    if _AUDIO_POOL is None:
        import os
        from concurrent.futures import ThreadPoolExecutor

        workers = int(os.environ.get("CSASR_AUDIO_WORKERS",
                                     os.environ.get("OMP_NUM_THREADS", "8")))
        _AUDIO_POOL = ThreadPoolExecutor(max_workers=max(1, workers),
                                         thread_name_prefix="csasr-audio")
    return _AUDIO_POOL


def batch_features(bundle: WhisperBundle, audio_paths: Sequence[str]) -> torch.Tensor:
    """Log-mel features for a batch, padded/truncated to the 30 s window."""
    return batch_model_inputs(bundle, audio_paths)["input_features"]


def batch_model_inputs(bundle: WhisperBundle,
                       audio_paths: Sequence[str]) -> dict[str, torch.Tensor]:
    """Whisper features plus the real padding mask produced by the processor.

    Whisper uses the same id for padding and EOS, so generation cannot infer an
    attention mask from ids.  The feature extractor knows the unpadded waveform
    lengths and is the authoritative place to create it.
    """
    paths = list(audio_paths)
    if len(paths) > 1:
        # executor.map preserves input order, so the batch order is unchanged
        audios = list(_audio_pool().map(
            lambda p: load_audio(p, bundle.sample_rate), paths))
    else:
        audios = [load_audio(p, bundle.sample_rate) for p in paths]
    feats = bundle.processor.feature_extractor(
        audios, sampling_rate=bundle.sample_rate, return_tensors="pt",
        return_attention_mask=True,
    )
    if not hasattr(feats, "attention_mask"):
        raise RuntimeError("Whisper feature extractor returned no attention_mask")
    return {
        "input_features": feats.input_features.to(bundle.device, bundle.dtype),
        "attention_mask": feats.attention_mask.to(bundle.device),
    }


def write_model_metadata(bundle: WhisperBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle.metadata(), indent=2), encoding="utf-8")
