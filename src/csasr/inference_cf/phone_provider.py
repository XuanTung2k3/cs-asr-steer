"""DIR-SPRINT0 D5 frozen phonetic provider adapter (provider engineering; no steering, no lexical evaluation).

Provider: facebook/wav2vec2-xlsr-53-espeak-cv-ft at revision 2c733782 (Apache-2.0), a wav2vec 2.0 CTC phone
recognizer over a 392-symbol espeak-ng phone inventory. It is used frozen: ``eval()``, float32, no gradient, no
training or fine-tuning. Phone symbols map to articulatory features through the frozen panphon-derived table
``results/inference_cf/dir_sprint0/d5_provider/feature_table.json``. panphon itself is not needed at runtime.

Interface (all acoustic; no text argument anywhere):

* ``PhoneProvider.posteriors(waveform, sample_rate)``: frame-indexed float32 log-posteriors ``[T, 392]`` over the
  phone inventory (CTC blank = id 0) for one mono 16 kHz waveform, with sample spans, hashes, status and provenance.
* ``frame_spans`` / ``frames_in_interval`` / ``interval_weights`` / ``encoder_frame_weights``: provider frame j covers
  samples ``[320 j, 320 j + 400)`` (conv stride 320, receptive field 400). The frozen R2 localizer uses 320-sample
  encoder frames ``[320 f, 320 f + 320)`` and windows given as ``[start_sample, end_sample)``. These helpers map either
  onto provider frames by exact sample overlap.
* ``FeatureTable`` / ``feature_evidence``: weighted phone-feature evidence over chosen frames. The blank, special,
  tone-only and excluded masses are reported separately and never redistributed. With no frames or zero valid phone
  mass the result is a deterministic ``None`` vector with an explicit status.

No reference transcript, phonemizer, gold timing, alignment, language label or lexical outcome is accepted, imported
or produced here. Text phonemization is deliberately absent; the provider's own tokenizer class (which would load a
phonemizer) is not used.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

VERSION = "dir_sprint0_d5_phone_provider_v1"
SAMPLE_RATE = 16000
FRAME_STRIDE = 320
RECEPTIVE_FIELD = 400
ENCODER_FRAME = 320
STATUS_KINDS = ("segmental", "blank", "special", "tone_only", "excluded")


class ProviderError(RuntimeError):
    """Critical provider integrity failure (hash mismatch, malformed table, wrong model shape)."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def array_sha256(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return "sha256:" + hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def canonical_digest(obj) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                                 separators=(",", ":")).encode()).hexdigest()


# ---- frame geometry -----------------------------------------------------------------------------------------------

def n_frames(n_samples: int) -> int:
    n = int(n_samples)
    return 0 if n < RECEPTIVE_FIELD else (n - RECEPTIVE_FIELD) // FRAME_STRIDE + 1


def frame_spans(count: int) -> np.ndarray:
    """[count, 2] int64 half-open sample spans [320 j, 320 j + 400)."""
    j = np.arange(int(count), dtype=np.int64)
    return np.stack([j * FRAME_STRIDE, j * FRAME_STRIDE + RECEPTIVE_FIELD], axis=1)


def frames_in_interval(start_sample: int, end_sample: int, count: int) -> np.ndarray:
    """Provider frames whose span centre (320 j + 200) lies in [start_sample, end_sample)."""
    centers = frame_spans(count).mean(axis=1)
    return np.flatnonzero((centers >= int(start_sample)) & (centers < int(end_sample)))


def interval_weights(start_sample: int, end_sample: int, count: int) -> np.ndarray:
    """Fraction of each provider frame's 400-sample span inside [start_sample, end_sample) (float64)."""
    sp = frame_spans(count)
    ov = np.clip(np.minimum(sp[:, 1], int(end_sample)) - np.maximum(sp[:, 0], int(start_sample)), 0, None)
    return ov.astype(np.float64) / RECEPTIVE_FIELD


def encoder_frame_weights(encoder_weights: Sequence[float], count: int) -> np.ndarray:
    """Spread a per-encoder-frame weight vector (frame f = samples [320 f, 320 f + 320)) onto provider frames by exact
    sample overlap: w_j = sum_f a_f * |E_f intersect P_j| / 320, i.e. the encoder weight (as a uniform per-sample
    density) integrated over provider span P_j = [320 j, 320 j + 400). Provider spans overlap by 80 samples, so an
    interior encoder frame contributes 400/320 = 1.25 times its weight in total. No normalization is applied here."""
    a = np.asarray(encoder_weights, dtype=np.float64)
    if a.ndim != 1 or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("encoder weights must be a finite nonnegative vector")
    sp = frame_spans(count)
    f = np.arange(a.size, dtype=np.int64)
    e0, e1 = f * ENCODER_FRAME, (f + 1) * ENCODER_FRAME
    ov = np.clip(np.minimum(sp[:, None, 1], e1[None, :]) - np.maximum(sp[:, None, 0], e0[None, :]), 0, None)
    return (ov.astype(np.float64) / ENCODER_FRAME) @ a


# ---- frozen feature table -----------------------------------------------------------------------------------------

class FeatureTable:
    """Frozen symbol -> status / tone / articulatory-feature table (verified against its own digest)."""

    def __init__(self, path: Path, expected_digest: str | None = None):
        self.path = Path(path)
        t = json.loads(self.path.read_text(encoding="utf-8"))
        digest = canonical_digest_table(t)
        if t.get("table_digest") != digest:
            raise ProviderError("feature table digest mismatch")
        if expected_digest is not None and digest != expected_digest:
            raise ProviderError("feature table is not the manifest-pinned table")
        rows = t["rows"]
        if [r["id"] for r in rows] != list(range(len(rows))):
            raise ProviderError("feature table ids are not contiguous")
        self.digest, self.file_sha256 = digest, file_sha256(self.path)
        self.names = list(t["feature_names"])
        self.symbols = [r["symbol"] for r in rows]
        self.status = np.array([r["status"] for r in rows])
        if not set(self.status.tolist()) <= set(STATUS_KINDS):
            raise ProviderError("unknown symbol status")
        self.valid = self.status == "segmental"
        if self.valid.tolist() != list(t["valid_mask"]):
            raise ProviderError("valid mask disagrees with row status")
        self.blank_id = int(t["blank_id"])
        if self.status[self.blank_id] != "blank":
            raise ProviderError("blank id is not the blank row")
        self.tone = np.array([0 if r["tone"] is None else int(r["tone"]) for r in rows], dtype=np.int64)
        F = np.zeros((len(rows), len(self.names)), dtype=np.float64)
        for r in rows:
            if r["status"] == "segmental":
                if r["features"] is None or len(r["features"]) != len(self.names):
                    raise ProviderError(f"segmental row {r['id']} lacks features")
                F[r["id"]] = r["features"]
            elif r["features"] is not None:
                raise ProviderError(f"non-segmental row {r['id']} carries features")
        if not np.all(np.isfinite(F)) or np.any(np.abs(F) > 1):
            raise ProviderError("feature values outside [-1, 1]")
        self.F = F

    def __len__(self) -> int:
        return len(self.symbols)


def canonical_digest_table(t: dict) -> str:
    return "sha256:" + hashlib.sha256(json.dumps({k: t[k] for k in ("feature_names", "valid_mask", "rows")},
                                                 sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def feature_evidence(log_probs: np.ndarray, weights: Sequence[float], table: FeatureTable) -> dict:
    """Weighted phone-feature evidence over frames.

    p = exp(log_probs) in float64. Valid mass M = sum_j w_j sum_{v segmental} p_jv. q_k = sum_j w_j sum_{v segmental}
    p_jv F_vk / M. Blank / special / tone-only / excluded masses are reported separately and never enter q. With no
    positive weight the status is ``no_frames``; with M == 0 it is ``zero_valid_mass``; q is then None.
    """
    lp = np.asarray(log_probs)
    w = np.asarray(weights, dtype=np.float64)
    if lp.ndim != 2 or lp.shape[1] != len(table) or w.shape != (lp.shape[0],):
        raise ValueError("log_probs must be [T, V] and weights [T]")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and nonnegative")
    out = {"status": None, "q": None, "feature_names": table.names, "weight_total": float(w.sum()),
           "frames_used": int(np.count_nonzero(w)), "mass": None, "valid_phone_mass": None, "tone_mass": None}
    if not w.sum() > 0:
        out["status"] = "no_frames"
        return out
    if not np.all(np.isfinite(lp)):
        out["status"] = "nonfinite_posteriors"
        return out
    p = np.exp(lp.astype(np.float64))
    per_symbol = w @ p                                        # [V] weighted symbol mass
    mass = {k: float(per_symbol[table.status == k].sum()) for k in STATUS_KINDS}
    out["mass"] = mass
    out["valid_phone_mass"] = np.where(table.valid, per_symbol, 0.0)
    out["tone_mass"] = {str(k): float(per_symbol[table.valid & (table.tone == k)].sum()) for k in range(6)}
    if not mass["segmental"] > 0:
        out["status"] = "zero_valid_mass"
        return out
    out["q"] = (out["valid_phone_mass"] @ table.F) / mass["segmental"]
    out["status"] = "ok"
    return out


# ---- provider -----------------------------------------------------------------------------------------------------

def load_manifest(path: Path) -> dict:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in m.items() if k != "manifest_digest"}
    if m.get("manifest_digest") != canonical_digest(body):
        raise ProviderError("provider manifest digest mismatch")
    return m


class PhoneProvider:
    """Frozen wav2vec2 espeak phone recognizer behind a hash-verified, text-free interface."""

    def __init__(self, manifest_path: Path, table_path: Path, *, device: str = "cpu", verify_weights: bool = True):
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        self.manifest = load_manifest(manifest_path)
        self.model_dir = Path(self.manifest["model"]["dir"])
        for name, expected in self.manifest["model"]["files"].items():
            if name == "pytorch_model.bin" and not verify_weights:
                continue
            if file_sha256(self.model_dir / name) != expected:
                raise ProviderError(f"provider file hash mismatch: {name}")
        self.table = FeatureTable(table_path, expected_digest=self.manifest["feature_table"]["table_digest"])
        self.device = torch.device(device)
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_dir, dtype=torch.float32).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if self.model.config.vocab_size != len(self.table) or self.model.config.pad_token_id != self.table.blank_id:
            raise ProviderError("model vocabulary does not match the feature table")
        if int(self.extractor.sampling_rate) != SAMPLE_RATE:
            raise ProviderError("feature extractor sampling rate is not 16 kHz")
        self.provenance = {"adapter_version": VERSION, "manifest_digest": self.manifest["manifest_digest"],
                           "model_revision": self.manifest["model"]["revision"],
                           "weights_sha256": self.manifest["model"]["files"]["pytorch_model.bin"],
                           "weights_verified": bool(verify_weights), "table_digest": self.table.digest,
                           "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                           "device": str(self.device), "dtype": "float32"}

    def posteriors(self, waveform, sample_rate: int) -> dict:
        """Frame log-posteriors for one mono 16 kHz waveform; invalid input returns a status, never a guess."""
        import torch

        out = {"status": None, "reason": None, "log_probs": None, "n_samples": None, "n_frames": None,
               "input_sha256": None, "log_probs_sha256": None, "provenance": self.provenance}
        x = np.asarray(waveform)
        if int(sample_rate) != SAMPLE_RATE:
            out.update(status="invalid_input", reason=f"sample_rate {sample_rate} != 16000 (no internal resampling)")
            return out
        if x.ndim != 1:
            out.update(status="invalid_input", reason=f"expected mono 1-D waveform, got shape {x.shape}")
            return out
        if x.dtype not in (np.float32, np.float64):
            out.update(status="invalid_input", reason=f"expected float waveform, got {x.dtype}")
            return out
        x = x.astype(np.float32, copy=False)
        out["n_samples"], out["input_sha256"] = int(x.size), array_sha256(x)
        if not np.all(np.isfinite(x)):
            out.update(status="invalid_input", reason="nonfinite samples")
            return out
        if x.size < RECEPTIVE_FIELD:
            out.update(status="too_short", reason=f"{x.size} samples < receptive field {RECEPTIVE_FIELD}")
            return out
        feats = self.extractor(x, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True)
        tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            with torch.inference_mode():
                logits = self.model(feats.input_values.to(self.device),
                                    attention_mask=feats.attention_mask.to(self.device)).logits[0].float()
                lp = torch.log_softmax(logits, dim=-1).cpu().numpy()
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        if lp.shape != (n_frames(x.size), len(self.table)):
            raise ProviderError(f"unexpected posterior shape {lp.shape} for {x.size} samples")
        if not np.all(np.isfinite(lp)):
            out.update(status="nonfinite_output", reason="nonfinite log-posteriors", n_frames=int(lp.shape[0]))
            return out
        out.update(status="ok", log_probs=lp, n_frames=int(lp.shape[0]), log_probs_sha256=array_sha256(lp))
        return out


def greedy_phones(log_probs: np.ndarray, table: FeatureTable) -> list[str]:
    """Collapsed CTC argmax symbol sequence (engineering display only; never a feature input)."""
    ids = np.argmax(np.asarray(log_probs), axis=1).tolist()
    out = []
    for j, i in enumerate(ids):
        if i != table.blank_id and (j == 0 or ids[j - 1] != i):
            out.append(table.symbols[i])
    return out
