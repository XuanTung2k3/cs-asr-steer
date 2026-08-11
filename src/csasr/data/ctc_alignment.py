"""Independent CTC forced alignment, used only to audit the DTW alignment.

Why this exists
---------------
E1's boundaries come from Whisper cross-attention DTW, the guide's *third*
priority source. Its reliability was previously checked only against a second
DTW configuration, which shares every failure mode of the first -- both absorb
leading silence identically and then agree. That is a consistency proxy, not an
accuracy measurement.

A CTC model infers boundaries from frame-local acoustic posteriors of an
encoder-only network. Its failure modes are largely disjoint from an
autoregressive decoder's attention artefacts, so agreement between the two is
genuine evidence.

Scope
-----
This module never supplies the spans used by E2/E4/E5. DTW remains primary;
CTC only measures it. Promoting CTC to primary would be a pre-registration
decision, taken before any E4 result is seen.

Frame geometry
--------------
wav2vec2-style encoders stride 320 samples at 16 kHz -> 20 ms per frame, exactly
matching Whisper's encoder step (hop 160 x conv stride 2). Frame indices here and
in the alignment tables therefore refer to the same instants, with no rescaling.

Availability
------------
Everything degrades gracefully: if the model directory is absent, ``load_ctc_aligner``
raises ``CTCUnavailable`` and callers record that the check was skipped rather
than failing the stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..utils.logging import get_logger

log = get_logger(__name__)

FRAME_SEC = 0.02


class CTCUnavailable(RuntimeError):
    """The CTC aligner model or its romanizer is not installed."""


@dataclass
class CTCAligner:
    model: object
    processor: object
    device: str
    blank_id: int
    frame_sec: float = FRAME_SEC

    def vocab(self) -> dict:
        return self.processor.tokenizer.get_vocab()


def load_ctc_aligner(cfg: dict) -> CTCAligner:
    """Load the CTC acoustic model used for the independent alignment check."""
    ccfg = (cfg.get("alignment", {}) or {}).get("ctc", {}) or {}
    model_id = ccfg.get("model_id")
    if not model_id:
        raise CTCUnavailable("alignment.ctc.model_id is not configured")
    if not Path(model_id).exists() and "/" not in str(model_id):
        raise CTCUnavailable(f"CTC model path does not exist: {model_id}")
    try:
        from transformers import AutoModelForCTC, AutoProcessor
    except Exception as exc:                                   # pragma: no cover
        raise CTCUnavailable(f"transformers CTC classes unavailable: {exc}") from exc

    try:
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCTC.from_pretrained(model_id, local_files_only=True)
    except Exception as exc:
        raise CTCUnavailable(f"could not load CTC aligner {model_id}: {exc}") from exc

    device = cfg.get("model", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blank_id = int(getattr(model.config, "pad_token_id", 0) or 0)
    log.info("loaded CTC aligner %s on %s (blank id %d)", model_id, device, blank_id)
    return CTCAligner(model=model, processor=processor, device=device, blank_id=blank_id)


def romanize(text: str) -> str:
    """Map any script to Latin so one vocabulary covers Mandarin and English.

    Mixed-script utterances are the whole problem: a Mandarin CTC vocabulary has
    no Latin letters and an English one has no Han characters. Romanizing both
    sides collapses them into a single alphabet, which is how the MMS aligner
    handles arbitrary languages.
    """
    try:
        import uroman as ur
    except Exception as exc:                                   # pragma: no cover
        raise CTCUnavailable(
            "the `uroman` package is required to align mixed-script text") from exc
    return ur.Uroman().romanize_string(str(text))


@torch.inference_mode()
def ctc_log_posteriors(aligner: CTCAligner, audio: np.ndarray,
                       sample_rate: int = 16000) -> torch.Tensor:
    """(T, V) log-probabilities on the 20 ms grid."""
    inputs = aligner.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    values = inputs.input_values.to(aligner.device)
    logits = aligner.model(values).logits[0]
    return torch.log_softmax(logits.float(), dim=-1).cpu()


def viterbi_align(log_probs: torch.Tensor, token_ids: list[int],
                  blank_id: int = 0) -> list[tuple[int, int]]:
    """Most likely monotonic CTC path; returns [start_frame, end_frame) per token.

    The trellis is the standard one over the blank-interleaved sequence
    ``eps t0 eps t1 ... eps``, with three moves: stay, advance one, or skip a
    blank when the two surrounding tokens differ. ``max`` replaces ``sum`` and
    backpointers are kept so the path can be recovered.
    """
    if not token_ids:
        return []
    T, _ = log_probs.shape
    ext = [blank_id]
    for t in token_ids:
        ext.extend([int(t), blank_id])
    S = len(ext)
    if T < len(token_ids):
        raise ValueError(f"cannot align {len(token_ids)} tokens to {T} frames")

    neg = float("-inf")
    score = torch.full((S,), neg, dtype=torch.float64)
    score[0] = float(log_probs[0, ext[0]])
    if S > 1:
        score[1] = float(log_probs[0, ext[1]])
    back = np.zeros((T, S), dtype=np.int8)          # 0 stay, 1 from s-1, 2 from s-2

    for t in range(1, T):
        prev = score
        cur = torch.full((S,), neg, dtype=torch.float64)
        for s in range(S):
            best, arg = prev[s], 0
            if s >= 1 and prev[s - 1] > best:
                best, arg = prev[s - 1], 1
            # a blank may be skipped only between two *different* real tokens
            if s >= 2 and ext[s] != blank_id and ext[s] != ext[s - 2] and prev[s - 2] > best:
                best, arg = prev[s - 2], 2
            if best > neg:
                cur[s] = best + float(log_probs[t, ext[s]])
                back[t, s] = arg
        score = cur

    s = S - 1 if score[S - 1] >= score[S - 2] else S - 2
    path = np.zeros(T, dtype=np.int32)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
        s = max(0, s)

    spans: list[tuple[int, int]] = []
    for i in range(len(token_ids)):
        state = 2 * i + 1                            # real tokens sit at odd states
        frames = np.nonzero(path == state)[0]
        if len(frames):
            spans.append((int(frames[0]), int(frames[-1]) + 1))
        else:                                        # token absorbed by a neighbour
            spans.append((-1, -1))
    return spans


def align_units(aligner: CTCAligner, audio: np.ndarray, surfaces: list[str],
                sample_rate: int = 16000) -> list[dict]:
    """Frame spans for each *reference unit*, preserving the unit indexing.

    Each unit is romanized separately, so the mapping from CTC characters back to
    reference units is exact. Romanizing the whole utterance and re-splitting on
    whitespace would not work: uroman expands one Han character into a multi-letter
    syllable, so word counts stop matching unit counts.
    """
    vocab = aligner.processor.tokenizer.get_vocab()
    token_ids: list[int] = []
    owner: list[int] = []                            # unit index for each token
    for ui, surface in enumerate(surfaces):
        for ch in romanize(str(surface)).lower():
            if ch.isspace():
                continue
            tid = vocab.get(ch)
            if tid is None:
                continue                             # outside the CTC vocabulary
            token_ids.append(int(tid))
            owner.append(ui)
    if not token_ids:
        return []

    log_probs = ctc_log_posteriors(aligner, audio, sample_rate)
    if log_probs.shape[0] < len(token_ids):
        raise ValueError(f"cannot align {len(token_ids)} tokens to "
                         f"{log_probs.shape[0]} frames")
    spans = viterbi_align(log_probs, token_ids, aligner.blank_id)

    out: list[dict] = []
    for ui, surface in enumerate(surfaces):
        picks = [spans[i] for i, o in enumerate(owner) if o == ui and spans[i][0] >= 0]
        if not picks:
            continue                                 # unit absorbed by its neighbours
        start = min(p[0] for p in picks)
        end = max(p[1] for p in picks)
        out.append({
            "unit_id": ui,
            "surface": surface,
            "start_frame": int(start),
            "end_frame": int(end),
            "start_sec": float(start * aligner.frame_sec),
            "end_sec": float(end * aligner.frame_sec),
        })
    return out
