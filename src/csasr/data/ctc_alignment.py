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
from functools import lru_cache
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


#: Han numerals, and the pronunciation uroman throws away.
#:
#: uroman transliterates CJK numerals to **digits**: `一` -> `1`, `十` -> `10`,
#: `百` -> `100`, and `一百二十三` -> `123`. MMS-FA's vocabulary is 31 romanized
#: letters with no digits, so `align_units` skipped every character of the
#: romanization and the unit came out of `run_existing_ctc` as `mapping_failed`
#: with no span at all. Measured on job 38573's D-construct sweep: 2,283 of
#: 88,426 CTC candidates were invalid, **all** of them Mandarin, and 2,010 of
#: those were the single character `一`. Every one of the fourteen distinct
#: surfaces involved is in this table.
#:
#: That is a 2.58% invalid rate against a 1% limit, i.e. it alone kept
#: `existing_ctc` -- the family that covers 97.4% of units -- from qualifying as
#: an independent aligner. The repair is to give the model the syllable the
#: speaker actually said instead of a digit it cannot represent; it invents no
#: boundary and relabels nothing, the acoustic model still has to find the span.
#:
#: Toneless pinyin, because the vocabulary has no tone marks. Financial and
#: variant forms are included: they read the same and uroman digitises them too
#: (`陌` -> `100`, which is also simply wrong -- it reads `mo`).
CJK_NUMERAL_PINYIN = {
    "零": "ling", "〇": "ling", "一": "yi", "二": "er", "三": "san", "四": "si",
    "五": "wu", "六": "liu", "七": "qi", "八": "ba", "九": "jiu", "十": "shi",
    "百": "bai", "千": "qian",
    # financial/variant forms, same readings
    "壹": "yi", "贰": "er", "叁": "san", "肆": "si", "伍": "wu", "陆": "liu",
    "柒": "qi", "捌": "ba", "玖": "jiu", "拾": "shi", "佰": "bai", "仟": "qian",
    "陌": "mo",
}


def pronounce_numerals(text: str) -> str:
    """Replace Han numerals with their pinyin before uroman sees them.

    Separated by spaces so two adjacent numerals cannot fuse into a syllable
    neither of them is; `align_units` drops whitespace after romanizing, and the
    CTC trellis is over characters, so the spacing costs nothing.
    """
    out: list[str] = []
    for char in str(text):
        pinyin = CJK_NUMERAL_PINYIN.get(char)
        out.append(f" {pinyin} " if pinyin else char)
    return "".join(out)


def romanize(text: str) -> str:
    """Map any script to Latin so one vocabulary covers Mandarin and English.

    Mixed-script utterances are the whole problem: a Mandarin CTC vocabulary has
    no Latin letters and an English one has no Han characters. Romanizing both
    sides collapses them into a single alphabet, which is how the MMS aligner
    handles arbitrary languages.

    Han numerals are pronounced first -- see `CJK_NUMERAL_PINYIN` for the
    measurement that made this necessary.
    """
    return _uroman().romanize_string(pronounce_numerals(str(text)))


@lru_cache(maxsize=1)
def _uroman():
    """One romanizer for the process.

    Constructing `Uroman()` loads its romanization tables and costs ~3.1 s,
    against ~1 ms to romanize a string with an existing instance. `align_units`
    romanizes every reference unit separately, so building one per call charged
    that 3.1 s per *unit*: ~59 s for a typical utterance, which is what made
    CTC forced alignment look like an 80 GPU-hour job. Exceptions are not
    cached by `lru_cache`, so a missing package still raises on every call.
    """
    try:
        import uroman as ur
    except Exception as exc:                                   # pragma: no cover
        raise CTCUnavailable(
            "the `uroman` package is required to align mixed-script text") from exc
    return ur.Uroman()


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
    ext_a = np.asarray(ext, dtype=np.int64)
    # (T, S): each trellis state's emission score at each frame, gathered once
    # instead of indexed scalar-by-scalar inside the loop. numpy rather than
    # torch throughout: the trellis is small (a few hundred states) and at that
    # size torch's per-call overhead dominates -- a torch version of this same
    # loop measured 6x *slower* than the scalar original.
    emit = np.asarray(log_probs.to(torch.float64).numpy())[:, ext_a]

    # a blank may be skipped only between two *different* real tokens
    skip_ok = np.zeros(S, dtype=bool)
    if S > 2:
        skip_ok[2:] = (ext_a[2:] != blank_id) & (ext_a[2:] != ext_a[:-2])

    score = np.full(S, neg, dtype=np.float64)
    score[0] = emit[0, 0]
    if S > 1:
        score[1] = emit[0, 1]
    back = np.zeros((T, S), dtype=np.int8)          # 0 stay, 1 from s-1, 2 from s-2

    cand = np.empty((3, S), dtype=np.float64)
    # All three moves are evaluated for every state at once. `argmax` returns
    # the *first* maximal index, which reproduces the original's tie-break
    # order exactly: stay, then advance, then skip.
    for t in range(1, T):
        cand[0] = score
        cand[1, 0] = neg
        cand[1, 1:] = score[:-1]
        cand[2, :2] = neg
        cand[2, 2:] = np.where(skip_ok[2:], score[:-2], neg)
        arg = cand.argmax(axis=0)
        best = cand[arg, np.arange(S)]
        finite = best > neg
        score = np.where(finite, best + emit[t], neg)
        back[t] = np.where(finite, arg, 0)

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
