"""Alignment-quality checks for E1 production gating.

These checks do not replace the guide's required human audit. They provide
blocking automatic evidence before E2/E4 may consume DTW-derived language spans.

Two independent signals live here:

``silence_absorption``
    Detects spans whose edges sit inside silence. This is the concrete defect
    already observed on this corpus (a single-character unit spanning 0.00-0.88 s),
    and it needs no second model -- only the waveform.

``synthetic_boundary_error``
    Builds utterances by concatenating monolingual Mandarin and English segments
    at *known* times, then measures what the aligner reports against that truth.
    This is the only component that yields real ground-truth error, at the cost of
    being concatenated rather than natural speech (no cross-boundary
    coarticulation, no natural switch prosody), so it is an optimistic bound and
    must be reported as such.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..models.whisper import load_audio
from ..utils.hashing import sha256_file, sha256_obj
from ..utils.logging import get_logger

log = get_logger(__name__)

# The encoder step; keeping the RMS grid identical to it means frame indices in
# this module and in the alignment tables refer to the same instants.
FRAME_SEC = 0.02


def frame_rms(audio: np.ndarray, sample_rate: int,
              frame_sec: float = FRAME_SEC) -> np.ndarray:
    """Per-frame RMS energy on the encoder's own 20 ms grid."""
    n = int(round(frame_sec * sample_rate))
    if n <= 0 or len(audio) < n:
        return np.zeros(0, dtype=np.float32)
    count = len(audio) // n
    trimmed = np.asarray(audio[: count * n], dtype=np.float64).reshape(count, n)
    return np.sqrt((trimmed ** 2).mean(axis=1)).astype(np.float32)


def _edge_silence_frames(rms: np.ndarray, threshold: float, start: int, end: int,
                         from_end: bool = False) -> int:
    """Length of the silent run at one edge of the half-open span [start, end)."""
    start = max(0, min(start, len(rms)))
    end = max(start, min(end, len(rms)))
    order = range(end - 1, start - 1, -1) if from_end else range(start, end)
    run = 0
    for i in order:
        if rms[i] > threshold:
            break
        run += 1
    return run


def silence_absorption(units: pd.DataFrame, manifest: pd.DataFrame, *,
                       sample_utterances: int = 200, seed: int = 42,
                       silence_ms: float = 100.0,
                       rel_threshold: float = 0.05) -> dict:
    """Fraction of aligned units whose span begins or ends inside silence.

    A unit boundary sitting more than ``silence_ms`` inside a silent stretch is
    evidence the aligner absorbed non-speech into the span. The threshold is
    relative to each utterance's own loudness (``rel_threshold`` of its 95th
    percentile frame RMS), so it adapts to recording level.
    """
    content = units[units["language_tag"].isin(["EN", "ZH"])]
    if not len(content) or not len(manifest):
        return {"num_units_checked": 0,
                "note": "no content units available for silence analysis"}

    utts = content["utterance_id"].drop_duplicates()
    if len(utts) > sample_utterances:
        utts = utts.sample(sample_utterances, random_state=seed)
    paths = manifest.set_index("utterance_id")["audio_path"].to_dict()
    min_run = int(round(silence_ms / 1000.0 / FRAME_SEC))

    lead, trail, checked = [], [], 0
    for utt in utts:
        path = paths.get(utt)
        if path is None:
            continue
        try:
            audio = load_audio(str(path))
        except Exception as exc:                     # unreadable audio is diagnostic
            log.warning("silence check skipped %s: %s", utt, exc)
            continue
        rms = frame_rms(audio, 16000)
        if not len(rms):
            continue
        threshold = float(rel_threshold * np.percentile(rms, 95))
        for _, r in content[content["utterance_id"] == utt].iterrows():
            s = int(round(float(r["start_sec"]) / FRAME_SEC))
            e = int(round(float(r["end_sec"]) / FRAME_SEC))
            if e <= s:
                continue
            lead.append(_edge_silence_frames(rms, threshold, s, e))
            trail.append(_edge_silence_frames(rms, threshold, s, e, from_end=True))
            checked += 1

    if not checked:
        return {"num_units_checked": 0,
                "note": "no unit could be checked against its waveform"}
    lead_a, trail_a = np.asarray(lead), np.asarray(trail)
    absorbed = (lead_a >= min_run) | (trail_a >= min_run)
    return {
        "num_units_checked": int(checked),
        "silence_ms_threshold": float(silence_ms),
        "leading_silence_rate": float((lead_a >= min_run).mean()),
        "trailing_silence_rate": float((trail_a >= min_run).mean()),
        "absorption_rate": float(absorbed.mean()),
        "median_leading_silence_ms": float(np.median(lead_a) * FRAME_SEC * 1000),
        "p95_leading_silence_ms": float(np.percentile(lead_a, 95) * FRAME_SEC * 1000),
        "note": ("fraction of aligned units whose span edge lies more than "
                 f"{silence_ms:.0f} ms inside silence; high values indicate the "
                 "aligner is absorbing non-speech into language spans"),
    }


# ---------------------------------------------------------------------------
# synthetic ground truth
# ---------------------------------------------------------------------------
def build_synthetic_pairs(manifest: pd.DataFrame, *, n_pairs: int = 100,
                          seed: int = 42, gap_ms: float = 0.0,
                          sample_rate: int = 16000,
                          max_duration_sec: float = 28.0) -> list[dict]:
    """Concatenate a monolingual ZH utterance with a monolingual EN one.

    The concatenation point is a *known* language boundary, which is what makes
    this ground truth. Only utterances whose reference contains a single script
    are used, so the resulting item is genuinely ZH-then-EN.

    ``max_duration_sec`` keeps every constructed item inside Whisper's 30 s
    receptive field. Without it the aligner simply never sees the boundary of a
    long pair, and the resulting "error" measures the window, not the aligner.
    The budget is applied to the *untrimmed* durations, so silence trimming in
    ``render_synthetic_pair`` can only make the item shorter.

    Returns descriptors; audio is rendered lazily by ``render_synthetic_pair`` so
    that nothing large is held in memory.
    """
    from .language_tags import EN, ZH, tag_unit
    from .normalize import normalize_text, segment_units

    def single_language(text: str) -> str | None:
        tags = {t for t in (tag_unit(u) for u in segment_units(normalize_text(str(text))))
                if t in (EN, ZH)}
        return next(iter(tags)) if len(tags) == 1 else None

    lang = manifest["transcript_raw"].map(single_language)
    zh = manifest[lang == "ZH"]
    en = manifest[lang == "EN"]
    if not len(zh) or not len(en):
        log.warning("cannot build synthetic pairs: %d ZH-only, %d EN-only utterances",
                    len(zh), len(en))
        return []

    # Halve the budget per side so any admissible ZH can pair with any admissible
    # EN; this is stricter than necessary but keeps the sampling independent.
    budget = (float(max_duration_sec) - float(gap_ms) / 1000.0) / 2.0
    if "duration_sec" in manifest.columns and budget > 0:
        zh_long, en_long = len(zh), len(en)
        zh = zh[zh["duration_sec"] <= budget]
        en = en[en["duration_sec"] <= budget]
        if zh_long - len(zh) or en_long - len(en):
            log.info("synthetic pairs: dropped %d ZH / %d EN utterances longer than "
                     "%.1f s to stay inside Whisper's %.0f s window",
                     zh_long - len(zh), en_long - len(en), budget, max_duration_sec)
        if not len(zh) or not len(en):
            log.warning("no monolingual utterance short enough for a synthetic pair")
            return []

    n = min(n_pairs, len(zh), len(en))
    zh_pick = zh.sample(n, random_state=seed).reset_index(drop=True)
    en_pick = en.sample(n, random_state=seed + 1).reset_index(drop=True)

    def source_content_id(row: pd.Series) -> str:
        for col in ("audio_sha256", "source_audio_sha256"):
            if col in row and str(row[col]):
                return str(row[col])
        path = Path(str(row["audio_path"]))
        if path.exists():
            return sha256_file(path)
        return f"missing:{path}"

    pairs = []
    for i in range(n):
        pairs.append({
            "pair_id": f"syn_{i:04d}",
            "zh_utterance_id": zh_pick.loc[i, "utterance_id"],
            "en_utterance_id": en_pick.loc[i, "utterance_id"],
            "zh_audio_path": zh_pick.loc[i, "audio_path"],
            "en_audio_path": en_pick.loc[i, "audio_path"],
            "zh_audio_content_sha256": source_content_id(zh_pick.loc[i]),
            "en_audio_content_sha256": source_content_id(en_pick.loc[i]),
            "zh_duration_sec": float(zh_pick.loc[i, "duration_sec"])
            if "duration_sec" in zh_pick.columns else None,
            "en_duration_sec": float(en_pick.loc[i, "duration_sec"])
            if "duration_sec" in en_pick.columns else None,
            "zh_text": zh_pick.loc[i, "transcript_raw"],
            "en_text": en_pick.loc[i, "transcript_raw"],
            "gap_sec": float(gap_ms) / 1000.0,
            "sample_rate": int(sample_rate),
        })
    return pairs


def synthetic_harness_fingerprint(pairs: list[dict], settings: dict) -> dict:
    """Stable configuration/content hash for rendered synthetic audio.

    The rendered files are invalid if either the harness settings change or any
    source clip changes. Pair descriptors include source audio content hashes
    when the manifest provides them, otherwise this module hashes the source
    clips directly.
    """
    payload = {
        "settings": settings,
        "pairs": [
            {
                "pair_id": p.get("pair_id"),
                "zh_utterance_id": p.get("zh_utterance_id"),
                "en_utterance_id": p.get("en_utterance_id"),
                "zh_audio_path": p.get("zh_audio_path"),
                "en_audio_path": p.get("en_audio_path"),
                "zh_audio_content_sha256": p.get("zh_audio_content_sha256"),
                "en_audio_content_sha256": p.get("en_audio_content_sha256"),
                "zh_text": p.get("zh_text"),
                "en_text": p.get("en_text"),
                "gap_sec": p.get("gap_sec"),
                "sample_rate": p.get("sample_rate"),
            }
            for p in pairs
        ],
    }
    return {"sha256": sha256_obj(payload), "payload": payload}


def speech_bounds(audio: np.ndarray, sample_rate: int, *,
                  rel_threshold: float = 0.05, pad_ms: float = 20.0) -> tuple[int, int]:
    """Sample range spanned by speech, with a short pad on each side.

    Used to strip the lead-in and lead-out silence that corpus segmentation
    leaves on every clip before the clips are spliced together.
    """
    rms = frame_rms(audio, sample_rate)
    if not len(rms):
        return 0, len(audio)
    voiced = np.flatnonzero(rms > float(rel_threshold * np.percentile(rms, 95)))
    if not len(voiced):
        return 0, len(audio)
    pad = int(round(pad_ms / 1000.0 / FRAME_SEC))
    per_frame = int(round(FRAME_SEC * sample_rate))
    first = max(0, int(voiced[0]) - pad) * per_frame
    last = min(len(rms), int(voiced[-1]) + 1 + pad) * per_frame
    return int(first), int(min(len(audio), last))


def render_synthetic_pair(pair: dict, out_path, *, trim_silence: bool = True,
                          pad_ms: float = 20.0) -> dict:
    """Write the concatenated waveform and return its known boundary time.

    Both clips are stripped of their edge silence first. Without that the splice
    lands in the middle of the ~1.3 s of silence that the two clips contribute
    between them, and the "true" boundary is then over half a second away from
    the instant speech actually changes language -- which is what the aligner
    marks, and what E2/E4 care about. Trimming makes the constructed instant a
    genuine speech transition, so the measured error is the aligner's own.
    """
    import soundfile as sf

    sr = int(pair["sample_rate"])
    zh = load_audio(str(pair["zh_audio_path"]), sr)
    en = load_audio(str(pair["en_audio_path"]), sr)
    if trim_silence:
        s, e = speech_bounds(zh, sr, pad_ms=pad_ms)
        zh = zh[s:e]
        s, e = speech_bounds(en, sr, pad_ms=pad_ms)
        en = en[s:e]
    gap = np.zeros(int(round(float(pair["gap_sec"]) * sr)), dtype=np.float32)
    audio = np.concatenate([zh, gap, en]).astype(np.float32)
    sf.write(str(out_path), audio, sr)
    # The boundary is the instant Mandarin stops. With a gap it is the midpoint of
    # the inserted silence, matching switch_boundaries()' own midpoint convention.
    zh_end = len(zh) / sr
    boundary = zh_end + float(pair["gap_sec"]) / 2.0
    return {
        **pair,
        "audio_path": str(out_path),
        "true_boundary_sec": float(boundary),
        "zh_end_sec": float(zh_end),
        "en_start_sec": float(zh_end + float(pair["gap_sec"])),
        "duration_sec": float(len(audio) / sr),
        "transcript_raw": f"{pair['zh_text']} {pair['en_text']}",
    }


def summarize_boundary_error(predicted_sec: np.ndarray,
                             true_sec: np.ndarray) -> dict:
    """Error distribution of predicted boundaries against known truth."""
    predicted_sec = np.asarray(predicted_sec, dtype=float)
    true_sec = np.asarray(true_sec, dtype=float)
    ok = np.isfinite(predicted_sec) & np.isfinite(true_sec)
    if not ok.any():
        return {"num_boundaries": 0,
                "note": "no synthetic boundary could be scored"}
    signed_ms = (predicted_sec[ok] - true_sec[ok]) * 1000.0
    abs_ms = np.abs(signed_ms)
    stats = {
        "num_boundaries": int(ok.sum()),
        "mean_signed_error_ms": float(signed_ms.mean()),
        "median_signed_error_ms": float(np.median(signed_ms)),
        "median_abs_error_ms": float(np.median(abs_ms)),
        "p95_abs_error_ms": float(np.percentile(abs_ms, 95)),
        "note": ("measured against known concatenation points; concatenated speech "
                 "lacks cross-boundary coarticulation and natural switch prosody, "
                 "so this is an optimistic bound on real alignment error"),
    }
    # Reported across the whole range so the boundary-exclusion window can be
    # calibrated from the measured distribution rather than assumed.
    for tol in (50, 100, 150, 200, 250, 300, 500):
        stats[f"pct_within_{tol}ms"] = float((abs_ms <= tol).mean())
    return stats
