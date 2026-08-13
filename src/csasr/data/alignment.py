"""Reference-unit to encoder-frame alignment (E1).

The local CS-Dialogue release ships only `short_wav` — the `long_wav` TextGrids
were not downloaded — so dataset-provided word boundaries (guide section 10,
priority 1) are unavailable. We therefore use priority 3: forced alignment from
Whisper's own cross-attention, restricted to the model's published alignment
heads and resolved with DTW, exactly as in ``whisper.timing.find_alignment``
but *teacher-forced on the reference* rather than on a hypothesis.

That choice is audited in E1 (section 13); if the audit fails, an external
multilingual aligner must be compared before any representational claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from ..utils.logging import get_logger
from .language_tags import EN, UNKNOWN, ZH, is_content, tag_unit
from .normalize import Unit, normalize_and_segment

log = get_logger(__name__)

ALIGNMENT_SOURCE = "whisper_crossattn_dtw"

UNIT_COLUMNS = [
    "utterance_id", "unit_id", "surface", "normalized_surface", "language_tag",
    "start_sec", "end_sec", "start_frame", "end_frame", "is_content",
    "is_boundary_adjacent", "alignment_source", "alignment_confidence",
    "baseline_status", "baseline_error_type",
]


# --------------------------------------------------------------------------
# DTW
# --------------------------------------------------------------------------
def dtw(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monotonic DTW backtrace over a (num_tokens, num_frames) cost matrix.

    Returns index arrays (token_indices, frame_indices) along the optimal path.
    """
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    acc[0, 0] = 0.0
    trace = np.full((n + 1, m + 1), -1, dtype=np.int8)
    trace[0, :] = 2
    trace[:, 0] = 1

    for i in range(1, n + 1):
        ci = cost[i - 1]
        for j in range(1, m + 1):
            c0 = acc[i - 1, j - 1]   # diagonal: consume token and frame
            c1 = acc[i - 1, j]       # up: token advances, same frame
            c2 = acc[i, j - 1]       # left: frame advances, same token
            if c0 <= c1 and c0 <= c2:
                best, t = c0, 0
            elif c1 <= c0 and c1 <= c2:
                best, t = c1, 1
            else:
                best, t = c2, 2
            acc[i, j] = ci[j - 1] + best
            trace[i, j] = t

    i, j = n, m
    ti, tj = [], []
    while i > 0 or j > 0:
        ti.append(i - 1)
        tj.append(j - 1)
        t = trace[i, j]
        if t == 0:
            i, j = i - 1, j - 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
        if i < 0 or j < 0:
            break
    return np.array(ti[::-1]), np.array(tj[::-1])


def median_filter_1d(x: np.ndarray, width: int) -> np.ndarray:
    """Median filter along the last axis with edge replication."""
    if width <= 1 or x.shape[-1] < width:
        return x
    pad = width // 2
    padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width, axis=-1)
    return np.median(windows, axis=-1)

def token_frame_spans(text_indices: np.ndarray, frame_indices: np.ndarray,
                      n_text: int, n_valid: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert DTW rows for text tokens plus EOS into disjoint token spans.

    Decoder query position j predicts token j+1, so callers provide n_text+1
    DTW rows: one for each text token and one for EOS. EOS onset is the final
    boundary. Strictly increasing boundaries prevent cross-language overlap.
    """
    if n_text <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if n_valid < n_text:
        raise ValueError(f"{n_text} text tokens cannot fit in {n_valid} encoder frames")
    raw = []
    for k in range(n_text + 1):
        frames = frame_indices[text_indices == k]
        raw.append(int(frames.min()) if len(frames) else (raw[-1] if raw else 0))
    # Project raw boundaries onto a strictly increasing integer sequence.
    idx = np.arange(n_text + 1, dtype=np.int64)
    shifted = np.asarray(raw, dtype=np.int64) - idx
    shifted = np.maximum.accumulate(shifted)
    shifted = np.clip(shifted, 0, n_valid - n_text)
    bounds = shifted + idx
    return bounds[:-1], bounds[1:]


# --------------------------------------------------------------------------
# token / unit bookkeeping
# --------------------------------------------------------------------------
def token_char_offsets(tokenizer, token_ids: Sequence[int]) -> list[tuple[int, int]]:
    """Character span of each token in the decoded string (incremental decode)."""
    offsets = []
    prev = 0
    for k in range(len(token_ids)):
        text = tokenizer.decode(list(token_ids[: k + 1]), skip_special_tokens=True)
        cur = len(text)
        offsets.append((prev, max(prev, cur)))
        prev = max(prev, cur)
    return offsets


def build_prefix(processor, language: str = "zh", task: str = "transcribe") -> list[int]:
    tok = processor.tokenizer
    names = ["<|startoftranscript|>", f"<|{language}|>", f"<|{task}|>", "<|notimestamps|>"]
    ids = tok.convert_tokens_to_ids(names)
    if any(i is None for i in ids):
        raise RuntimeError(f"tokenizer lacks special tokens {names}")
    return list(ids)


@dataclass
class UtteranceAlignment:
    utterance_id: str
    units: list[Unit]
    tags: list[str]
    start_sec: np.ndarray
    end_sec: np.ndarray
    confidence: np.ndarray
    num_valid_frames: int


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------
@torch.inference_mode()
def align_batch(bundle, rows: pd.DataFrame, *, language: str = "zh",
                median_filter_width: int = 7,
                max_text_tokens: int = 220,
                pred_start_offset: int = -1) -> list[UtteranceAlignment]:
    """Force-align the normalized reference of each row to encoder frames.

    ``pred_start_offset`` selects which decoder query is treated as token k's
    attention row: ``-1`` uses the query that *predicts* token k (this module's
    original convention), ``0`` uses the query *at* token k (the convention in
    OpenAI's ``whisper.timing.find_alignment``, which slices at
    ``len(sot_sequence)``). One decoder step is 220-400 ms on this corpus, so
    the two conventions differ by roughly the systematic offset E1 measured
    relative to a constructed audio seam. The default preserves existing
    behaviour; `csasr.lss.align.devselect` compares the alternatives using an
    explicitly seam-relative development diagnostic. That diagnostic is not
    lexical-boundary accuracy evidence.
    """
    from ..models.generation import teacher_forced_forward

    tok = bundle.processor.tokenizer
    prefix = build_prefix(bundle.processor, language=language)
    eot = tok.eos_token_id

    seqs, metas = [], []
    for _, row in rows.iterrows():
        norm, units = normalize_and_segment(row["transcript_raw"])
        text_ids = tok.encode(norm, add_special_tokens=False)[:max_text_tokens]
        seqs.append(prefix + list(text_ids) + [eot])
        metas.append((row, norm, units, text_ids))

    out, padded, mask = teacher_forced_forward(
        bundle, rows["audio_path"].tolist(), seqs, output_attentions=True
    )
    heads = bundle.model.generation_config.alignment_heads
    if heads is None:
        raise RuntimeError("model generation_config has no alignment_heads")

    # cross_attentions: tuple[num_layers] of (B, heads, tgt, src)
    cross = out.cross_attentions
    stacked = torch.stack([cross[l][:, h] for l, h in heads], dim=1)  # (B, H, tgt, src)
    stacked = stacked.float().cpu().numpy()
    del out, cross

    results = []
    for bi, (row, norm, units, text_ids) in enumerate(metas):
        n_valid = bundle.valid_frames(row["duration_sec"])
        n_text = len(text_ids)
        w = stacked[bi][:, :, :n_valid]                     # (H, tgt, frames)
        # per-frame standardisation across the token axis, as in whisper.timing
        std = w.std(axis=-2, keepdims=True)
        mean = w.mean(axis=-2, keepdims=True)
        w = (w - mean) / np.maximum(std, 1e-8)
        w = median_filter_1d(w, median_filter_width)
        # Query j predicts token j+1: include the query before the first text
        # token and the final text query that predicts EOS.
        pred_start = len(prefix) + int(pred_start_offset)
        pred_start = max(0, pred_start)
        matrix = w.mean(axis=0)[pred_start: pred_start + n_text + 1]

        if n_text == 0 or matrix.size == 0:
            results.append(UtteranceAlignment(row["utterance_id"], [], [],
                                              np.zeros(0), np.zeros(0), np.zeros(0), n_valid))
            continue

        ti, tj = dtw(-matrix.astype(np.float64))
        tok_start, tok_end = token_frame_spans(ti, tj, n_text, n_valid)
        tok_conf = np.zeros(n_text, dtype=np.float64)
        for k in range(n_text):
            frames = np.arange(tok_start[k], tok_end[k], dtype=np.int64)
            tok_conf[k] = float(matrix[k, frames].mean()) if len(frames) else 0.0

        offsets = token_char_offsets(tok, text_ids)

        starts, ends, confs = [], [], []
        for u in units:
            hit = [k for k, (a, b) in enumerate(offsets)
                   if not (b <= u.char_start or a >= u.char_end)]
            if not hit:
                starts.append(np.nan)
                ends.append(np.nan)
                confs.append(0.0)
                continue
            s = int(tok_start[hit[0]])
            e = int(max(tok_end[k] for k in hit))
            starts.append(s * bundle.encoder_step_sec)
            ends.append(max(e, s + 1) * bundle.encoder_step_sec)
            confs.append(float(np.mean([tok_conf[k] for k in hit])))

        results.append(UtteranceAlignment(
            utterance_id=row["utterance_id"],
            units=units,
            tags=[tag_unit(u) for u in units],
            start_sec=np.asarray(starts, dtype=float),
            end_sec=np.asarray(ends, dtype=float),
            confidence=np.asarray(confs, dtype=float),
            num_valid_frames=n_valid,
        ))
    return results


def alignment_to_rows(bundle, alignment: UtteranceAlignment) -> list[dict]:
    """Flatten an utterance alignment into unit-table rows with frame spans."""
    rows = []
    tags = alignment.tags
    n = len(alignment.units)
    for i, unit in enumerate(alignment.units):
        s, e = alignment.start_sec[i], alignment.end_sec[i]
        if not np.isfinite(s) or not np.isfinite(e):
            continue
        sf, ef = bundle.sec_to_frames(float(s), float(e), alignment.num_valid_frames)
        assert sf < ef, f"empty frame span for unit {i} of {alignment.utterance_id}"
        prev_tag = tags[i - 1] if i > 0 else None
        next_tag = tags[i + 1] if i + 1 < n else None
        switch = (
            (prev_tag in (EN, ZH) and tags[i] in (EN, ZH) and prev_tag != tags[i])
            or (next_tag in (EN, ZH) and tags[i] in (EN, ZH) and next_tag != tags[i])
        )
        rows.append({
            "utterance_id": alignment.utterance_id,
            "unit_id": i,
            "surface": unit.surface,
            "normalized_surface": unit.normalized_surface,
            "language_tag": tags[i],
            "start_sec": float(s),
            "end_sec": float(e),
            "start_frame": int(sf),
            "end_frame": int(ef),
            "is_content": bool(is_content(unit, tags[i])),
            "is_boundary_adjacent": bool(switch),
            "alignment_source": ALIGNMENT_SOURCE,
            "alignment_confidence": float(alignment.confidence[i]),
            "baseline_status": "",
            "baseline_error_type": "",
        })
    return rows


def core_frames(start_frame: int, end_frame: int, exclude_frames: int,
                trim_start: bool = True, trim_end: bool = True) -> tuple[int, int]:
    """Trim boundary-adjacent frames from a unit span (guide section 12).

    Returns an empty span (s, s) when nothing survives the trim.
    """
    s = start_frame + (exclude_frames if trim_start else 0)
    e = end_frame - (exclude_frames if trim_end else 0)
    if e <= s:
        return start_frame, start_frame
    return s, e


def frames_for_language(units: pd.DataFrame, language: str, exclude_frames: int,
                        num_valid: int, drop_boundary: bool = True) -> np.ndarray:
    """Indices of encoder frames belonging to ``language`` in one utterance."""
    keep = np.zeros(num_valid, dtype=bool)
    sub = units[units["language_tag"] == language]
    for _, r in sub.iterrows():
        if drop_boundary and r["is_boundary_adjacent"]:
            s, e = core_frames(int(r["start_frame"]), int(r["end_frame"]), exclude_frames)
        else:
            s, e = int(r["start_frame"]), int(r["end_frame"])
        if e > s:
            keep[max(0, s): min(num_valid, e)] = True
    return np.flatnonzero(keep)


def validate_unit_table(units: pd.DataFrame, manifest: pd.DataFrame,
                        max_frames: int, step_sec: float,
                        strict: bool = True) -> dict:
    """E1 unit tests 3-6 applied to the produced table.

    ``strict`` raises on violation, which is what the unit tests want. The E1
    stage calls with ``strict=False`` so that a violation is *reported* and fails
    the gate cleanly, instead of killing the stage and discarding every artifact
    produced up to that point.
    """
    report: dict = {}
    report["num_units"] = int(len(units))
    bad_range = units[(units["start_frame"] < 0) | (units["end_frame"] > max_frames)]
    report["frames_out_of_range"] = int(len(bad_range))
    if strict:
        assert len(bad_range) == 0, "frame spans outside the encoder frame range"

    empty = units[units["end_frame"] <= units["start_frame"]]
    report["empty_spans"] = int(len(empty))
    if strict:
        assert len(empty) == 0, "non-positive frame spans present"

    dur = manifest.set_index("utterance_id")["duration_sec"].to_dict()
    over = 0
    for utt, grp in units.groupby("utterance_id"):
        limit = int(np.ceil(dur[utt] / step_sec))
        over += int((grp["end_frame"] > min(limit, max_frames)).sum())
    report["padding_frames_labelled"] = over
    if strict:
        assert over == 0, "units extend into padding frames"

    non_monotonic = 0
    for _, grp in units.groupby("utterance_id"):
        g = grp.sort_values("unit_id")
        non_monotonic += int((g["start_frame"].diff().fillna(0) < 0).sum())
    report["non_monotonic_units"] = non_monotonic

    overlap_conflicts = 0
    for _, grp in units.groupby("utterance_id"):
        g = grp[grp["language_tag"].isin([EN, ZH])].sort_values("start_frame")
        prev_end, prev_tag = -1, None
        for _, r in g.iterrows():
            if r["start_frame"] < prev_end and prev_tag != r["language_tag"]:
                overlap_conflicts += 1
            prev_end, prev_tag = r["end_frame"], r["language_tag"]
    report["conflicting_language_overlaps"] = overlap_conflicts
    if strict:
        assert non_monotonic == 0, "unit ordering is not monotonic"
        assert overlap_conflicts == 0, "overlapping conflicting language spans present"
    return report


# Each pass either shrinks an overlap or withdraws a claim, so convergence is
# fast; the cap only bounds pathological input.
_MAX_OVERLAP_PASSES = 8


def resolve_boundary_overlaps(units: pd.DataFrame,
                              step_sec: float) -> tuple[pd.DataFrame, dict]:
    """Make adjacent EN/ZH spans disjoint by truncating the earlier unit.

    ``sec_to_frames`` floors starts and ceils ends, so two adjacent units both
    claim the single frame that contains the boundary between them. That is
    conservative but leaves the spans formally overlapping, which matters twice:
    a frame shared by an EN and a ZH unit would be accumulated into *both*
    centroids in E2, and E4's steering masks would not be crisp.

    The earlier unit is truncated rather than the later one delayed, so the onset
    of the following span -- the thing E4 actually steers -- is preserved. A unit
    is never truncated to nothing.
    """
    if not len(units):
        return units, {"overlaps_resolved": 0, "overlaps_unresolved": 0,
                       "overlaps_dropped": 0}

    out = units.copy()
    resolved = dropped = 0
    max_shrink = 0

    def _drop(i) -> None:
        """Withdraw a unit's language claim, keeping the row for provenance."""
        out.at[i, "language_tag"] = UNKNOWN
        if "is_content" in out.columns:
            out.at[i, "is_content"] = False

    def _conflicting_pairs() -> list[tuple]:
        """Index pairs still violating the postcondition.

        Deliberately mirrors ``validate_unit_table``'s own check, so that a clean
        result here means a clean result there.
        """
        pairs = []
        for _, grp in out.groupby("utterance_id"):
            g = grp[grp["language_tag"].isin([EN, ZH])].sort_values("start_frame")
            idx = g.index.tolist()
            for a, b in zip(idx, idx[1:]):
                if (int(out.at[b, "start_frame"]) < int(out.at[a, "end_frame"])
                        and out.at[a, "language_tag"] != out.at[b, "language_tag"]):
                    pairs.append((a, b))
        return pairs

    def _sweep() -> bool:
        """One left-to-right pass. Returns True if anything changed."""
        nonlocal resolved, dropped, max_shrink
        changed = False
        for _, grp in out.groupby("utterance_id"):
            g = grp[grp["language_tag"].isin([EN, ZH])].sort_values("start_frame")
            # Compare against the last unit that still holds a claim rather than
            # the static predecessor: dropping a unit makes its neighbours
            # adjacent, and that newly exposed pair can itself conflict.
            prev = None
            for cur in g.index.tolist():
                if out.at[cur, "language_tag"] not in (EN, ZH):
                    continue
                if prev is None:
                    prev = cur
                    continue
                tag_a, tag_b = out.at[prev, "language_tag"], out.at[cur, "language_tag"]
                end_a = int(out.at[prev, "end_frame"])
                start_b = int(out.at[cur, "start_frame"])
                if start_b >= end_a or tag_a == tag_b:
                    prev = cur
                    continue
                changed = True
                if start_b > int(out.at[prev, "start_frame"]):  # leaves >=1 frame
                    max_shrink = max(max_shrink, end_a - start_b)
                    out.at[prev, "end_frame"] = start_b
                    # keep end_sec on the frame grid, never extending it
                    out.at[prev, "end_sec"] = min(float(out.at[prev, "end_sec"]),
                                                  float(start_b) * float(step_sec))
                    resolved += 1
                    prev = cur
                elif int(out.at[cur, "end_frame"]) > end_a:
                    # both start on the same frame, so the earlier one cannot be
                    # truncated without emptying it; delay the later one instead
                    max_shrink = max(max_shrink, end_a - start_b)
                    out.at[cur, "start_frame"] = end_a
                    out.at[cur, "start_sec"] = max(float(out.at[cur, "start_sec"]),
                                                   float(end_a) * float(step_sec))
                    resolved += 1
                    prev = cur
                elif int(out.at[cur, "end_frame"]) == end_a:
                    # Identical spans: the aligner gave two different words the
                    # same frames and there is no basis for preferring either, so
                    # both forfeit their claim.
                    _drop(prev)
                    _drop(cur)
                    dropped += 2
                    prev = None
                else:
                    # Strictly nested. The enclosing unit is corroborated by the
                    # rest of the sequence; the nested one is the anomaly.
                    _drop(cur)
                    dropped += 1
        return changed

    # One pass suffices for the rounding artefact this exists for, but units piled
    # on identical frames can expose a fresh conflict each time one is settled, so
    # iterate to a fixed point.
    for _ in range(_MAX_OVERLAP_PASSES):
        if not _sweep():
            break
    # Unconditional postcondition. Whatever the aligner produced, no EN frame may
    # also be a ZH frame downstream; anything the sweep could not separate forfeits
    # its language claim rather than being carried forward as a silent conflict.
    forced = 0
    for _ in range(_MAX_OVERLAP_PASSES):
        residual = _conflicting_pairs()
        if not residual:
            break
        for a, b in residual:
            for i in (a, b):
                if out.at[i, "language_tag"] in (EN, ZH):
                    _drop(i)
                    forced += 1
    dropped += forced
    return out, {
        "overlaps_resolved": int(resolved),
        "overlaps_unresolved": 0,
        "overlaps_dropped": int(dropped),
        "max_frames_trimmed": int(max_shrink),
        "note": ("adjacent EN/ZH spans sharing the boundary frame are made disjoint "
                 "by truncating the earlier unit; this is a floor/ceil rounding "
                 "artefact of sec_to_frames, not an alignment failure. Fully nested "
                 "conflicting spans cannot be separated and instead forfeit their "
                 "language tag (overlaps_dropped)."),
    }
