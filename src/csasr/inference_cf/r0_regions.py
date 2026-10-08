"""R0 reference-free acoustic regions, frame masks, query mapping and region controls
(frozen docs/inference_cf/R0_REGION_VECTOR_SPEC.md sections 3, 4 and 6).

Pure NumPy. Inputs are a waveform geometry, the 100-way native LID distributions of the fixed window grid, the
frozen alignment-head attention of the baseline replay and the generated baseline tokens. There is no reference,
gold region, CTC boundary, target, correctness or oracle argument anywhere in this module.

Label codes: 0 = U (uncertain), 1 = EN, 2 = ZH.
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

import numpy as np

U, EN, ZH = 0, 1, 2
NAMES = {U: "U", EN: "EN", ZH: "ZH"}
FRAME = 320


def window_grid(n: int, w: int = 16000, s: int = 8000) -> list[tuple[int, int]]:
    """Regular full windows plus a unique right-anchored final full window; one crop [0,N) if N<W."""
    if n <= 0:
        raise ValueError("empty waveform")
    if n < w:
        return [(0, n)]
    starts = set(range(0, n - w + 1, s)) | {n - w}
    return [(a, a + w) for a in sorted(starts)]


def classify_window(probs: np.ndarray, language_ids: Sequence[int], crop: np.ndarray, cfg: dict) -> dict:
    """Frozen EN/ZH/U rule for one window. Malformed probability vectors raise (engineering INVALID)."""
    p = np.asarray(probs, dtype=np.float64)
    if p.shape != (len(language_ids),) or len(language_ids) != cfg["softmax_size"]:
        raise ValueError("invalid_lid:shape")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or abs(float(p.sum()) - 1.0) > cfg["probability_tolerance"]:
        raise ValueError("invalid_lid:probabilities")
    ids = list(language_ids)
    pe, pm = float(p[ids.index(cfg["en_id"])]), float(p[ids.index(cfg["zh_id"])])
    q = pe + pm
    eps = cfg["epsilon"]
    pe_pair, pm_pair = pe / (q + eps), pm / (q + eps)
    x = np.asarray(crop, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    reasons = []
    if x.size < cfg["minimum_crop_samples"]:
        reasons.append("short_crop")
    if rms < cfg["minimum_rms"]:
        reasons.append("low_rms")
    if q < cfg["pair_mass_min"]:
        reasons.append("low_pair_mass")
    if reasons:
        label = U
    elif pe_pair >= cfg["conditional_language_mass_min"]:
        label = EN
    elif pm_pair >= cfg["conditional_language_mass_min"]:
        label = ZH
    else:
        label = U
        reasons.append("ambiguous_pair")
    nz = p[p > 0]
    return {"label": NAMES[label], "code": label, "reasons": reasons, "P_E": pe, "P_M": pm, "Q": q, "pE_pair": pe_pair,
            "pM_pair": pm_pair, "log_odds_EM": math.log((pe + eps) / (pm + eps)), "entropy": float(-(nz * np.log(nz)).sum()),
            "rms": rms, "samples": int(x.size)}


def sample_intervals(bounds: Sequence[tuple[int, int]], codes: Sequence[int], n: int, frac_min: float) -> list[tuple[int, int, int]]:
    """Interval sweep: at each sample, fraction of ALL covering windows labelled EN / ZH (U in the denominator);
    EN if >= frac_min, ZH if >= frac_min, else U; merge maximal identical runs. Returns (start, end, code)."""
    pts = sorted({0, n} | {a for a, _ in bounds} | {b for _, b in bounds})
    out: list[list[int]] = []
    for a, b in zip(pts[:-1], pts[1:]):
        cover = [c for (s, e), c in zip(bounds, codes) if s <= a and b <= e]
        if not cover:
            lab = U
        else:
            fe = sum(c == EN for c in cover) / len(cover)
            fz = sum(c == ZH for c in cover) / len(cover)
            lab = EN if fe >= frac_min else (ZH if fz >= frac_min else U)
        if out and out[-1][2] == lab and out[-1][1] == a:
            out[-1][1] = b
        else:
            out.append([a, b, lab])
    return [tuple(x) for x in out]


def intervals_to_track(intervals: Sequence[tuple[int, int, int]], n: int) -> np.ndarray:
    t = np.zeros(n, dtype=np.int8)
    for a, b, c in intervals:
        t[max(0, a):min(n, b)] = c
    return t


def track_to_intervals(track: np.ndarray) -> list[tuple[int, int, int]]:
    if track.size == 0:
        return []
    cut = np.flatnonzero(np.diff(track)) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [track.size]])
    return [(int(a), int(b), int(track[a])) for a, b in zip(starts, ends)]


def frame_masks(track_heard: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact sample-duration fractions of EN / ZH inside each 320-sample heard frame (partial last frame uses its
    actual duration)."""
    n = track_heard.size
    f = -(-n // FRAME)
    pad = np.full(f * FRAME, -1, dtype=np.int8)
    pad[:n] = track_heard
    blk = pad.reshape(f, FRAME)
    dur = (blk >= 0).sum(axis=1).astype(np.float64)
    we = (blk == EN).sum(axis=1) / dur
    wm = (blk == ZH).sum(axis=1) / dur
    return we, wm


def check_attention(heads_full: np.ndarray) -> None:
    """Malformed attention (nonfinite / negative / zero total over ALL encoder frames) is engineering INVALID."""
    h = np.asarray(heads_full, dtype=np.float64)
    if h.ndim != 3:
        raise ValueError("invalid_attention:shape")
    if not np.all(np.isfinite(h)) or np.any(h < 0):
        raise ValueError("invalid_attention:values")
    if np.any(h.mean(axis=0).sum(axis=1) <= 0):
        raise ValueError("invalid_attention:zero")


def query_mapping(heads_real: np.ndarray, mass_min: float) -> dict:
    """Arithmetic mean over the frozen alignment heads restricted to the actual heard frames; raw mass over those
    frames; queries with raw mass < mass_min are uncertain; otherwise normalized to sum 1 in float64.
    heads_real: (H, Q, F_real) -- exactly the sealed real-frame attention values."""
    h = np.asarray(heads_real, dtype=np.float64)
    if h.ndim != 3 or h.shape[2] <= 0:
        raise ValueError("invalid_attention:shape")
    real = h.mean(axis=0)
    raw = real.sum(axis=1)
    ok = raw >= mass_min
    norm = np.zeros_like(real)
    norm[ok] = real[ok] / raw[ok, None]
    argmax = np.argmax(real, axis=1)            # earliest maximal real frame
    return {"raw_mass": raw, "mapped": ok, "a": norm, "argmax_frame": argmax}


def assign_queries(a: np.ndarray, mapped: np.ndarray, eligible: np.ndarray, we: np.ndarray, wm: np.ndarray, cfg: dict) -> dict:
    """q_E = sum_f a(t,f) wE(f); EN iff q_E >= .70 and q_M <= .20; ZH symmetric; else U. Ineligible / unmapped => U."""
    qe = a @ we
    qm = a @ wm
    g = cfg["group_mass_min"]
    o = cfg["opposite_mass_max"]
    lab = np.zeros(a.shape[0], dtype=np.int8)
    use = mapped & eligible
    lab[use & (qe >= g) & (qm <= o)] = EN
    lab[use & (qm >= g) & (qe <= o)] = ZH
    return {"qE": np.where(use, qe, np.nan), "qM": np.where(use, qm, np.nan), "label": lab}


def time_bins(argmax_frame: np.ndarray, bin_samples: int) -> np.ndarray:
    return (np.asarray(argmax_frame, dtype=np.int64) * FRAME) // bin_samples


# ---- reference-free controls -------------------------------------------------------------------------------------

def shuffle_offsets(uid: str, n_heard: int, k: int, margin: int, key_fmt: str) -> list[int] | None:
    """k distinct offsets without replacement from inclusive [margin, n_heard-margin], PCG64 seeded by the first 16
    SHA256 hex digits of the frozen key; sorted. None if fewer than k legal offsets (control unavailable)."""
    lo, hi = margin, n_heard - margin
    if hi - lo + 1 < k:
        return None
    seed = int(hashlib.sha256(key_fmt.replace("<utterance_id>", uid).encode()).hexdigest()[:16], 16)
    pick = np.random.Generator(np.random.PCG64(seed)).choice(np.arange(lo, hi + 1), size=k, replace=False)
    return sorted(int(x) for x in pick)


def shuffle_seed(uid: str, key_fmt: str) -> int:
    return int(hashlib.sha256(key_fmt.replace("<utterance_id>", uid).encode()).hexdigest()[:16], 16)


def erode(track: np.ndarray, d: int) -> np.ndarray:
    out = np.zeros_like(track)
    for a, b, c in track_to_intervals(track):
        if c != U and b - d > a + d:
            out[a + d:b - d] = c
    return out


def dilate(track: np.ndarray, d: int) -> np.ndarray:
    n = track.size
    e = np.zeros(n, dtype=bool)
    m = np.zeros(n, dtype=bool)
    for a, b, c in track_to_intervals(track):
        if c == EN:
            e[max(0, a - d):min(n, b + d)] = True
        elif c == ZH:
            m[max(0, a - d):min(n, b + d)] = True
    out = np.zeros(n, dtype=np.int8)
    out[e & ~m] = EN
    out[m & ~e] = ZH
    return out
