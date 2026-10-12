"""ABTS-0 K1 -- reference-free candidate extraction, acoustic-span association and the frozen acoustic scoring rule.

Design: docs/research_review/ABTS0_DESIGN.md (K1 only); frozen parameters: configs/inference_cf/abts0_k1.json;
interpretations of underspecified clauses: docs/research_review/ABTS0_K1_REPORT.md section 2 (I1-I12).

* ``matched_decode``    -- the matched forced-prompt engine (Branch.step + processed_argmax, lowest-ID ties), identical
                           to ``episodic_tta.forced_decode`` but also returning each step's cached alignment-head attention.
* ``token_intervals``   -- DTW token timing from a branch's own cached attention (prediction-query offset -1).
* ``content_units``     -- Han characters and Latin words of a generated token sequence with their token spans.
* ``associate``         -- step 2: forced-EN word <-> contiguous B0 Han block; target query = block's first token.
* ``region_check``      -- step 3: R0 predicted-EN fraction and the unchanged S1 target / off-target crops.
* ``choose``            -- step 4: earliest structurally eligible pair, fixed before any acoustic score.
* ``score_word`` / ``acoustic_rule`` -- step 5: F, A, C and the accept / abstain rule.

No reference, evaluator annotation, target registry or outcome input anywhere in this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Sequence

import numpy as np

from csasr.data.alignment import dtw, median_filter_1d, token_frame_spans
from csasr.data.normalize import is_han

FRAME = 320
HEARD_MAX = 480000
WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
LN2 = math.log(2.0)
MIN_COVER = 0.5
ONSET_TOL = 3200
PRESENT_TOL = 3200
R0_EN_MIN = 0.5
CROP_COVER_MIN = 0.5
HEARD_MASS_MIN = 0.5
R0_EN_CODE = 1


def heard_samples(n: int) -> int:
    return min(int(n), HEARD_MAX)


def heard_frames(heard: int) -> int:
    return -(-int(heard) // FRAME)


# ---- engine (GPU at run time; CPU-testable on a tiny model) -----------------------------------------------------------

def matched_decode(bundle, encoded, prompt: Sequence[int], *, eos: int, suppress: Sequence[int], begin: Sequence[int],
                   max_new_tokens: int = 200, capture_layer: int = 16, keep_heads: bool = True) -> dict:
    """``episodic_tta.forced_decode`` numerics (fresh cache, prompt then one token per step, attention=True + passive
    recorder, processed argmax with lowest-ID ties), additionally keeping every step's alignment-head attention and a
    SHA256 of every step's raw float32 logits (bitwise engine-identity checks for later replays)."""
    import hashlib
    import torch
    import experiments.inference_cf_cached as cached
    from csasr.inference_cf.core_p1 import processed_argmax
    from csasr.lss.sites import assert_no_site_hooks
    br = cached.Branch(bundle, encoded, list(prompt), "K1")
    new, tokens, heads, sha, term = list(prompt), [], [], [], "cap"
    with torch.inference_mode():
        for t in range(max_new_tokens):
            assert_no_site_hooks(bundle)
            logits, h, _ = br.step(new, capture_layer=capture_layer, attention=True)
            sha.append("sha256:" + hashlib.sha256(logits.numpy().astype(np.float32).tobytes()).hexdigest())
            if keep_heads:
                heads.append(h.numpy().astype(np.float32))
            nxt = processed_argmax(logits, t, suppress, begin)
            if nxt == eos:
                term = "eos"
                break
            tokens.append(nxt)
            new = [nxt]
    assert_no_site_hooks(bundle)
    if br.positions != list(range(br.length)):
        raise RuntimeError("matched_decode: cache positions")
    return {"tokens": tokens, "terminated": term, "logits_sha256": sha,
            "heads": np.stack(heads) if keep_heads else None}


def score_word(bundle, encoded, prompt: Sequence[int], prefix: Sequence[int], word: Sequence[int], *,
               suppress: Sequence[int], begin: Sequence[int], capture_layer: int = 16) -> dict:
    """F(w) = mean_j log P(w_j | a, prompt + prefix, w_<j) on a fresh cold Branch fed one token per step (the
    matched-engine numerics). Returns per-piece log-probs, the query logits hash and the processed argmax at the query."""
    import hashlib
    import torch
    import experiments.inference_cf_cached as cached
    from csasr.lss.sites import assert_no_site_hooks
    if not word:
        raise ValueError("empty candidate")
    br = cached.Branch(bundle, encoded, list(prompt), "K1S")
    feed = [list(prompt)] + [[int(x)] for x in prefix]
    lp, q_sha, q_arg = [], None, None
    with torch.inference_mode():
        logits = None
        for f in feed:
            assert_no_site_hooks(bundle)
            logits, _, _ = br.step(f, capture_layer=capture_layer, attention=True)
        for j, w in enumerate(word):
            step = len(prefix) + j
            x = cached._processed(logits, step, suppress, begin)
            ls = torch.log_softmax(x, dim=-1)
            if j == 0:
                q_sha = "sha256:" + hashlib.sha256(logits.numpy().astype(np.float32).tobytes()).hexdigest()
                q_arg = int(torch.argmax(x))
            lp.append(float(ls[int(w)].double()))
            if j + 1 < len(word):
                assert_no_site_hooks(bundle)
                logits, _, _ = br.step([int(w)], capture_layer=capture_layer, attention=True)
    assert_no_site_hooks(bundle)
    if br.fed != list(prompt) + [int(x) for x in prefix] + [int(x) for x in word[:-1]]:
        raise RuntimeError("score_word: lineage")
    return {"logp": lp, "F": float(np.mean(np.asarray(lp, dtype=np.float64))), "query_logits_sha256": q_sha,
            "query_argmax": q_arg}


def valid_pieces(ids: Sequence[int], eos: int, suppress: Sequence[int]) -> bool:
    s = set(int(x) for x in suppress)
    return bool(ids) and all(0 <= int(i) < eos and int(i) not in s for i in ids)


# ---- step 1: token timing from the branch's own cached attention ------------------------------------------------------

def token_intervals(heads: np.ndarray, n_tokens: int, terminated: str, heard: int, median_width: int = 7) -> dict:
    """heads: (S, H, F>=heard frames) cached attention; row s is the query predicting content token s, row n_tokens
    predicts EOS when terminated == 'eos'. Per-frame standardization over the prediction rows, median filter, mean over
    heads, monotonic DTW. Returns half-open sample intervals for the first ``n_timed`` tokens."""
    F = heard_frames(heard)
    n_timed = int(n_tokens) if terminated == "eos" else int(n_tokens) - 1
    if n_timed <= 0:
        return {"status": "NO_TOKENS", "n_timed": 0, "start": [], "end": []}
    h = np.asarray(heads, dtype=np.float64)
    if h.ndim != 3 or h.shape[0] < n_timed + 1 or h.shape[2] < F:
        raise ValueError("attention_shape")
    if not np.isfinite(h).all() or (h < 0).any():
        raise ValueError("attention_values")
    if F < n_timed:
        return {"status": "TOO_FEW_FRAMES", "n_timed": 0, "start": [], "end": []}
    w = h[: n_timed + 1, :, :F].transpose(1, 0, 2)
    w = (w - w.mean(axis=1, keepdims=True)) / np.maximum(w.std(axis=1, keepdims=True), 1e-8)
    w = median_filter_1d(w, median_width)
    ti, tj = dtw(-w.mean(axis=0))
    s, e = token_frame_spans(ti, tj, n_timed, F)
    return {"status": "OK", "n_timed": n_timed, "start": [int(min(a * FRAME, heard)) for a in s],
            "end": [int(min(b * FRAME, heard)) for b in e]}


# ---- content units ----------------------------------------------------------------------------------------------------

def utf8_chars(data: bytes) -> list[tuple[str, int, int, bool]]:
    """Strict incremental UTF-8: (char, byte_start, byte_end, valid); an invalid lead/continuation byte becomes one
    invalid U+FFFD char of one byte, so byte <-> char offsets stay exact."""
    out, i, n = [], 0, len(data)
    while i < n:
        b = data[i]
        k = 1 if b < 0x80 else 2 if 0xC2 <= b < 0xE0 else 3 if 0xE0 <= b < 0xF0 else 4 if 0xF0 <= b < 0xF5 else 0
        if k and i + k <= n:
            try:
                out.append((data[i:i + k].decode("utf-8"), i, i + k, True))
                i += k
                continue
            except UnicodeDecodeError:
                pass
        out.append(("�", i, i + 1, False))
        i += 1
    return out


@dataclass(frozen=True)
class Unit:
    kind: str           # HAN | LATIN
    surface: str
    char_start: int
    char_end: int
    byte_start: int
    tok_first: int
    tok_last: int       # inclusive

    def to_dict(self) -> dict:
        return asdict(self)


def content_units(token_bytes: Sequence[bytes]) -> dict:
    """Han characters (one unit each) and maximal Latin words of the decoded byte stream, with covering token spans."""
    bounds = np.cumsum([0] + [len(b) for b in token_bytes])
    chars = utf8_chars(b"".join(token_bytes))
    text = "".join(c for c, _, _, _ in chars)

    def toks(b0: int, b1: int) -> tuple[int, int]:
        first = int(np.searchsorted(bounds, b0, side="right") - 1)
        last = int(np.searchsorted(bounds, b1, side="left") - 1)
        return first, last

    units = []
    for ci, (ch, b0, b1, ok) in enumerate(chars):
        if ok and is_han(ch):
            units.append(Unit("HAN", ch, ci, ci + 1, b0, *toks(b0, b1)))
    for m in WORD_RE.finditer(text):
        b0, b1 = chars[m.start()][1], chars[m.end() - 1][2]
        units.append(Unit("LATIN", m.group(0), m.start(), m.end(), b0, *toks(b0, b1)))
    units.sort(key=lambda u: u.char_start)
    return {"text": text, "units": units, "chars_valid": [ok for _, _, _, ok in chars], "token_bounds": bounds.tolist(),
            "char_bytes": [(b0, b1) for _, b0, b1, _ in chars]}


def unit_interval(u: Unit, timing: dict) -> tuple[int, int] | None:
    if timing["status"] != "OK" or u.tok_last >= timing["n_timed"]:
        return None
    return int(timing["start"][u.tok_first]), int(timing["end"][u.tok_last])


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


# ---- step 2: association ----------------------------------------------------------------------------------------------

def associate(word: Unit, w_iv: tuple[int, int] | None, b0: dict, b0_timing: dict, heard: int) -> dict:
    """Associate one forced-EN word with the contiguous B0 Han block intersecting its interval (design K1 step 2)."""
    rec = {"status": None, "surface": word.surface, "en_interval": None, "en_first_token": word.tok_first, "block": None,
           "t": None, "b_surface": None, "han_cover": None, "onset_diff": None}
    if w_iv is None:
        rec["status"] = "EN_UNTIMED"
        return rec
    s, e = int(w_iv[0]), min(int(w_iv[1]), heard)
    if s >= heard:
        rec["status"] = "BEYOND_HEARD"
        return rec
    if e <= s:
        rec["status"] = "ZERO_DURATION"
        return rec
    rec["en_interval"] = [s, e]
    units = b0["units"]
    ivs = [unit_interval(u, b0_timing) for u in units]
    lo, hi = s - PRESENT_TOL, e + PRESENT_TOL
    for u, iv in zip(units, ivs):
        if u.kind == "LATIN" and u.surface.casefold() == word.surface.casefold() and (iv is None or _overlap(iv, (lo, hi)) > 0):
            rec["status"] = "PRESENT_IN_B0"
            return rec
    hit = [k for k, (u, iv) in enumerate(zip(units, ivs)) if u.kind == "HAN" and iv is not None and _overlap(iv, (s, e)) > 0]
    if not hit:
        rec["status"] = "NO_HAN_OVERLAP"
        return rec
    if hit != list(range(hit[0], hit[-1] + 1)):
        rec["status"] = "NONCONTIGUOUS_BLOCK"
        return rec
    block = [units[k] for k in hit]
    rec["block"] = [hit[0], hit[-1]]
    cov = np.zeros(e - s, dtype=bool)
    for k in hit:
        a, b = ivs[k]
        cov[max(a, s) - s:max(min(b, e), s) - s] = True
    rec["han_cover"] = float(cov.mean())
    rec["onset_diff"] = int(ivs[hit[0]][0] - s)
    rec["b_surface"] = b0["text"][block[0].char_start:block[-1].char_end]
    if rec["han_cover"] < MIN_COVER:
        rec["status"] = "LOW_HAN_COVER"
        return rec
    if abs(rec["onset_diff"]) > ONSET_TOL:
        rec["status"] = "ONSET_MISMATCH"
        return rec
    t = block[0].tok_first
    rec["t"] = int(t)
    if b0["token_bounds"][t] != block[0].byte_start or not all(b0["chars_valid"][:block[0].char_start]):
        rec["status"] = "MID_CHARACTER_ONSET"
        return rec
    if t < 1:
        rec["status"] = "CONTENT_T0"
        return rec
    rec["status"] = "ASSOCIATED"
    return rec


# ---- step 3: predicted-EN region and S1 crops -------------------------------------------------------------------------

def en_fraction(track_intervals: np.ndarray, iv: tuple[int, int], heard: int) -> float:
    s, e = max(0, int(iv[0])), min(int(iv[1]), heard)
    if e <= s:
        return 0.0
    n = 0
    for a, b, c in np.asarray(track_intervals, dtype=np.int64).tolist():
        if int(c) == R0_EN_CODE:
            n += _overlap((max(0, a), min(heard, b)), (s, e))
    return n / (e - s)


def shift_track(track_intervals: np.ndarray, heard: int) -> np.ndarray:
    """Circular shift of the heard track by floor(heard/2) (shuffled-association control)."""
    from csasr.inference_cf.r0_regions import intervals_to_track, track_to_intervals
    tr = intervals_to_track([tuple(r) for r in np.asarray(track_intervals).tolist()], heard)
    return np.asarray(track_to_intervals(np.roll(tr, heard // 2)), dtype=np.int64).reshape(-1, 3)


def region_check(rec: dict, heads_t: np.ndarray, track_intervals: np.ndarray, x: np.ndarray, heard: int, s1cfg: dict) -> dict:
    """Design K1 step 3 on an ASSOCIATED proposal; adds region fields and sets status ELIGIBLE or a rejection."""
    from csasr.inference_cf.s1_evidence import heard_attention, select_offtarget, select_target
    out = dict(rec)
    iv = tuple(rec["en_interval"])
    out["r0_en_fraction"] = en_fraction(track_intervals, iv, heard)
    out["target"] = out["off_target"] = None
    if out["r0_en_fraction"] < R0_EN_MIN:
        out["status"] = "NOT_IN_R0_EN"
        return out
    att = heard_attention(heads_t, heard, HEARD_MASS_MIN)
    tgt = select_target(att, track_intervals, x, heard, s1cfg)
    out["target"] = {k: tgt[k] for k in ("status", "bounds", "integral", "rms", "raw_mass")}
    if tgt["status"] != "OK":
        out["status"] = f"TARGET_{tgt['status']}"
        return out
    cover = _overlap(tuple(tgt["bounds"]), iv) / (iv[1] - iv[0])
    out["target"]["word_cover"] = cover
    if cover < CROP_COVER_MIN:
        out["status"] = "CROP_MISSES_WORD"
        return out
    off = select_offtarget(att, track_intervals, x, heard, tgt, s1cfg)
    out["off_target"] = {k: off[k] for k in ("status", "bounds", "integral", "rms", "energy_ratio", "en_fraction")}
    if off["status"] != "OK":
        out["status"] = f"OFF_{off['status']}"
        return out
    out["status"] = "ELIGIBLE"
    return out


# ---- step 4: deterministic earliest choice ----------------------------------------------------------------------------

def choose(proposals: Sequence[dict]) -> dict | None:
    elig = [p for p in proposals if p["status"] == "ELIGIBLE"]
    if not elig:
        return None
    return min(elig, key=lambda p: (p["en_interval"][0], p["en_first_token"], p["surface"]))


# ---- step 5: frozen acoustic rule -------------------------------------------------------------------------------------

def acoustic_rule(F: dict) -> dict:
    """F[cond][w] for cond in (x, target, off), w in (c, b). Accept iff A_target(c) >= ln2 and C >= ln2."""
    vals = [F[k][w] for k in ("x", "target", "off") for w in ("c", "b")]
    if not all(math.isfinite(v) for v in vals):
        return {"status": "NONFINITE", "accept": False, "off_only_accept": False}
    A = {k: {w: F["x"][w] - F[k][w] for w in ("c", "b")} for k in ("target", "off")}
    C = (A["target"]["c"] - A["target"]["b"]) - (A["off"]["c"] - A["off"]["b"])
    acc = A["target"]["c"] >= LN2 and C >= LN2
    return {"status": "ACCEPT" if acc else "ABSTAIN", "accept": bool(acc), "A": A, "C": C,
            "off_only_accept": bool(A["off"]["c"] - A["off"]["b"] >= LN2)}


# ---- per-utterance structural pools (reference-free) ------------------------------------------------------------------

def structural_pools(*, en: dict, en_timing: dict, b0: dict, b0_timing: dict, heard: int, b0_heads: np.ndarray,
                     track: np.ndarray, x: np.ndarray, s1cfg: dict) -> dict:
    """All forced-EN word proposals with their step-2 / step-3 status and the earliest eligible choice."""
    b0_latin = {u.surface.casefold() for u in b0["units"] if u.kind == "LATIN"}
    props, u0 = [], []
    for u in en["units"]:
        if u.kind != "LATIN":
            continue
        iv = unit_interval(u, en_timing)
        if iv is not None and iv[0] < heard and u.surface.casefold() not in b0_latin:
            u0.append({"surface": u.surface, "en_interval": [int(iv[0]), int(min(iv[1], heard))],
                       "en_first_token": u.tok_first})
        rec = associate(u, iv, b0, b0_timing, heard)
        if rec["status"] == "ASSOCIATED":
            rec = region_check(rec, b0_heads[rec["t"]], track, x, heard, s1cfg)
        props.append(rec)
    return {"U0": u0, "proposals": props, "chosen": choose(props)}
