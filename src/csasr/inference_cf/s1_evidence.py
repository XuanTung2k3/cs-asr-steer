"""S1 acoustic-evidence pure helpers (frozen docs/inference_cf/S1_ACOUSTIC_EVIDENCE_SPEC.md sections 3-6,
configs/inference_cf/s1_acoustic_evidence.json; freeze a89067b).

* ``runtime_projection``: the label-free runtime view of the S1 panel (queries + audio/prefix/source metadata).
* ``topk`` / ``union``: finite processed-logit Top-K (descending value, ascending token-ID ties) per branch and the
  deduplicated ascending-ID union with origin bits and per-branch ranks. EOS -> TERMINATE, other IDs >= EOS ->
  OTHER_ACTION; nothing is filtered by script or by any reference.
* ``heard_attention`` / ``Integrals``: mean frozen alignment heads restricted to real heard frames
  (r0_regions.query_mapping), each 320-sample frame spread uniformly over its actual heard samples, exact half-open
  interval integrals.
* ``select_target`` / ``select_offtarget``: one predicted-English target crop and one matched-duration off-target crop.
* ``scores`` / ``ranking`` / ``shuffle_seed`` / ``shuffled``: fixed-candidate S(c)=2*l0-lm (lambda 1) and controls.

No reference, acceptable-token set, stratum, competitor, oracle region or masked-logit quantity enters candidate or
region selection here.
"""
from __future__ import annotations

import hashlib
import math
from fractions import Fraction
from typing import Sequence

import numpy as np

from csasr.inference_cf.r0_regions import EN, intervals_to_track, query_mapping

FRAME = 320
BRANCHES = ("M", "E", "AUTO")
BIT = {"M": 1, "E": 2, "AUTO": 4}
RUNTIME_QUERY_KEYS = ("utterance_id", "dialogue_id", "t", "absolute_query", "content_prefix_sha256",
                      "forced_zh_query_input_sha256", "forced_en_query_input_sha256")
RUNTIME_UTTERANCE_KEYS = ("utterance_id", "dialogue_id", "audio_path", "audio_sha256", "audio_full_file_sha256",
                          "audio_geometry", "baseline_row", "baseline_row_sha256", "baseline_system",
                          "r0_primary_arrays", "r0_primary_arrays_sha256")


def runtime_projection(panel: dict) -> dict:
    """Runtime view: queries (frozen key set) and per-utterance audio / R0-array / prefix metadata. The content tokens
    are truncated to the utterance's largest query t (content[:t] for every query; no later tokens)."""
    tmax: dict[str, int] = {}
    queries = []
    for q in panel["runtime_queries"]:
        if set(q) != set(RUNTIME_QUERY_KEYS):
            raise ValueError("unexpected runtime query keys")
        queries.append({k: q[k] for k in RUNTIME_QUERY_KEYS})
        tmax[q["utterance_id"]] = max(tmax.get(q["utterance_id"], 0), int(q["t"]))
    lac = {r["utterance_id"]: r for r in panel["lac_sources"]}
    utts = []
    for u in panel["utterances"]:
        rec = {k: u[k] for k in RUNTIME_UTTERANCE_KEYS}
        rec["content_prefix_tokens"] = [int(x) for x in u["baseline_content_tokens"][:tmax[u["utterance_id"]]]]
        rec["max_t"] = tmax[u["utterance_id"]]
        rec["lac_row"], rec["lac_row_sha256"] = lac[u["utterance_id"]]["row"], lac[u["utterance_id"]]["row_sha256"]
        rec["lac_logits"], rec["lac_logits_sha256"] = lac[u["utterance_id"]]["logits"], lac[u["utterance_id"]]["logits_sha256"]
        utts.append(rec)
    return {"runtime_queries": queries, "utterances": utts}


# ---- candidates ----------------------------------------------------------------------------------------------------

def topk(processed: np.ndarray, k: int) -> list[int]:
    """Finite processed-logit Top-K: descending value, ascending token ID on exact ties."""
    x = np.asarray(processed)
    if x.ndim != 1 or x.dtype != np.float32:
        raise ValueError("processed logits must be float32 1-D")
    if np.isnan(x).any() or np.isposinf(x).any():
        raise ValueError("processed logits contain NaN/+inf")
    ids = np.flatnonzero(np.isfinite(x))
    if ids.size < k:
        raise ValueError("fewer finite tokens than K")
    order = np.lexsort((ids, -x[ids].astype(np.float64)))
    return [int(i) for i in ids[order[:k]]]


def action_type(token: int, eos: int) -> str:
    if token == eos:
        return "TERMINATE"
    if token > eos:
        return "OTHER_ACTION"
    return "LEXICAL"


def union(tops: dict[str, list[int]], eos: int) -> dict:
    """Ascending-ID union of the branch Top-K lists with origin bits and 1-based per-branch ranks (None if absent)."""
    ids = sorted({int(i) for b in BRANCHES for i in tops[b]})
    ranks = {b: {int(i): r + 1 for r, i in enumerate(tops[b])} for b in BRANCHES}
    return {"ids": ids,
            "origin": [sum(BIT[b] for b in BRANCHES if i in ranks[b]) for i in ids],
            "ranks": {b: [ranks[b].get(i) for i in ids] for b in BRANCHES},
            "types": [action_type(i, eos) for i in ids]}


def membership_digest(ids: Sequence[int]) -> str:
    return "sha256:" + hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


# ---- predicted-English region and attention integration --------------------------------------------------------------
#
# The frozen float64-normalized attention a_f = mean_h(A_hf) / sum_heard mean_h(A_hf) and the interval integral
# sum_f a_f * overlap_f / dur_f are realized EXACTLY: every float32 attention value times 2**160 is an integer, so a crop
# integral is the rational N / (D * sum_f H_f) with H_f = sum_h A_hf * 2**160 and D = lcm(320, last-frame duration).
# Crop ranking / ties and the 1/10 thresholds compare these integers, so mathematically tied crops are tied (no float
# rounding decides a frozen tie rule); reported integrals are the float64 value of the same rational.

SCALE = 2.0 ** 160


def heard_attention(heads: np.ndarray, heard: int, mass_min: float) -> dict:
    """heads: (H, F_total) frozen alignment-head attention at the current M query. Mean over heads restricted to the
    real heard frames (ceil(heard/320)); raw mass / mapped / float64-normalized a via r0_regions.query_mapping, plus the
    exact integer frame masses H_f used for every integral."""
    h32 = np.asarray(heads, dtype=np.float32)
    F = -(-int(heard) // FRAME)
    if h32.ndim != 2 or h32.shape[1] < F or not np.isfinite(h32).all() or (h32 < 0).any():
        raise ValueError("invalid_attention")
    m = query_mapping(h32[:, None, :F].astype(np.float64), mass_min)
    sc = h32[:, :F].astype(np.float64) * SCALE
    H = [sum(int(v) for v in sc[:, f]) for f in range(F)]
    return {"raw_mass": float(m["raw_mass"][0]), "mapped": bool(m["mapped"][0]), "a": m["a"][0], "frames": F, "H": H}


class Integrals:
    """Exact crop integrals. num(s, e) is an integer; the normalized integral is num / den."""

    def __init__(self, H: Sequence[int], heard: int):
        self.F, self.heard = len(H), int(heard)
        self.H = [int(v) for v in H]
        last = self.heard - (self.F - 1) * FRAME
        self.D = FRAME * last // math.gcd(FRAME, last)
        self.cum = [0]
        for v in self.H:
            self.cum.append(self.cum[-1] + v * self.D)
        self.den = self.cum[-1]
        self.dur = [min(FRAME, self.heard - f * FRAME) for f in range(self.F)]

    def C(self, x: int) -> int:
        f = min(int(x) // FRAME, self.F - 1)
        part = int(x) - f * FRAME
        return self.cum[f] + self.H[f] * part * (self.D // self.dur[f])

    def num(self, s: int, e: int) -> int:
        return self.C(e) - self.C(s)

    def value(self, n: int) -> float:
        return float(Fraction(n, self.den)) if self.den else 0.0

    def ge_tenth(self, n: int) -> bool:          # n / den >= 1/10
        return 10 * n >= self.den


def crop_starts(left: int, right: int, length: int, step: int = FRAME) -> list[int]:
    """left + n*step starts fitting inside [left, right) plus the unique right-anchored last start."""
    if right - left < length:
        return []
    return sorted(set(range(left, right - length + 1, step)) | {right - length})


def rms(x: np.ndarray, s: int, e: int) -> float:
    v = np.asarray(x[s:e], dtype=np.float64)
    return float(np.sqrt(np.mean(v * v))) if v.size else 0.0


def energy(x: np.ndarray, s: int, e: int) -> float:
    v = np.asarray(x[s:e], dtype=np.float64)
    return float((v * v).sum())


def en_intervals(track_intervals: np.ndarray, heard: int, min_samples: int) -> list[tuple[int, int]]:
    out = []
    for a, b, c in np.asarray(track_intervals, dtype=np.int64).tolist():
        a, b = max(0, int(a)), min(int(heard), int(b))
        if int(c) == EN and b - a >= min_samples:
            out.append((a, b))
    return out


def select_target(att: dict, track_intervals: np.ndarray, x: np.ndarray, heard: int, cfg: dict) -> dict:
    """Frozen predicted-English target crop for one query (spec section 4). Abstentions are explicit statuses."""
    R = cfg["regions"]
    if R["associated_mass_min"] != 0.1:
        raise ValueError("association guard changed")
    ivs = en_intervals(track_intervals, heard, R["minimum_target_samples"])
    rec = {"status": None, "en_intervals": [list(i) for i in ivs], "raw_mass": att["raw_mass"], "bounds": None,
           "integral": None, "integral_num": None, "integral_den": None, "rms": None, "energy": None, "n_crops": 0}
    if not ivs:
        rec["status"] = "NO_EN_REGION"
        return rec
    if not att["mapped"]:
        rec["status"] = "LOW_HEARD_MASS"
        return rec
    I = Integrals(att["H"], heard)
    best = None
    for left, right in ivs:
        L = min(right - left, R["maximum_target_samples"])
        for s in crop_starts(left, right, L):
            n = I.num(s, s + L)
            rec["n_crops"] += 1
            key = (-n, s, s + L)
            if best is None or key < best[0]:
                best = (key, s, s + L, n)
    _, s, e, n = best
    rec.update(bounds=[int(s), int(e)], integral=I.value(n), integral_num=str(n), integral_den=str(I.den), rms=rms(x, s, e), energy=energy(x, s, e))
    if not I.ge_tenth(n):
        rec["status"] = "LOW_ASSOCIATION"
    elif rec["rms"] < R["target_minimum_rms"]:
        rec["status"] = "LOW_ENERGY"
    else:
        rec["status"] = "OK"
    return rec


def select_offtarget(att: dict, track_intervals: np.ndarray, x: np.ndarray, heard: int, target: dict, cfg: dict) -> dict:
    """Matched-duration off-target crop (spec section 5); requires an OK target. NO_MATCHED_OFFTARGET otherwise."""
    O = cfg["counterfactual"]["off_target"]
    if O["attention_mass_max"] != "min(0.10,target_mass/2)" or O["predicted_EN_fraction_max"] != 0.1:
        raise ValueError("off-target guards changed")
    rec = {"status": None, "bounds": None, "integral": None, "integral_num": None, "rms": None, "energy": None, "energy_ratio": None,
           "en_fraction": None, "n_starts": 0, "n_eligible": 0}
    if target["status"] != "OK":
        rec["status"] = "NO_TARGET"
        return rec
    a, b = target["bounds"]
    L = b - a
    starts = crop_starts(0, heard, L)
    rec["n_starts"] = len(starts)
    I = Integrals(att["H"], heard)
    nt = int(target["integral_num"])
    en = (intervals_to_track([tuple(r) for r in np.asarray(track_intervals).tolist()], heard) == EN).astype(np.int64)
    ce = np.concatenate([[0], np.cumsum(en)])
    Et = target["energy"]
    elig = []
    for s in starts:
        if not (s + L <= a or s >= b):
            continue
        cnt = int(ce[s + L] - ce[s])
        if 10 * cnt > L:                                   # predicted-EN sample fraction > 1/10
            continue
        n = I.num(s, s + L)
        if 10 * n > I.den or 2 * n > nt:                  # attention mass > min(1/10, target_mass/2)
            continue
        r = rms(x, s, s + L)
        if r < O["rms_min"]:
            continue
        e = energy(x, s, s + L)
        ratio = e / Et
        if not (O["energy_ratio_range"][0] <= ratio <= O["energy_ratio_range"][1]):
            continue
        elig.append(((n, abs(math.log(ratio)), s), s, n, r, e, ratio, cnt / L))
    rec["n_eligible"] = len(elig)
    if not elig:
        rec["status"] = "NO_MATCHED_OFFTARGET"
        return rec
    _, s, n, r, e, ratio, f = min(elig, key=lambda z: z[0])
    rec.update(status="OK", bounds=[int(s), int(s + L)], integral=I.value(n), integral_num=str(n), rms=r, energy=e, energy_ratio=ratio, en_fraction=f)
    return rec


# ---- fixed-candidate scoring ---------------------------------------------------------------------------------------

def scores(l0: np.ndarray, lm: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """S(c) = l0 + lambda*(l0 - lm) in float64 from float32 full-vocabulary log-probabilities."""
    l0, lm = np.asarray(l0, dtype=np.float64), np.asarray(lm, dtype=np.float64)
    if not (np.isfinite(l0).all() and np.isfinite(lm).all()):
        raise ValueError("nonfinite legal candidate log-probability")
    return l0 + lam * (l0 - lm)


def ranking(ids: Sequence[int], score: np.ndarray) -> list[int]:
    """Union IDs ordered by descending score, lower token ID on exact ties."""
    ids = np.asarray(ids, dtype=np.int64)
    s = np.asarray(score, dtype=np.float64)
    if not np.isfinite(s).all():
        raise ValueError("nonfinite score")
    return [int(i) for i in ids[np.lexsort((ids, -s))]]


def shuffle_seed(uid: str, t: int, k: int) -> int:
    return int(hashlib.sha256(f"s1-shuffle-v1|{uid}|{int(t)}|{int(k)}".encode()).hexdigest()[:16], 16)


def shuffled(support: np.ndarray, uid: str, t: int, k: int) -> np.ndarray:
    """One PCG64 permutation of the EN support vector (indexed by ascending union ID) among the union IDs."""
    s = np.asarray(support, dtype=np.float64)
    perm = np.random.Generator(np.random.PCG64(shuffle_seed(uid, t, k))).permutation(s.size)
    return s[perm]
