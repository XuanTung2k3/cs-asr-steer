"""R0 per-utterance small-rank unique-subspace constructor (frozen spec section 5).

Raw uncentered float64 moments of ONE utterance's own grouped decoder states, via thin SVD of the row matrix (eigen-
values s^2/n, eigenvectors = right singular vectors; missing eigenvalues padded with 0). The largest rank r in
{8,4,2} supported in BOTH groups by every pre-winner guard is chosen; no lower-rank retry after a winner-guard
failure. Principal pairs of U_E, U_M; score_i = (a_i' C_E a_i)(1 - sigma_i^2); guarded winner; orientation toward
mean_E - mean_M; float64 normalize, float32 serialize. Invalid constructions return explicit reasons and NO vector.
The same function serves predicted, oracle, shuffled, perturbed and script memberships (spec: identical policy).
Centering is used ONLY for the participation-ratio redundancy guard.
"""
from __future__ import annotations

import hashlib

import numpy as np

VERSION = "R0_SMALL_RANK_UNIQUE_V1"


def array_hash(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode() + b"|" + repr(tuple(a.shape)).encode() + b"|")
    h.update(a.tobytes())
    return "sha256:" + h.hexdigest()


def _group(H: np.ndarray, bins: np.ndarray, sv_floor: float) -> dict:
    H = np.asarray(H, dtype=np.float64)
    n = H.shape[0]
    out = {"n": int(n), "bins": int(len(set(np.asarray(bins).tolist()))) if n else 0}
    if n == 0:
        out.update(s=np.zeros(0), Vt=np.zeros((0, H.shape[1])), lam=np.zeros(0), num_rank=0, pr=0.0, lam_c=np.zeros(0))
        return out
    _, s, vt = np.linalg.svd(H, full_matrices=False)
    lam = s * s / n
    num_rank = int(np.sum(s > sv_floor * s[0])) if s[0] > 0 else 0
    sc = np.linalg.svd(H - H.mean(axis=0), compute_uv=False)
    lam_c = sc * sc / n
    tot = float(lam_c.sum())
    pr = 0.0 if tot <= 0 else float(tot * tot / float(lam_c @ lam_c))
    out.update(s=s, Vt=vt, lam=lam, num_rank=num_rank, pr=pr, lam_c=lam_c)
    return out


def _lam(g: dict, i: int) -> float:
    """i-th (1-based) eigenvalue of C, zero-padded."""
    return float(g["lam"][i - 1]) if i - 1 < g["lam"].size else 0.0


def rank_evidence(g: dict, r: int, cfg: dict) -> dict:
    lr, lr1, l1 = _lam(g, r), _lam(g, r + 1), _lam(g, 1)
    ev = {"count": g["n"] >= 2 * r + 2, "bins": g["bins"] >= r + 1, "numerical_rank": g["num_rank"] >= r,
          "participation": g["pr"] >= r,
          "eigen_floor": l1 > 0 and lr > cfg["eigen_floor_relative_to_largest"] * l1,
          "eigen_gap": lr > 0 and (lr - lr1) > cfg["boundary_gap_relative_to_retained"] * lr}
    return ev


def construct(H_E: np.ndarray, H_M: np.ndarray, bins_E: np.ndarray, bins_M: np.ndarray, cfg: dict, *, keep: bool = False) -> dict:
    """Return a complete record; rec['vector'] is a float32 unit vector or None."""
    gE = _group(H_E, bins_E, cfg["numerical_rank_sv_relative_floor"])
    gM = _group(H_M, bins_M, cfg["numerical_rank_sv_relative_floor"])
    rec = {"version": VERSION, "n_E": gE["n"], "n_M": gM["n"], "bins_E": gE["bins"], "bins_M": gM["bins"],
           "num_rank_E": gE["num_rank"], "num_rank_M": gM["num_rank"], "pr_E": gE["pr"], "pr_M": gM["pr"],
           "candidates": {}, "rank": None, "status": "invalid", "reasons": [], "vector": None, "vector_sha256": None}
    rec["lam_E_head"] = [float(x) for x in gE["lam"][:9]]
    rec["lam_M_head"] = [float(x) for x in gM["lam"][:9]]
    for r in sorted(cfg["candidate_ranks"], reverse=True):
        evE, evM = rank_evidence(gE, r, cfg), rank_evidence(gM, r, cfg)
        rec["candidates"][str(r)] = {"E": evE, "M": evM}
        if rec["rank"] is None and all(evE.values()) and all(evM.values()):
            rec["rank"] = r
    if keep:
        rec["_spectra"] = {"s_E": gE["s"], "s_M": gM["s"], "lamc_E": gE["lam_c"], "lamc_M": gM["lam_c"]}
    if rec["rank"] is None:
        fails = set()
        for r, ev in rec["candidates"].items():
            for side in ("E", "M"):
                fails |= {f"{side}:{k}" for k, v in ev[side].items() if not v}
        rec["reasons"] = ["NO_SUPPORTED_RANK"] + sorted(fails)
        return rec
    r = rec["rank"]
    UE, UM = gE["Vt"][:r].T, gM["Vt"][:r].T
    HE = np.asarray(H_E, dtype=np.float64)
    HM = np.asarray(H_M, dtype=np.float64)
    P, sig_raw, Qt = np.linalg.svd(UE.T @ UM)
    reasons = []
    if np.any(sig_raw > 1 + cfg["principal_cosine_out_of_bounds_tolerance"]) or np.any(sig_raw < -cfg["principal_cosine_out_of_bounds_tolerance"]):
        reasons.append("sigma_out_of_bounds")
    sig = np.clip(sig_raw, 0.0, 1.0)
    a, b = UE @ P, UM @ Qt.T
    I = np.eye(r)
    ortho = {"U_E": float(np.abs(UE.T @ UE - I).max()), "U_M": float(np.abs(UM.T @ UM - I).max()),
             "a": float(np.abs(a.T @ a - I).max()), "b": float(np.abs(b.T @ b - I).max()),
             "pairs": float(np.abs(a.T @ b - np.diag(sig)).max())}
    if max(ortho.values()) > cfg["orthogonality_tolerance"]:
        reasons.append("orthonormality")
    energy = np.sum((HE @ a) ** 2, axis=0) / HE.shape[0]
    score = energy * (1.0 - sig ** 2)
    w = int(np.argmax(score))
    srt = np.sort(score)[::-1]
    gap = float(srt[0] - srt[1]) if r > 1 else float(srt[0])
    iso = float(min(abs(sig[j] - sig[w]) for j in range(r) if j != w)) if r > 1 else float("inf")
    # thin-SVD eigenvalues s^2/n are >= 0 by construction, so the frozen "materially negative C eigenvalue" guard
    # (-1e-10 lambda_1) cannot trigger on this path; explicit-moment agreement is tested on synthetic fixtures.
    if min(float(gE["lam"].min(initial=0.0)), float(gM["lam"].min(initial=0.0))) < -cfg["negative_eigenvalue_relative_tolerance"] * max(_lam(gE, 1), _lam(gM, 1)):
        reasons.append("negative_eigenvalue")
    if not score[w] > cfg["score_floor_relative_to_energy"] * float(energy.max()):
        reasons.append("score_floor")
    if not gap > cfg["winner_gap_relative_to_winner"] * float(score[w]):
        reasons.append("winner_gap")
    if not iso > cfg["principal_cosine_isolation"]:
        reasons.append("sigma_isolation")
    c = HE.mean(axis=0) - HM.mean(axis=0)
    cn = float(np.linalg.norm(c))
    v = a[:, w]
    cos_c = float(v @ c / (np.linalg.norm(v) * cn)) if cn > 0 else 0.0
    if not cn > cfg["mean_contrast_norm_min"]:
        reasons.append("mean_contrast_norm")
    elif not abs(cos_c) > cfg["sign_cosine_abs_min"]:
        reasons.append("sign_degenerate")
    flipped = cos_c < 0
    v = (-v if flipped else v)
    v = v / np.linalg.norm(v)
    v32 = v.astype(np.float32)
    unit_err = abs(float(np.linalg.norm(v32.astype(np.float64))) - 1.0)
    if unit_err > cfg["float32_unit_tolerance"]:
        reasons.append("float32_unit")
    rec.update(sigma_raw=[float(x) for x in sig_raw], sigma=[float(x) for x in sig], energy=[float(x) for x in energy],
               score=[float(x) for x in score], winner=w, score_gap=gap, sigma_isolation=iso, ortho=ortho,
               mean_contrast_norm=cn, cos_winner_contrast=cos_c, sign_flipped=bool(flipped), unit_error=unit_err)
    if keep:
        rec["_basis"] = {"U_E": UE, "U_M": UM, "P": P, "Q": Qt.T}
    if reasons:
        rec["reasons"] = reasons
        return rec
    rec["status"] = "ok"
    rec["vector"] = v32
    rec["vector_sha256"] = array_hash(v32)
    return rec


def public(rec: dict) -> dict:
    """JSON-safe record without arrays."""
    return {k: v for k, v in rec.items() if k not in ("vector", "_basis", "_spectra")}
