"""P2-DIR D1 UNIQUE direction: ``NEW_P2_DIR_CROSSFIT_V1`` (frozen spec section 4).

Pure CPU float64 construction of one sealed unit vector per leave-one-dialogue-out fold from
unedited L16 DG-02 baseline site states of the frozen correct strata:

* A = frozen EN-correct positions outside the evaluated dialogue, B = frozen ZH-correct positions
  outside the evaluated dialogue (no confusion states, no reference tokens, no outcomes);
* uncentered second moments ``C = sum(h h^T) / n`` (symmetrized), ``numpy.linalg.eigh``,
  exactly the top-32 eigenvectors of each group;
* principal pairs from ``svd(U_A^T U_B)``; score ``s_i = E_A(i) (1 - sigma_i^2)`` with
  ``E_A(i) = a_i^T C_A a_i``; argmax over all 32 (lowest index on exact ties);
* sign toward ``mu_A - mu_B``; exact float64 normalization, float32 serialization.

Every numerical guard failure makes the fold INVALID (zero edit for its dialogue); there is no
reduced-rank, alternate-vector or global-refit fallback. The runtime provider only ever reads the
sealed float32 vector; this module is never given reference information.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import numpy as np

VERSION = "NEW_P2_DIR_CROSSFIT_V1"
GROUP_A = "EN-correct"
GROUP_B = "ZH-correct"
RANK = 32
MIN_COUNT = 32
REL_TOL = 1e-10          # degeneracy guards (relative to the largest eigenvalue / max E_A)
ORTHO_TOL = 1e-8         # orthonormality / principal-pair identity
UNIT_TOL = 2e-6          # float32 serialization norm error
ALLOWED_FIELDS = ("utterance_id", "dialogue_id", "t", "stratum")


def sort_key(p: dict) -> tuple:
    return (str(p["dialogue_id"]), str(p["utterance_id"]), int(p["t"]))


def array_hash(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode() + b"|" + repr(tuple(a.shape)).encode() + b"|")
    h.update(a.tobytes())
    return "sha256:" + h.hexdigest()


def check_construction_fields(positions: Iterable[dict]) -> None:
    """The construction projection may carry ONLY ids, dialogue ids, t and stratum labels."""
    for p in positions:
        extra = set(p) - set(ALLOWED_FIELDS)
        if extra:
            raise ValueError(f"forbidden construction fields: {sorted(extra)}")


def fold_dialogues(positions: Sequence[dict]) -> list[str]:
    """Evaluation folds: one per dialogue present in the frozen population (sorted)."""
    return sorted({str(p["dialogue_id"]) for p in positions})


def fold_members(positions: Sequence[dict], k: str) -> dict:
    """Leave-dialogue-k-out A/B membership (sorted by dialogue, utterance, t)."""
    check_construction_fields(positions)
    keep = sorted((p for p in positions if str(p["dialogue_id"]) != k), key=sort_key)
    excluded = sorted((p for p in positions if str(p["dialogue_id"]) == k), key=sort_key)
    key = lambda p: [str(p["utterance_id"]), int(p["t"])]
    return {"dialogue": k,
            "A": [key(p) for p in keep if p["stratum"] == GROUP_A],
            "B": [key(p) for p in keep if p["stratum"] == GROUP_B],
            "excluded": [key(p) for p in excluded],
            "excluded_utterances": sorted({str(p["utterance_id"]) for p in excluded})}


def moments(H: np.ndarray) -> dict:
    """sum(h) and sum(h h^T) in CPU float64 over rows in the given (frozen) order."""
    H = np.asarray(H, dtype=np.float64)
    return {"n": int(H.shape[0]), "sum": H.sum(axis=0), "sum_outer": H.T @ H}


def _top(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, V = np.linalg.eigh(C)
    return w[::-1].copy(), V[:, ::-1].copy()


def _ortho_err(M: np.ndarray) -> float:
    return float(np.max(np.abs(M.T @ M - np.eye(M.shape[1]))))


def fit_fold(H_A: np.ndarray, H_B: np.ndarray, *, rank: int = RANK) -> dict:
    """Construct the fold vector. Returns a record with status 'ok' or 'invalid' and a reason."""
    H_A = np.asarray(H_A, dtype=np.float64)
    H_B = np.asarray(H_B, dtype=np.float64)
    rec: dict = {"version": VERSION, "rank": rank, "n_A": int(H_A.shape[0]), "n_B": int(H_B.shape[0]),
                 "status": "invalid", "reason": None, "vector": None}

    def fail(reason: str) -> dict:
        rec["reason"] = reason
        return rec

    if H_A.ndim != 2 or H_B.ndim != 2 or H_A.shape[1] != H_B.shape[1]:
        return fail("shape")
    if not (np.isfinite(H_A).all() and np.isfinite(H_B).all()):
        return fail("nonfinite_state")
    if H_A.shape[0] < MIN_COUNT or H_B.shape[0] < MIN_COUNT:
        return fail("insufficient_count")
    D = H_A.shape[1]
    if rank > D:
        return fail("rank_exceeds_dim")
    rec["numerical_rank_A"] = int(np.linalg.matrix_rank(H_A))
    rec["numerical_rank_B"] = int(np.linalg.matrix_rank(H_B))
    if rec["numerical_rank_A"] < rank or rec["numerical_rank_B"] < rank:
        return fail("numerical_rank")
    mA, mB = moments(H_A), moments(H_B)
    C_A = mA["sum_outer"] / mA["n"]
    C_B = mB["sum_outer"] / mB["n"]
    C_A = 0.5 * (C_A + C_A.T)
    C_B = 0.5 * (C_B + C_B.T)
    rec["moment_hashes"] = {"A_sum": array_hash(mA["sum"]), "A_sum_outer": array_hash(mA["sum_outer"]),
                            "B_sum": array_hash(mB["sum"]), "B_sum_outer": array_hash(mB["sum_outer"])}
    wA, VA = _top(C_A)
    wB, VB = _top(C_B)
    rec["eig_A"], rec["eig_B"] = wA, wB
    for name, w in (("A", wA), ("B", wB)):
        lam1 = float(w[0])
        lam_r = float(w[rank - 1])
        lam_next = float(w[rank]) if w.shape[0] > rank else 0.0
        if not lam1 > 0 or not lam_r > REL_TOL * lam1:
            return fail(f"eigen_floor_{name}")
        if not (lam_r - lam_next) > REL_TOL * lam1:
            return fail(f"eigen_gap_{name}")
    U_A, U_B = VA[:, :rank], VB[:, :rank]
    rec["U_A"], rec["U_B"] = U_A, U_B
    errs = {"U_A": _ortho_err(U_A), "U_B": _ortho_err(U_B)}
    P, sig_raw, Qt = np.linalg.svd(U_A.T @ U_B, full_matrices=False)
    sigma = np.clip(sig_raw, 0.0, 1.0)
    a = U_A @ P
    b = U_B @ Qt.T
    errs["a"], errs["b"] = _ortho_err(a), _ortho_err(b)
    errs["principal_pairs"] = float(np.max(np.abs(a.T @ b - np.diag(sigma))))
    rec["ortho_errors"] = errs
    rec["sigma_raw"], rec["sigma"], rec["P"], rec["Q"], rec["a"], rec["b"] = sig_raw, sigma, P, Qt.T, a, b
    if max(errs.values()) > ORTHO_TOL:
        return fail("orthonormality")
    E_A = np.einsum("di,de,ei->i", a, C_A, a)
    scores = E_A * (1.0 - sigma ** 2)
    rec["E_A"], rec["scores"] = E_A, scores
    i = int(np.argmax(scores))                 # lowest index on exact ties
    rec["selected_index"] = i
    scale = float(np.max(E_A))
    order = np.sort(scores)[::-1]
    rec["score_gap"] = float(order[0] - order[1]) if order.shape[0] > 1 else float(order[0])
    rec["sigma_isolation"] = float(np.min(np.abs(np.delete(sigma, i) - sigma[i]))) if rank > 1 else float("inf")
    if not float(scores[i]) > REL_TOL * scale:
        return fail("score_floor")
    if not rec["score_gap"] > REL_TOL * scale:
        return fail("score_tie")
    if not rec["sigma_isolation"] > REL_TOL:
        return fail("sigma_degenerate")
    mu_A = mA["sum"] / mA["n"]
    mu_B = mB["sum"] / mB["n"]
    contrast = mu_A - mu_B
    cn = float(np.linalg.norm(contrast))
    rec["mean_contrast_norm"] = cn
    v = a[:, i].copy()
    dot = float(v @ contrast)
    rec["sign_dot_raw"] = dot
    if not cn > REL_TOL:
        return fail("mean_contrast_zero")
    if not abs(dot) > REL_TOL * cn:
        return fail("sign_indeterminate")
    rec["sign_flipped"] = dot < 0
    if dot < 0:
        v = -v
    v = v / np.linalg.norm(v)
    v32 = v.astype(np.float32)
    rec["unit_error"] = abs(float(np.linalg.norm(v32.astype(np.float64))) - 1.0)
    if rec["unit_error"] > UNIT_TOL:
        return fail("unit_norm")
    rec["vector64"] = v
    rec["vector"] = v32
    rec["vector_sha256"] = array_hash(v32)
    rec["status"], rec["reason"] = "ok", None
    return rec


SCALAR_KEYS = ("version", "rank", "n_A", "n_B", "status", "reason", "numerical_rank_A", "numerical_rank_B",
               "moment_hashes", "ortho_errors", "selected_index", "score_gap", "sigma_isolation",
               "mean_contrast_norm", "sign_dot_raw", "sign_flipped", "unit_error", "vector_sha256")
ARRAY_KEYS = ("eig_A", "eig_B", "U_A", "U_B", "sigma_raw", "sigma", "P", "Q", "a", "b", "E_A", "scores",
              "vector64", "vector")


def summary(rec: dict) -> dict:
    """JSON-serializable scalar provenance of a fold record (arrays go to an npz)."""
    out = {k: rec[k] for k in SCALAR_KEYS if k in rec}
    for k in ("sigma", "E_A", "scores"):
        if k in rec:
            out[k] = [float(x) for x in rec[k]]
    for k in ("eig_A", "eig_B"):
        if k in rec:
            out[k + "_top33"] = [float(x) for x in rec[k][:RANK + 1]]
    if out.get("selected_index") is not None and "sigma" in rec:
        out["selected_sigma"] = float(rec["sigma"][out["selected_index"]])
        out["selected_score"] = float(rec["scores"][out["selected_index"]])
    return out


def arrays(rec: dict) -> dict:
    return {k: np.asarray(rec[k]) for k in ARRAY_KEYS if k in rec and rec[k] is not None}
