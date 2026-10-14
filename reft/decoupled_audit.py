#!/usr/bin/env python
"""P1-E -- decoupled conditioning (read U) / correction (write V) subspace feasibility. CPU only.

  freeze -- write-once record of protocol, config, code, test and launcher hashes
  audit  -- integrity of the P1 layer-31 and P1-D artifacts (P1-D loader re-checks every source hash)
  build  -- construction rows only: refit the P1-D FULL WHEN/HOW readers, verify reproduction, unscale, block-normalize,
            SVD -> frozen rank-4 U (p1e/conditioning_basis.pt). No audit row is touched.
  run    -- E1-E4: every probe/head/threshold fitted on construction rows and written to p1e/frozen_models.json, then a
            single audit scoring; frozen verdict (docs/reft/P1E_PROTOCOL.md).

The write basis is always the frozen P1 V. All gains are first-order diagnostics from saved q and h; no Whisper pass.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import art, read_json, write_json  # noqa: E402
from reft.conditional_audit import (cluster_bootstrap, fit_logistic, fit_ridge, load_all, nuisance_matrix,  # noqa: E402
                                    position_stratified_auc, rms_norm, unit)
from reft.provenance import environment, git_state, now, sha  # noqa: E402

CONFIG = ROOT / "configs/reft/p1e.yaml"
FREEZE_SOURCES = ["docs/reft/P1E_PROTOCOL.md", "configs/reft/p1e.yaml", "reft/decoupled_audit.py",
                  "tests/reft/test_decoupled_audit.py", "slurm/reft_p1e.sbatch"]


def ecfg() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def out(*p) -> Path:
    return art(ecfg()["output_dir"], *p)


# ====================================================================== pure helpers (unit-tested)

def unscale_linear(coef_std: np.ndarray, intercept_std, mean: np.ndarray, scale: np.ndarray):
    """y = coef_std . (x - mean)/scale + b  ==  coef_raw . x + b_raw."""
    coef_std = np.atleast_2d(coef_std)
    coef_raw = coef_std / scale[None, :]
    b_raw = np.atleast_1d(intercept_std) - (coef_std * (mean / scale)[None, :]).sum(1)
    return coef_raw, b_raw


def block_matrix(w_when: np.ndarray, W_how: np.ndarray) -> np.ndarray:
    """[w / ||w||_2 ; W / ||W||_F]: equal total energy for WHEN and HOW."""
    w = np.asarray(w_when, float).reshape(1, -1)
    return np.concatenate([w / np.linalg.norm(w), W_how / np.linalg.norm(W_how)], 0)


def top_right_singular(B: np.ndarray, k: int):
    _, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Vt[:k].T.copy()
    for j in range(k):
        if U[np.abs(U[:, j]).argmax(), j] < 0:
            U[:, j] = -U[:, j]
    return U, s


def principal_cosines(A: np.ndarray, B: np.ndarray) -> list:
    return np.linalg.svd(A.T @ B, compute_uv=False).tolist()


def first_order(hnorm, q, Vb, u, eps):
    """Per-row first-order margin change eps*||h|| * q^T V u (u: V coordinates, any norm)."""
    return eps * hnorm * ((q @ Vb) * u).sum(1)


def net_benefit(sel, gain):
    """NB = (sum of first-order margin changes over selected rows) / (all rows); P-row changes are signed (damage < 0)."""
    sel = np.asarray(sel, bool)
    return float(np.where(sel, gain, 0.0).sum() / len(sel))


def class_weights(y: np.ndarray) -> np.ndarray:
    nA, nP = int((y == 1).sum()), int((y == 0).sum())
    return np.where(y == 1, 0.5 / max(nA, 1), 0.5 / max(nP, 1))


def fit_joint_head(Z, T, w, lam, max_iter):
    """a = tanh(A z + b); minimize sum_i w_i ||a_i - T_i||^2 + lam ||A||_F^2 (float64 L-BFGS from zero)."""
    import torch
    Zt, Tt, wt = (torch.tensor(x, dtype=torch.float64) for x in (Z, T, w))
    A = torch.zeros(T.shape[1], Z.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(T.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([A, b], lr=1.0, max_iter=max_iter, tolerance_grad=1e-12, tolerance_change=1e-14,
                            history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (wt * ((torch.tanh(Zt @ A.T + b) - Tt) ** 2).sum(1)).sum() + lam * (A ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return A.detach().numpy(), b.detach().numpy()


def head_apply(Z, A, b):
    return np.tanh(Z @ A.T + b)


def weighted_loss(a, T, w):
    return float((w * ((a - T) ** 2).sum(1)).sum())


def verdict(integrity_ok, e2, e3, e4b):
    if not integrity_ok:
        return "P1E_INCONCLUSIVE"
    if e2 and e3 and e4b:
        return "P1E_DECOUPLED_FEASIBLE_EXPLORATORY"
    if e2:
        return "P1E_WHEN_ONLY"
    if e3:
        return "P1E_HOW_ONLY"
    return "P1E_NO_CONVINCING_SIGNAL"


# ====================================================================== joint head with construction-only CV

class JointHead:
    def __init__(self, mean, scale, A, b, lam, tau5, tau10, cv):
        self.mean, self.scale, self.A, self.b, self.lam = mean, scale, A, b, lam
        self.tau5, self.tau10, self.cv = tau5, tau10, cv

    def __call__(self, Z):
        return head_apply((Z - self.mean) / self.scale, self.A, self.b)

    def folded(self):
        """Equivalent 20 parameters acting on raw z: A' = A / scale, b' = b - A (mean/scale)."""
        return (self.A / self.scale[None, :]).tolist(), (self.b - self.A @ (self.mean / self.scale)).tolist()


def fit_joint(Z, T, y, g, grid, k, max_iter):
    from sklearn.model_selection import GroupKFold
    cv = GroupKFold(n_splits=k)
    res = []
    for lam in grid:
        oof = np.zeros_like(T)
        for tr, va in cv.split(Z, y, g):
            mu, sd = Z[tr].mean(0), Z[tr].std(0)
            A, b = fit_joint_head((Z[tr] - mu) / sd, T[tr], class_weights(y[tr]), lam, max_iter)
            oof[va] = head_apply((Z[va] - mu) / sd, A, b)
        res.append((weighted_loss(oof, T, class_weights(y)), lam, oof))
    loss, lam, oof = min(res, key=lambda r: (round(r[0], 12), -r[1]))
    mu, sd = Z.mean(0), Z.std(0)
    A, b = fit_joint_head((Z - mu) / sd, T, class_weights(y), lam, max_iter)
    mag = np.linalg.norm(oof, axis=1)[y == 0]
    return JointHead(mu, sd, A, b, lam, float(np.quantile(mag, 0.95, method="higher")),
                     float(np.quantile(mag, 0.90, method="higher")),
                     {"lambda": lam, "cv_oof_weighted_loss": loss, "grid": {str(r[1]): r[0] for r in res}})


# ====================================================================== commands

def run_freeze():
    rec = {"schema": "reft_p1e_freeze_v1", "utc": now(), "git": git_state(),
           "sources": {p: sha(ROOT / p) for p in FREEZE_SOURCES}}
    print(write_json(art("freeze/p1e.json"), rec))


def check_freeze():
    rec = read_json(art("freeze/p1e.json"))
    bad = [p for p, h in rec["sources"].items() if sha(ROOT / p) != h]
    if bad:
        raise SystemExit(f"P1-E sources changed since freeze: {bad}")
    return {"freeze_sha256": sha(art("freeze/p1e.json")), "freeze_git_head": rec["git"]["head"]}


def integrity():
    C = ecfg()
    data, checks, hashes, info = load_all()
    pm = read_json(art(C["p1d_dir"], "manifest.json"))
    checks["p1d_outputs_hash"] = all(sha(art(C["p1d_dir"], f)) == h for f, h in pm["outputs"].items())
    fr = read_json(art(C["p1d_dir"], "frozen_models.json"))["models"]
    exp = C["p1d_full_reader"]
    checks["p1d_full_hyperparameters"] = fr["d1"]["FULL"]["C"] == exp["when"]["C"] and fr["d2"]["FULL"]["alpha"] == exp["how"]["alpha"]
    checks["basis_sha_expected_p1e"] = hashes["basis"] == C["expected_basis_sha256"]
    return data, checks, hashes, info


def run_audit():
    fz = check_freeze()
    data, checks, hashes, info = integrity()
    ok = all(checks.values())
    write_json(out("audit.json"), {"schema": "reft_p1e_audit_v1", "utc": now(), **fz, "integrity_ok": ok, "checks": checks,
                                   "source_hashes": hashes, **info}, overwrite=True)
    print(json.dumps({"integrity_ok": ok, "checks": checks, **info}, indent=1, default=str))


def masks(data):
    meta = data["meta"]
    part = np.array([m["partition"] for m in meta])
    src = np.array([m["source"] for m in meta])
    dlg = np.array([m["dialogue"] for m in meta])
    return part, src, dlg


def run_build():
    t0 = time.time()
    C = ecfg()
    fz = check_freeze()
    data, checks, hashes, _ = integrity()
    assert all(checks.values()), checks
    part, src, dlg = masks(data)
    X = rms_norm(data["h"], C["norm"]["eps"])
    V = data["V"]
    exp, tol = C["p1d_full_reader"], C["p1d_full_reader"]["reproduction_tol"]
    con = part == "construction"
    # WHEN: construction A vs P (audit rows never enter this function's fits)
    iw = np.where(con & np.isin(src, ["A", "P"]))[0]
    yw = (src[iw] == "A").astype(int)
    when, wi = fit_logistic(X[iw], yw, dlg[iw], [exp["when"]["C"]], C["cv_folds"])
    # HOW: construction A rows -> c* in V
    ih = np.where(con & (src == "A"))[0]
    proj = data["q"] @ V
    cstar = unit(proj)
    how, hi = fit_ridge(X[ih], cstar[ih], dlg[ih], [exp["how"]["alpha"]], C["cv_folds"])
    repro = {"when_cv_oof_auroc": abs(wi["cv_oof_auroc"] - exp["when"]["cv_oof_auroc"]),
             "when_train_auroc": abs(wi["train_auroc_in_sample"] - exp["when"]["train_auroc_in_sample"]),
             "how_cv_oof_mean_cos": abs(hi["cv_oof_mean_cos"] - exp["how"]["cv_oof_mean_cos"])}
    repro_ok = all(v <= tol for v in repro.values())
    sw, lw = when[0], when[-1]
    w_raw, bw_raw = unscale_linear(lw.coef_, lw.intercept_, sw.mean_, sw.scale_)
    sh, lh = how[0], how[-1]
    W_raw, bh_raw = unscale_linear(lh.coef_, lh.intercept_, sh.mean_, sh.scale_)
    # unscaling check on construction rows: raw-coordinate readers reproduce the pipeline outputs
    unscale_err = {"when": float(np.abs(X[iw] @ w_raw[0] + bw_raw[0] - when.decision_function(X[iw])).max()),
                   "how": float(np.abs(X[ih] @ W_raw.T + bh_raw - how.predict(X[ih])).max())}
    B = block_matrix(w_raw[0], W_raw)
    U, s = top_right_singular(B, C["rank_u"])
    bchecks = {"orthonormal_max_dev": float(np.abs(U.T @ U - np.eye(C["rank_u"])).max()), "finite": bool(np.isfinite(U).all()),
               "shape": list(U.shape), "reader_reproduction_abs_err": repro, "reader_reproduction_ok": repro_ok,
               "unscale_max_abs_err": unscale_err, "unscale_ok": all(v < 1e-6 for v in unscale_err.values())}
    bchecks["ok"] = bchecks["orthonormal_max_dev"] < 1e-10 and bchecks["finite"] and repro_ok and bchecks["unscale_ok"] \
        and bchecks["shape"] == [V.shape[0], C["rank_u"]]
    import torch
    p = out("conditioning_basis.pt")
    assert not p.exists()
    torch.save({"U": torch.tensor(U), "B": torch.tensor(B), "singular_values": torch.tensor(s), "layer": 31,
                "rank": C["rank_u"], "write_basis_sha256": hashes["basis"]}, p)
    secs = time.time() - t0
    write_json(out("readout_provenance.json"), {
        "schema": "reft_p1e_readout_provenance_v1", "utc": now(), **fz, "git": git_state(),
        "when": {"model": "StandardScaler + LogisticRegression(lbfgs, balanced, max_iter 5000)", **wi,
                 "n_rows": int(len(iw)), "n_A": int(yw.sum()), "n_P": int((1 - yw).sum()),
                 "raw_norm": float(np.linalg.norm(w_raw)), "raw_intercept": float(bw_raw[0])},
        "how": {"model": "StandardScaler + Ridge", **hi, "n_rows": int(len(ih)),
                "raw_frobenius": float(np.linalg.norm(W_raw)), "raw_intercept": bh_raw.tolist()},
        "fitted_parameters": {"when": int(w_raw.size + 1), "how": int(W_raw.size + bh_raw.size),
                              "scaler_statistics": int(2 * X.shape[1] * 2)},
        "build_cpu_seconds": secs, "construction_only": True})
    write_json(out("conditioning_basis_manifest.json"), {
        "schema": "reft_p1e_basis_manifest_v1", "utc": now(), **fz, "basis_sha256": sha(p), "checks": bchecks,
        "B_singular_values": s.tolist(), "B_rank": int((s > 1e-10 * s[0]).sum()),
        "energy_rank4": float((s[:4] ** 2).sum() / (s ** 2).sum()),
        "construction": "B = [w_when/||w||_2 ; W_how/||W||_F] in raw RMSNorm(h) coordinates; U = top-4 right singular vectors; "
                        "sign: largest-|.| coordinate positive",
        "write_basis_sha256": hashes["basis"], "build_cpu_seconds": secs})
    print(json.dumps({"basis_sha256": sha(p), "checks": bchecks, "B_singular_values": s.tolist()}, indent=1))
    if not bchecks["ok"]:
        raise SystemExit("P1-E basis build FAILED (reader reproduction / unscaling / orthonormality)")


def run_main():
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    t0 = time.time()
    C = ecfg()
    G, BS, k, eps = C["gate"], C["bootstrap"], C["cv_folds"], C["epsilon"]
    fz = check_freeze()
    data, checks, hashes, info = integrity()
    bm = read_json(out("conditioning_basis_manifest.json"))
    checks["conditioning_basis_hash"] = sha(out("conditioning_basis.pt")) == bm["basis_sha256"]
    checks["conditioning_basis_checks"] = bm["checks"]["ok"]
    integrity_ok = all(checks.values())
    Ub = torch.load(out("conditioning_basis.pt"))["U"].numpy()
    meta = data["meta"]
    part, src, dlg = masks(data)
    X = rms_norm(data["h"], C["norm"]["eps"])
    V, q = data["V"], data["q"]
    hnorm = np.linalg.norm(data["h"], axis=1)
    seeds = [f"R{s}" for s in C["random_seeds"]]
    READ = {"U": Ub, "V": V, **{f"R{s}": R for s, R in data["R"].items()}}
    F = {n: X @ B_ for n, B_ in READ.items()}
    F["NUIS"] = nuisance_matrix(data["t"], data["duration"], data["lid"])
    F["U+NUIS"] = np.concatenate([F["U"], F["NUIS"]], 1)
    F["FULL"] = X
    proj = q @ V
    cstar = unit(proj)

    def rows(p, srcs):
        return np.where((part == p) & np.isin(src, srcs))[0]
    ic_ap, ia_ap = rows("construction", ["A", "P"]), rows("audit", ["A", "P"])
    yc, ya = (src[ic_ap] == "A").astype(int), (src[ia_ap] == "A").astype(int)
    ic_a, ia_a = rows("construction", ["A"]), rows("audit", ["A"])

    # ------------------------------------------------------------------ E1 (no audit labels used for fitting)
    def pcos(a, b):
        return principal_cosines(a, b)
    rp = read_json(out("readout_provenance.json"))
    zc, za = F["U"][part == "construction"], F["U"][part == "audit"]
    e1 = {"B_singular_values": bm["B_singular_values"], "energy_rank4": bm["energy_rank4"],
          "U_vs_V_principal_cosines": pcos(Ub, V), **{f"U_vs_{s}_principal_cosines": pcos(Ub, READ[s]) for s in seeds},
          "V_vs_R_mean_principal_cosine": float(np.mean([np.mean(pcos(V, READ[s])) for s in seeds])),
          "construction_to_audit_std_mean_shift": ((za.mean(0) - zc.mean(0)) / zc.std(0)).tolist(),
          "z_U_by_class_construction": {lab: {"mean": F["U"][ic_ap][yc == v].mean(0).tolist(), "std": F["U"][ic_ap][yc == v].std(0).tolist()}
                                        for lab, v in (("A", 1), ("P", 0))},
          "point_biserial_construction": [float(np.corrcoef(F["U"][ic_ap][:, j], yc)[0, 1]) for j in range(Ub.shape[1])],
          "readout_provenance": {"when_C": rp["when"]["C"], "how_alpha": rp["how"]["alpha"]}}

    # ------------------------------------------------------------------ fit everything on construction (frozen)
    M_when, F_when = {}, {}
    for n in ["U", "V", *seeds, "NUIS", "U+NUIS"]:
        M_when[n], F_when[n] = fit_logistic(F[n][ic_ap], yc, dlg[ic_ap], C["when_C_grid"], k)
    M_when["FULL"], F_when["FULL"] = fit_logistic(X[ic_ap], yc, dlg[ic_ap], [C["p1d_full_reader"]["when"]["C"]], k)
    M_how, F_how = {}, {}
    for n in ["U", "V", *seeds]:
        M_how[n], F_how[n] = fit_ridge(F[n][ic_a], cstar[ic_a], dlg[ic_a], C["how_alpha_grid"], k)
    M_how["FULL"], F_how["FULL"] = fit_ridge(X[ic_a], cstar[ic_a], dlg[ic_a], [C["p1d_full_reader"]["how"]["alpha"]], k)
    cbar = unit(cstar[ic_a].mean(0))
    Tc = np.where(yc[:, None] == 1, cstar[ic_ap], 0.0)
    heads = {n: fit_joint(F[n][ic_ap], Tc, yc, dlg[ic_ap], C["joint"]["lambda_grid"], k, C["joint"]["lbfgs_max_iter"])
             for n in ["U", "V", *seeds]}
    write_json(out("frozen_models.json"), {
        "schema": "reft_p1e_frozen_v1", "utc": now(), **fz, "note": "construction-only fits; written before audit scoring",
        "when": {n: {kk: v for kk, v in d.items()} for n, d in F_when.items()},
        "how": F_how, "const_direction": cbar.tolist(),
        "joint_heads": {n: {**h.cv, "tau_fpr05": h.tau5, "tau_fpr10": h.tau10, "A_folded": h.folded()[0], "b_folded": h.folded()[1]}
                        for n, h in heads.items()},
        "coef_when": {n: (m[-1].coef_.tolist(), m[-1].intercept_.tolist()) for n, m in M_when.items() if n != "FULL"},
        "coef_how": {n: (m[-1].coef_.tolist(), m[-1].intercept_.tolist()) for n, m in M_how.items() if n != "FULL"}})
    frozen_sha = sha(out("frozen_models.json"))

    # ------------------------------------------------------------------ E2 WHEN (single audit scoring)
    def auc(y, s):
        return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None
    sc = {n: m.decision_function(X[ia_ap] if n == "FULL" else F[n][ia_ap]) for n, m in M_when.items()}
    e2 = {"counts": {"A": int(ya.sum()), "P": int((1 - ya).sum()), "dialogues": len(set(dlg[ia_ap]))}}
    for n, s in sc.items():
        fr = F_when[n]
        r = {"auroc": auc(ya, s), "average_precision": float(average_precision_score(ya, s)), "prevalence": float(ya.mean()),
             "cv_C": fr["C"], "cv_oof_auroc": fr["cv_oof_auroc"], "train_auroc": fr["train_auroc_in_sample"]}
        for key in ("tau_fpr05", "tau_fpr10"):
            sel = s >= fr[key]
            r[key] = {"recall": float(sel[ya == 1].mean()), "achieved_fpr": float(sel[ya == 0].mean())}
        sb = s >= fr["tau_bal"]
        r["balanced_accuracy"] = float(0.5 * (sb[ya == 1].mean() + (~sb[ya == 0]).mean()))
        e2[n] = r
    gA = dlg[ia_ap]
    sU = sc["U"]
    e2["bootstrap"] = {
        "auroc_U": cluster_bootstrap(gA, lambda ii: auc(ya[ii], sU[ii]), BS["reps"], BS["seed"]),
        "auroc_U_minus_mean_random": cluster_bootstrap(gA, lambda ii: None if auc(ya[ii], sU[ii]) is None else
                                                       auc(ya[ii], sU[ii]) - np.mean([auc(ya[ii], sc[s][ii]) for s in seeds]), BS["reps"], BS["seed"]),
        "auroc_U_minus_NUIS": cluster_bootstrap(gA, lambda ii: None if auc(ya[ii], sU[ii]) is None else
                                                auc(ya[ii], sU[ii]) - auc(ya[ii], sc["NUIS"][ii]), BS["reps"], BS["seed"]),
        "auroc_U_minus_V": cluster_bootstrap(gA, lambda ii: None if auc(ya[ii], sU[ii]) is None else
                                             auc(ya[ii], sU[ii]) - auc(ya[ii], sc["V"][ii]), BS["reps"], BS["seed"])}
    e2["lodo_auroc_U"] = {d: auc(ya[gA != d], sU[gA != d]) for d in sorted(set(gA))}
    e2["position"] = {n: position_stratified_auc(ya, sc[n], data["t"][ia_ap], C["position_bins"]) for n in ["U", "V", "NUIS", "U+NUIS", *seeds]}
    tau = F_when["U"]["tau_fpr10"]
    e2["error_subgroup_recall_tau10"] = {}
    for lab, fn in (("CS", lambda m: m["stratum"] == "CS"), ("unit_EN", lambda m: m["lang"] == "EN"),
                    ("unit_ZH", lambda m: m["lang"] == "ZH"), ("ZH_ONLY", lambda m: m["stratum"] == "ZH_ONLY")):
        jj = [j for j, r in enumerate(ia_ap) if ya[j] == 1 and fn(meta[r])]
        e2["error_subgroup_recall_tau10"][lab] = {"n": len(jj), "recall": float(np.mean(sU[jj] >= tau)) if jj else None}

    # ------------------------------------------------------------------ E3 HOW (audit A rows; targets in V)
    TRANS = ("translation_like", "zh_to_latin")
    ceil = eps * hnorm[ia_a] * np.linalg.norm(proj[ia_a], axis=1)
    U_pred = {n: unit(m.predict(X[ia_a] if n == "FULL" else F[n][ia_a])) for n, m in M_how.items()}
    U_pred["CONST"] = np.tile(cbar, (len(ia_a), 1))
    e3 = {"n": int(len(ia_a)), "dialogues": len(set(dlg[ia_a])), "oracle_ceiling_mean_nats": float(ceil.mean())}
    cosd = {}
    for n, u in U_pred.items():
        cs = (u * cstar[ia_a]).sum(1)
        gn = first_order(hnorm[ia_a], q[ia_a], V, u, eps)
        cosd[n] = cs
        e3[n] = {"mean_cos": float(cs.mean()), "median_cos": float(np.median(cs)), "frac_pos": float((cs > 0).mean()),
                 "frac_neg": float((cs < 0).mean()), "mean_gain_nats": float(gn.mean()), "gain_over_ceiling": float(gn.mean() / ceil.mean()),
                 **({"cv_alpha": F_how[n]["alpha"], "cv_oof_mean_cos": F_how[n]["cv_oof_mean_cos"]} if n in F_how else {})}
    e3["random_read_max_mean_cos"] = max(e3[s]["mean_cos"] for s in seeds)
    g3 = dlg[ia_a]
    e3["bootstrap"] = {"mean_cos_U": cluster_bootstrap(g3, lambda ii: float(cosd["U"][ii].mean()), BS["reps"], BS["seed"]),
                       "mean_cos_U_minus_CONST": cluster_bootstrap(g3, lambda ii: float((cosd["U"][ii] - cosd["CONST"][ii]).mean()), BS["reps"], BS["seed"]),
                       "mean_cos_U_minus_V": cluster_bootstrap(g3, lambda ii: float((cosd["U"][ii] - cosd["V"][ii]).mean()), BS["reps"], BS["seed"])}
    e3["lodo_mean_cos_U_minus_CONST"] = {d: float((cosd["U"][g3 != d] - cosd["CONST"][g3 != d]).mean()) for d in sorted(set(g3))}
    grp = {"CS": lambda m: m["stratum"] == "CS", "translation_like": lambda m: m["category"] == "translation_like",
           "non_translation": lambda m: m["category"] not in TRANS, "en_lexical_substitution": lambda m: m["category"] == "en_lexical_substitution",
           "zh_lexical_substitution": lambda m: m["category"] == "zh_lexical_substitution",
           "unit_EN": lambda m: m["lang"] == "EN", "unit_ZH": lambda m: m["lang"] == "ZH"}
    e3["breakdown"] = {}
    for lab, fn in grp.items():
        jj = [j for j, r in enumerate(ia_a) if fn(meta[r])]
        e3["breakdown"][lab] = {"n": len(jj), "dialogues": len({g3[j] for j in jj}),
                                **{f"mean_cos_{n}": (float(cosd[n][jj].mean()) if jj else None) for n in ("U", "V", "CONST", "FULL")},
                                "frac_pos_U": float((cosd["U"][jj] > 0).mean()) if jj else None}
    scU = M_how["U"][0]
    Zc, Za = scU.transform(F["U"][ic_a]), scU.transform(F["U"][ia_a])
    nn = ((Za[:, None] - Zc[None]) ** 2).sum(-1).argmin(1)
    nnc = (cstar[ia_a] * cstar[ic_a][nn]).sum(1)
    e3["nearest_neighbour_target_cos_U"] = {"mean": float(nnc.mean()), "frac_negative": float((nnc < 0).mean())}

    # ------------------------------------------------------------------ E4 combined selectivity
    gainAP = lambda u: first_order(hnorm[ia_ap], q[ia_ap], V, u, eps)  # noqa: E731  (P rows: q_P = correct-margin grad)

    def summarize_sel(sel, u, label):
        gn = gainAP(u)
        cs = (u * cstar[ia_ap]).sum(1)
        sA, sP = sel & (ya == 1), sel & (ya == 0)
        r = {"A_selected": [int(sA.sum()), int((ya == 1).sum())], "P_selected": [int(sP.sum()), int((ya == 0).sum())],
             "recall": float(sA.sum() / (ya == 1).sum()), "achieved_fpr": float(sP.sum() / (ya == 0).sum()),
             "selected_A_cos_pos": int((cs[sA] > 0).sum()), "selected_A_cos_neg": int((cs[sA] < 0).sum()),
             "selected_A_frac_pos": float((cs[sA] > 0).mean()) if sA.any() else None,
             "selected_A_mean_gain_nats": float(gn[sA].mean()) if sA.any() else None,
             "selected_P_mean_change_nats": float(gn[sP].mean()) if sP.any() else None,
             "selected_P_frac_damage": float((gn[sP] < 0).mean()) if sP.any() else None,
             "net_benefit": net_benefit(sel, gn),
             "selected_A_dialogues": len(set(gA[sA])),
             "selected_A_nontranslation_pos": int(sum(1 for j in np.where(sA)[0] if cs[j] > 0 and meta[ia_ap[j]]["category"] not in TRANS)),
             "selected_A_by_category": dict(collections.Counter(meta[ia_ap[j]]["category"] for j in np.where(sA)[0])),
             "selected_P_by_lang": dict(collections.Counter(meta[ia_ap[j]]["lang"] for j in np.where(sP)[0]))}
        r["net_benefit_bootstrap"] = cluster_bootstrap(gA, lambda ii: net_benefit(sel[ii], gn[ii]), BS["reps"], BS["seed"])
        return r

    e4a = {}
    for n in ["U", "V"]:
        u = unit(M_how[n].predict(F[n][ia_ap]))
        for key in ("tau_fpr10", "tau_fpr05"):
            e4a[f"{n}@{key}"] = summarize_sel(sc[n] >= F_when[n][key], u, n)
    e4a["readout_parameters"] = {"when_probe": 5, "how_probe": 20, "total": 25}
    e4b = {}
    for n, h in heads.items():
        a = h(F[n][ia_ap])
        mag = np.linalg.norm(a, axis=1)
        u = unit(a)
        for key, tau_ in (("tau_fpr10", h.tau10), ("tau_fpr05", h.tau5)):
            e4b[f"{n}@{key}"] = summarize_sel(mag >= tau_, u, n)
        gm = first_order(hnorm[ia_ap], q[ia_ap], V, a, eps)  # actual magnitude, every row acted on
        e4b[f"{n}@magnitude_all_rows"] = {"A_mean_gain_nats": float(gm[ya == 1].mean()), "P_mean_change_nats": float(gm[ya == 0].mean()),
                                          "P_frac_damage": float((gm[ya == 0] < 0).mean()), "net_benefit": float(gm.mean()),
                                          "mean_norm_A": float(mag[ya == 1].mean()), "mean_norm_P": float(mag[ya == 0].mean())}
    cst = np.tile(cbar, (len(ia_ap), 1))
    e4b["CONST@all_rows"] = summarize_sel(np.ones(len(ia_ap), bool), cst, "CONST")
    e4b["CONST@U_head_tau_fpr10"] = summarize_sel(np.linalg.norm(heads["U"](F["U"][ia_ap]), axis=1) >= heads["U"].tau10, cst, "CONST")
    pd_ = {}
    aU = heads["U"](F["U"][ia_ap])
    selU = np.linalg.norm(aU, axis=1) >= heads["U"].tau10
    gnU = gainAP(unit(aU))
    for d in sorted(set(gA)):
        m_ = gA == d
        pd_[d] = {"A": int((m_ & (ya == 1)).sum()), "A_sel": int((m_ & selU & (ya == 1)).sum()), "P_sel": int((m_ & selU & (ya == 0)).sum()),
                  "net_benefit": net_benefit(selU[m_], gnU[m_])}
    e4b["U_per_dialogue_tau10"] = pd_
    e4b["fitted_parameters_per_head"] = 20

    # ------------------------------------------------------------------ decisions
    r2 = e2["U"]
    bs2 = e2["bootstrap"]
    pU = e2["position"]["U"]
    support = pU["supported_A"] >= G["e2_position_support_min_a"] and len(pU["bins_used"]) >= G["e2_position_support_min_bins"]
    e2c = {"auroc": r2["auroc"] >= G["e2_auroc_min"] and (bs2["auroc_U"]["ci95"] or [0])[0] > 0.5,
           "beyond_V_and_random": r2["auroc"] > e2["V"]["auroc"] and r2["auroc"] > max(e2[s]["auroc"] for s in seeds)
           and (bs2["auroc_U_minus_mean_random"]["ci95"] or [0])[0] > 0,
           "beyond_nuisance": (bs2["auroc_U_minus_NUIS"]["ci95"] or [0])[0] > 0,
           "recall_fpr10": r2["tau_fpr10"]["recall"] >= G["e2_recall_at_fpr10_min"]
           and r2["tau_fpr10"]["recall"] > max(e2[s]["tau_fpr10"]["recall"] for s in seeds),
           "lodo": all(v is not None and v >= G["e2_lodo_auroc_min"] for v in e2["lodo_auroc_U"].values()),
           "position": support and pU["stratified_auroc"] >= G["e2_position_stratified_auroc_min"]
           and e2["U+NUIS"]["auroc"] - e2["NUIS"]["auroc"] >= G["e2_incremental_over_nuisance_min"]}
    br = e3["breakdown"]
    e3c = {"beats_const": e3["U"]["mean_cos"] > e3["CONST"]["mean_cos"] and (e3["bootstrap"]["mean_cos_U_minus_CONST"]["ci95"] or [0])[0] > 0,
           "beats_random_read": e3["U"]["mean_cos"] > e3["random_read_max_mean_cos"],
           "beats_coupled_V": e3["U"]["mean_cos"] > e3["V"]["mean_cos"],
           "positive_fraction": e3["U"]["frac_pos"] >= G["e3_positive_fraction_min"],
           "lodo": all(v > 0 for v in e3["lodo_mean_cos_U_minus_CONST"].values()),
           "cs": (br["CS"]["mean_cos_U"] or -9) > (br["CS"]["mean_cos_CONST"] or 9),
           "non_translation": (br["non_translation"]["mean_cos_U"] or -9) > (br["non_translation"]["mean_cos_CONST"] or 9)}
    b10 = e4b["U@tau_fpr10"]
    e4c = {"J1_recall_fpr": b10["recall"] >= G["e4_recall_min"] and b10["achieved_fpr"] <= G["e4_achieved_fpr_max"],
           "J2_direction": (b10["selected_A_frac_pos"] or 0) >= G["e4_positive_fraction_min"] and (b10["selected_A_mean_gain_nats"] or -1) > 0,
           "J3_net_benefit": b10["net_benefit"] > 0 and (b10["net_benefit_bootstrap"]["ci95"] or [0])[0] > 0,
           "J4_beats_coupled_and_random": b10["net_benefit"] > e4b["V@tau_fpr10"]["net_benefit"]
           and b10["net_benefit"] > max(e4b[f"{s}@tau_fpr10"]["net_benefit"] for s in seeds)
           and b10["recall"] - e4b["V@tau_fpr10"]["recall"] >= G["e4_recall_gain_vs_coupled_min"],
           "J5_breadth": b10["selected_A_nontranslation_pos"] >= G["e4_min_nontranslation_positive"]
           and b10["selected_A_dialogues"] >= G["e4_min_dialogues"]}
    e2p, e3p, e4p = all(e2c.values()), all(e3c.values()), all(e4c.values())
    v = verdict(integrity_ok, e2p, e3p, e4p)
    rp = read_json(out("readout_provenance.json"))
    cost = {"U_construction_fitted_parameters": rp["fitted_parameters"], "U_build_cpu_seconds": rp["build_cpu_seconds"],
            "stored_U_coefficients": int(Ub.size), "stored_V_coefficients": int(V.size), "controller_parameters": 20,
            "artifact_coefficients_U_V_A_b": int(Ub.size + V.size + 20),
            "probe_parameters": {"when": 5, "how": 20, "joint_head": 20},
            "note": "supervised adaptation cost includes the 6,405 reader parameters fitted to build U, not only the 20 controller parameters"}
    common = {"utc": now(), **fz, "frozen_models_sha256": frozen_sha, "git": git_state()}
    for name, obj in (("e1_subspace.json", e1), ("e2_when.json", e2), ("e3_how.json", e3),
                      ("e4_joint_policy.json", {"E4A_separate": e4a, "E4B_joint_head": e4b})):
        write_json(out(name), {"schema": "reft_p1e_" + name.split(".")[0] + "_v1", **common, **obj})
    ev = {"schema": "reft_p1e_evaluation_v1", **common, "verdict": v, "integrity_ok": integrity_ok, "integrity": checks,
          "E2_criteria": e2c, "E2_pass": e2p, "E3_criteria": e3c, "E3_pass": e3p, "E4B_criteria": e4c, "E4B_pass": e4p,
          "cost": cost, "run_cpu_seconds": time.time() - t0,
          "note": "exploratory: the audit dialogues were previously examined in P1/P1-D; not a confirmation"}
    write_json(out("evaluation.json"), ev)
    write_json(out("manifest.json"), {"schema": "reft_p1e_manifest_v1", **common, "env": environment(), "source_hashes": hashes,
                                      "outputs": {f: sha(out(f)) for f in ("readout_provenance.json", "conditioning_basis.pt",
                                                                           "conditioning_basis_manifest.json", "frozen_models.json",
                                                                           "e1_subspace.json", "e2_when.json", "e3_how.json",
                                                                           "e4_joint_policy.json", "evaluation.json")},
                                      "verdict": v})
    print(json.dumps({k: ev[k] for k in ("verdict", "integrity_ok", "E2_criteria", "E2_pass", "E3_criteria", "E3_pass",
                                         "E4B_criteria", "E4B_pass", "run_cpu_seconds")}, indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "audit", "build", "run"])
    a = ap.parse_args(argv)
    {"freeze": run_freeze, "audit": run_audit, "build": run_build, "run": run_main}[a.cmd]()


if __name__ == "__main__":
    main()
