#!/usr/bin/env python
"""P1-D -- conditional controllability audit of the frozen layer-31 rank-4 basis (CPU only).

  freeze -- write-once record of protocol, config, code and test hashes (docs/reft/P1D_PROTOCOL.md)
  audit  -- integrity checks on the P1 layer-31 artifacts and controller-visible features z = V^T Norm(h)
  run    -- D1 (error-state separability), D2 (correction-coordinate predictability), combined selectivity, verdict.
            Every probe, scaler, hyperparameter and threshold is fitted on construction rows and written to
            p1d/frozen_models.json before any audit row is scored.

Predictor inputs are restricted to FEATURE_INPUTS: the site activation h (through z) and reference-free decoding
metadata. Labels, gradients, oracle coefficients, categories and IDs are supervision / grouping / reporting only.
No Whisper forward pass, no GPU, no change to V. Not a deployable controller.
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

from reft.common import art, read_json, read_jsonl, write_json  # noqa: E402
from reft.provenance import environment, git_state, now, sha  # noqa: E402

CONFIG = ROOT / "configs/reft/p1d.yaml"
FREEZE_SOURCES = ["docs/reft/P1D_PROTOCOL.md", "configs/reft/p1d.yaml", "reft/conditional_audit.py",
                  "tests/reft/test_conditional_audit.py", "slurm/reft_p1d.sbatch"]
# the only quantities a predictor may read (allowlist; checked in build_features)
FEATURE_INPUTS = ("h", "t", "duration_sec", "detected_language")
NUISANCE_NAMES = ("t", "log1p_t", "duration_sec", "t_per_sec", "lid_zh")


def pcfg() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def out(*p) -> Path:
    return art(pcfg()["output_dir"], *p)


# ====================================================================== pure helpers (unit-tested)

def rms_norm(h: np.ndarray, eps: float) -> np.ndarray:
    return h / np.sqrt((h * h).mean(axis=-1, keepdims=True) + eps)


def nuisance_matrix(t, dur, lid) -> np.ndarray:
    t = np.asarray(t, float)
    dur = np.asarray(dur, float)
    return np.stack([t, np.log1p(t), dur, t / dur, np.asarray([x == "<|zh|>" for x in lid], float)], 1)


def unit(x: np.ndarray, axis=-1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n > 0)


def fpr_threshold(neg_scores: np.ndarray, fpr: float) -> float:
    """Score threshold whose rate of negatives with score >= threshold is at most ``fpr`` (upper quantile)."""
    return float(np.quantile(neg_scores, 1 - fpr, method="higher"))


def best_balanced_threshold(y: np.ndarray, s: np.ndarray) -> float:
    best, thr = -1.0, float(s.max())
    for c in np.unique(s):
        p = s >= c
        ba = 0.5 * (p[y == 1].mean() + (~p[y == 0]).mean())
        if ba > best + 1e-12:
            best, thr = ba, float(c)
    return thr


def position_stratified_auc(y, s, t, bins):
    from sklearn.metrics import roc_auc_score
    tot_pairs, acc, sup_a, used = 0, 0.0, 0, []
    for lo, hi in bins:
        m = (t >= lo) & (t <= hi)
        na, npos = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        if na and npos:
            a = roc_auc_score(y[m], s[m])
            acc += a * na * npos
            tot_pairs += na * npos
            sup_a += na
            used.append({"bin": [lo, hi], "A": na, "P": npos, "auroc": float(a)})
    return {"stratified_auroc": acc / tot_pairs if tot_pairs else None, "supported_A": sup_a, "bins_used": used}


def cluster_bootstrap(groups, fn, reps, seed):
    """Dialogue-cluster bootstrap of fn(index array); resamples where fn returns None are skipped and counted."""
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)
    uniq = sorted(set(g.tolist()))
    idx = {u: np.where(g == u)[0] for u in uniq}
    vals, skipped = [], 0
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq))
        ii = np.concatenate([idx[u] for u in pick])
        v = fn(ii)
        if v is None:
            skipped += 1
        else:
            vals.append(v)
    ci = [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] if vals else None
    return {"ci95": ci, "n": len(vals), "skipped": skipped}


def verdict(integrity_ok: bool, d1_status: str, d2_pass: bool) -> str:
    if not integrity_ok:
        return "P1D_INCONCLUSIVE"
    if d1_status == "PASS":
        return "P1D_CONDITIONAL_INFORMATION_SIGNAL" if d2_pass else "P1D_ERROR_DETECTION_ONLY"
    if d2_pass:
        return "P1D_DIRECTION_ONLY"
    return "P1D_NO_CONVINCING_SIGNAL" if d1_status == "FAIL" else "P1D_INCONCLUSIVE"


# ====================================================================== probes (construction-only fitting)

def _groupkfold(groups, k):
    from sklearn.model_selection import GroupKFold
    return GroupKFold(n_splits=k)


def fit_logistic(X, y, g, grid, k):
    """Grouped-CV selection of C by pooled OOF AUROC (ties -> smaller C); refit on all rows; OOF-derived thresholds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cv = _groupkfold(g, k)
    res = []
    for C in grid:
        oof = np.zeros(len(y))
        for tr, va in cv.split(X, y, g):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
            m.fit(X[tr], y[tr])
            oof[va] = m.decision_function(X[va])
        res.append((roc_auc_score(y, oof), -C, C, oof))
    auc, _, C, oof = max(res, key=lambda r: (round(r[0], 12), r[1]))
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000)).fit(X, y)
    neg = oof[y == 0]
    return model, {"C": C, "cv_oof_auroc": float(auc), "cv_grid": {str(r[2]): float(r[0]) for r in res},
                   "tau_fpr05": fpr_threshold(neg, 0.05), "tau_fpr10": fpr_threshold(neg, 0.10),
                   "tau_bal": best_balanced_threshold(y, oof),
                   "train_auroc_in_sample": float(roc_auc_score(y, model.decision_function(X)))}


def fit_ridge(X, Y, g, grid, k):
    """Grouped-CV selection of alpha by mean OOF cosine(pred, target) (ties -> larger alpha); refit on all rows."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cv = _groupkfold(g, k)
    res = []
    for a in grid:
        oof = np.zeros_like(Y)
        for tr, va in cv.split(X, Y, g):
            m = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X[tr], Y[tr])
            oof[va] = m.predict(X[va])
        res.append((float((unit(oof) * Y).sum(1).mean()), a))
    cos, a = max(res, key=lambda r: (round(r[0], 12), r[1]))
    model = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X, Y)
    return model, {"alpha": a, "cv_oof_mean_cos": cos, "cv_grid": {str(r[1]): r[0] for r in res}}


# ====================================================================== data

def load_all():
    import torch
    C = pcfg()
    src = art(C["source_dir"])
    gm = read_json(src / "gradient_manifest.json")
    bm = read_json(src / "basis_manifest.json")
    files = {"rows": src / "gradients/rows.jsonl", "tensors": src / "gradients/tensors.pt", "basis": src / "basis_rank4.pt",
             "controls": src / "controls.pt", "panel": art("p1/panel.json"), "decode": Path(gm["decode_path"])}
    hashes = {k: sha(p) for k, p in files.items()}
    checks = {"rows_sha_matches_manifest": hashes["rows"] == gm["rows_sha256"],
              "tensors_sha_matches_manifest": hashes["tensors"] == gm["tensors_sha256"],
              "basis_sha_matches_manifest": hashes["basis"] == bm["basis_sha256"],
              "basis_sha_expected": hashes["basis"] == C["expected_basis_sha256"],
              "controls_sha_matches_manifest": hashes["controls"] == bm["controls_sha256"],
              "decode_sha_matches_manifest": hashes["decode"] == gm["decode_sha256"]}
    B = torch.load(files["basis"])
    K = torch.load(files["controls"])
    T = torch.load(files["tensors"])
    meta = read_jsonl(files["rows"])
    V = B["V"].double().numpy()
    R = {s: K["random"][s].double().numpy() for s in C["random_seeds"]}
    checks["layer_31"] = B["layer"] == C["expected_layer"] == gm.get("layer")
    checks["rank_4"] = B["rank"] == C["expected_rank"] == V.shape[1]
    checks["V_orthonormal"] = bool(np.abs(V.T @ V - np.eye(V.shape[1])).max() < 1e-10)
    N = len(meta)
    checks["row_order"] = [m["row"] for m in meta] == list(range(N)) and T["h"].shape[0] == N == T["q"].shape[0]
    panel = {r["utterance_id"]: r for r in read_json(files["panel"])["rows"]}
    checks["partitions_unchanged"] = all(panel[m["id"]]["partition"] == m["partition"] for m in meta)
    dl = {p: {m["dialogue"] for m in meta if m["partition"] == p} for p in ("construction", "audit")}
    checks["dialogues_disjoint"] = not (dl["construction"] & dl["audit"])
    cnt = {p: dict(collections.Counter(m["source"] for m in meta if m["partition"] == p)) for p in ("construction", "audit")}
    checks["counts_match"] = all(cnt[p].get(s, 0) == n for p, d in C["expected_counts"].items() for s, n in d.items())
    dec = {d["id"]: d for d in read_jsonl(files["decode"])}
    checks["A_competitor_is_auto_emitted_token"] = all(dec[m["id"]]["tokens"][m["t"]] == m["competitor"]
                                                       for m in meta if m["source"] == "A")
    checks["P_correct_is_argmax"] = all(m["correct_is_argmax"] for m in meta if m["source"] == "P")
    checks["B_competitor_is_argmax"] = all(m["competitor_is_argmax"] for m in meta if m["source"] == "B")
    data = {"meta": meta, "h": T["h"].double().numpy(), "q": T["q"].double().numpy(), "V": V, "R": R,
            "duration": np.array([panel[m["id"]]["duration_sec"] for m in meta]),
            "t": np.array([m["t"] for m in meta]), "lid": [m["language"] for m in meta]}
    return data, checks, hashes, {"counts": cnt, "dialogues": {p: len(v) for p, v in dl.items()}}


def build_features(data, inputs=FEATURE_INPUTS):
    assert tuple(inputs) == FEATURE_INPUTS, "predictor inputs must equal the frozen allowlist"
    C = pcfg()
    hn = rms_norm(data["h"], C["norm"]["eps"])
    F = {"V": hn @ data["V"], **{f"R{s}": hn @ R for s, R in data["R"].items()},
         "NUIS": nuisance_matrix(data["t"], data["duration"], data["lid"]), "FULL": hn}
    F["V+NUIS"] = np.concatenate([F["V"], F["NUIS"]], 1)
    return F


# ====================================================================== commands

def run_freeze():
    rec = {"schema": "reft_p1d_freeze_v1", "utc": now(), "git": git_state(),
           "sources": {p: sha(ROOT / p) for p in FREEZE_SOURCES}}
    print(write_json(art("freeze/p1d.json"), rec))


def check_freeze():
    rec = read_json(art("freeze/p1d.json"))
    bad = [p for p, h in rec["sources"].items() if sha(ROOT / p) != h]
    if bad:
        raise SystemExit(f"P1-D sources changed since freeze: {bad}")
    return {"freeze_sha256": sha(art("freeze/p1d.json")), "freeze_git_head": rec["git"]["head"]}


def run_audit():
    fz = check_freeze()
    data, checks, hashes, info = load_all()
    F = build_features(data)
    checks["features_finite"] = all(bool(np.isfinite(x).all()) for x in F.values())
    checks["feature_inputs_allowlist"] = list(FEATURE_INPUTS)
    ok = all(v for k, v in checks.items() if k != "feature_inputs_allowlist")
    write_json(out("feature_audit.json"), {"schema": "reft_p1d_feature_audit_v1", "utc": now(), **fz, "integrity_ok": ok,
                                          "checks": checks, "source_hashes": hashes, **info,
                                          "feature_dims": {k: int(v.shape[1]) for k, v in F.items()},
                                          "norm": pcfg()["norm"]}, overwrite=True)
    print(json.dumps({"integrity_ok": ok, "checks": checks, **info}, indent=1, default=str))
    return ok


def run_main():
    from sklearn.metrics import average_precision_score, roc_auc_score
    t0 = time.time()
    C = pcfg()
    G, BS, k = C["gate"], C["bootstrap"], C["cv_folds"]
    fz = check_freeze()
    data, checks, hashes, info = load_all()
    F = build_features(data)
    checks["features_finite"] = all(bool(np.isfinite(x).all()) for x in F.values())
    integrity_ok = all(checks.values())
    meta = data["meta"]
    part = np.array([m["partition"] for m in meta])
    src = np.array([m["source"] for m in meta])
    dlg = np.array([m["dialogue"] for m in meta])
    hnorm = np.linalg.norm(data["h"], axis=1)
    eps = C["d2"]["epsilon"]
    seeds = [f"R{s}" for s in C["random_seeds"]]

    def rows(p, srcs):
        return np.where((part == p) & np.isin(src, srcs))[0]

    # ------------------------------------------------------------------ fit everything on construction (frozen)
    frozen, models = {"d1": {}, "d1_B": {}, "d2": {}, "d2_B": {}}, {"d1": {}, "d1_B": {}, "d2": {}, "d2_B": {}}
    for tag, pos_src in (("d1", "A"), ("d1_B", "B")):
        ic = rows("construction", [pos_src, "P"])
        yc = (src[ic] == pos_src).astype(int)
        names = ["V", *seeds, "NUIS", "V+NUIS"] + (["FULL"] if tag == "d1" else [])
        for n in names:
            grid = C["d1"]["C_grid_full"] if n == "FULL" else C["d1"]["C_grid_low"]
            models[tag][n], frozen[tag][n] = fit_logistic(F[n][ic], yc, dlg[ic], grid, k)
    tol = C["d2"]["negligible_rel_norm"]
    q = data["q"]
    d2_targets, d2_excl = {}, {}
    for S in ["V", *seeds]:
        B_ = data["V"] if S == "V" else data["R"][int(S[1:])]
        proj = q @ B_
        rel = np.linalg.norm(proj, axis=1) / np.linalg.norm(q, axis=1)
        d2_targets[S] = (proj, unit(proj), rel >= tol)
    for tag, s_ in (("d2", "A"), ("d2_B", "B")):
        for n in ["V", *seeds, "FULL"]:
            S = "V" if n == "FULL" else n
            proj, c, ok_ = d2_targets[S]
            ic = rows("construction", [s_])
            ic = ic[ok_[ic]]
            d2_excl[f"{tag}/{n}/construction"] = int((~ok_[rows("construction", [s_])]).sum())
            grid = C["d2"]["alpha_grid_full"] if n == "FULL" else C["d2"]["alpha_grid_low"]
            models[tag][n], frozen[tag][n] = fit_ridge(F[n][ic], c[ic], dlg[ic], grid, k)
        ic = rows("construction", [s_])
        cbar = unit(d2_targets["V"][1][ic][d2_targets["V"][2][ic]].mean(0))
        frozen[tag]["CONST"] = {"cbar": cbar.tolist()}
    write_json(out("frozen_models.json"), {"schema": "reft_p1d_frozen_v1", "utc": now(), **fz,
                                          "note": "fitted on construction rows only; written before any audit row is scored",
                                          "models": frozen, "d2_excluded_negligible": d2_excl,
                                          "coef": {tag: {n: (m[-1].coef_.tolist(), np.atleast_1d(m[-1].intercept_).tolist())
                                                         for n, m in mm.items() if n != "FULL"} for tag, mm in models.items()}})
    frozen_sha = sha(out("frozen_models.json"))

    # ------------------------------------------------------------------ D1 audit scoring
    def d1_eval(tag, pos_src):
        ia = rows("audit", [pos_src, "P"])
        y = (src[ia] == pos_src).astype(int)
        g = dlg[ia]
        res, scores = {}, {}
        for n, m in models[tag].items():
            s = m.decision_function(F[n][ia])
            scores[n] = s
            fr = frozen[tag][n]
            r = {"auroc": float(roc_auc_score(y, s)), "average_precision": float(average_precision_score(y, s)),
                 "prevalence": float(y.mean())}
            for key in ("tau_fpr05", "tau_fpr10"):
                sel = s >= fr[key]
                r[key] = {"recall": float(sel[y == 1].mean()), "achieved_fpr": float(sel[y == 0].mean())}
            selb = s >= fr["tau_bal"]
            r["balanced_accuracy_tau_bal"] = float(0.5 * (selb[y == 1].mean() + (~selb[y == 0]).mean()))
            if n == "FULL":
                r["train_in_sample_auroc"] = fr["train_auroc_in_sample"]
            res[n] = r
        sv = scores["V"]

        def auc_on(ii, s):
            yy = y[ii]
            return float(roc_auc_score(yy, s[ii])) if 0 < yy.sum() < len(yy) else None
        res["bootstrap"] = {
            "auroc_V": cluster_bootstrap(g, lambda ii: auc_on(ii, sv), BS["reps"], BS["seed"]),
            "auroc_V_minus_mean_random": cluster_bootstrap(
                g, lambda ii: None if auc_on(ii, sv) is None else auc_on(ii, sv) - np.mean([auc_on(ii, scores[s]) for s in seeds]),
                BS["reps"], BS["seed"]),
            "auroc_V_minus_NUIS": cluster_bootstrap(
                g, lambda ii: None if auc_on(ii, sv) is None else auc_on(ii, sv) - auc_on(ii, scores["NUIS"]), BS["reps"], BS["seed"])}
        res["lodo_auroc_V"] = {d: auc_on(np.where(g != d)[0], sv) for d in sorted(set(g))}
        res["position"] = {"V": position_stratified_auc(y, sv, data["t"][ia], C["d1"]["position_bins"]),
                           **{n: position_stratified_auc(y, scores[n], data["t"][ia], C["d1"]["position_bins"])
                              for n in [*seeds, "NUIS", "V+NUIS"]}}
        res["position_distribution"] = {
            lab: {"median_t": float(np.median(data["t"][ia][y == v])), "q10_q90": [float(np.quantile(data["t"][ia][y == v], x)) for x in (0.1, 0.9)]}
            for lab, v in (("error", 1), ("correct", 0))}
        # sealed-prediction subgroup recall (labels used only after scoring)
        sub = {}
        tau = frozen[tag]["V"]["tau_fpr10"]
        for lab, fn in (("CS", lambda m: m["stratum"] == "CS"), ("unit_EN", lambda m: m["lang"] == "EN"),
                        ("unit_ZH", lambda m: m["lang"] == "ZH"), ("ZH_ONLY", lambda m: m["stratum"] == "ZH_ONLY")):
            ii = [j for j, r in enumerate(ia) if y[j] == 1 and fn(meta[r])]
            sub[lab] = {"n": len(ii), "recall_at_tau_fpr10": float(np.mean(sv[ii] >= tau)) if ii else None}
        res["error_subgroups"] = sub
        res["counts"] = {"errors": int(y.sum()), "correct": int((1 - y).sum()), "dialogues": len(set(g))}
        return res, sv, ia, y

    d1, sV, ia1, y1 = d1_eval("d1", "A")
    d1B, _, _, _ = d1_eval("d1_B", "B")

    # ------------------------------------------------------------------ D2 audit scoring
    def d2_eval(tag, s_):
        ia_all = rows("audit", [s_])
        out_, preds = {"excluded_negligible_audit": {}}, {}
        for n in ["V", *seeds, "FULL", "CONST"]:
            S = "V" if n in ("FULL", "CONST") else n
            proj, c, ok_ = d2_targets[S]
            ia = ia_all[ok_[ia_all]]
            out_["excluded_negligible_audit"][n] = int((~ok_[ia_all]).sum())
            if n == "CONST":
                u = np.tile(np.asarray(frozen[tag]["CONST"]["cbar"]), (len(ia), 1))
            else:
                u = unit(models[tag][n].predict(F[n][ia]))
            cos = (u * c[ia]).sum(1)
            gain = eps * hnorm[ia] * (proj[ia] * u).sum(1)
            ceil = eps * hnorm[ia] * np.linalg.norm(proj[ia], axis=1)
            preds[n] = (ia, cos, gain, ceil, u)
            out_[n] = {"n": int(len(ia)), "mean_cos": float(cos.mean()), "median_cos": float(np.median(cos)),
                       "frac_pos": float((cos > 0).mean()), "frac_neg": float((cos < 0).mean()),
                       "mean_gain_nats": float(gain.mean()), "mean_ceiling_nats": float(ceil.mean()),
                       "gain_over_ceiling": float(gain.mean() / ceil.mean())}
        out_["random_max_seed_mean_cos"] = max(out_[s]["mean_cos"] for s in seeds)
        iaV, cosV, gainV, _, _ = preds["V"]
        cosC = preds["CONST"][1]
        g = dlg[iaV]
        out_["bootstrap"] = {"mean_cos_V": cluster_bootstrap(g, lambda ii: float(cosV[ii].mean()), BS["reps"], BS["seed"]),
                             "mean_cos_V_minus_CONST": cluster_bootstrap(g, lambda ii: float((cosV[ii] - cosC[ii]).mean()),
                                                                         BS["reps"], BS["seed"])}
        out_["lodo_mean_cos_V_minus_CONST"] = {d: float((cosV[g != d] - cosC[g != d]).mean()) for d in sorted(set(g))}
        groups = {"CS": lambda m: m["stratum"] == "CS", "translation_like": lambda m: m["category"] == "translation_like",
                  "en_lexical_substitution": lambda m: m["category"] == "en_lexical_substitution",
                  "zh_lexical_substitution": lambda m: m["category"] == "zh_lexical_substitution",
                  "unit_EN": lambda m: m["lang"] == "EN", "unit_ZH": lambda m: m["lang"] == "ZH"}
        br = {}
        for lab, fn in groups.items():
            jj = [j for j, r in enumerate(iaV) if fn(meta[r])]
            br[lab] = {"n": len(jj), "dialogues": len({dlg[iaV[j]] for j in jj}),
                       "mean_cos_V": float(cosV[jj].mean()) if jj else None, "mean_cos_CONST": float(cosC[jj].mean()) if jj else None,
                       "frac_pos_V": float((cosV[jj] > 0).mean()) if jj else None,
                       "mean_gain_V": float(gainV[jj].mean()) if jj else None}
        out_["breakdown"] = br
        out_["per_dialogue"] = {d: {"n": int((g == d).sum()), "mean_cos_V": float(cosV[g == d].mean()),
                                    "mean_cos_CONST": float(cosC[g == d].mean())} for d in sorted(set(g))}
        # opposite-sign targets at similar projected states (nearest construction row in standardized z)
        proj, c, ok_ = d2_targets["V"]
        ic = rows("construction", [s_])
        ic = ic[ok_[ic]]
        sc = models[tag]["V"][0]
        Zc, Za = sc.transform(F["V"][ic]), sc.transform(F["V"][iaV])
        nn = ((Za[:, None, :] - Zc[None, :, :]) ** 2).sum(-1).argmin(1)
        nn_cos = (c[iaV] * c[ic][nn]).sum(1)
        out_["nearest_neighbour_target_cos"] = {"mean": float(nn_cos.mean()), "frac_negative": float((nn_cos < 0).mean())}
        return out_, preds

    d2, preds = d2_eval("d2", "A")
    d2B, _ = d2_eval("d2_B", "B")

    # ------------------------------------------------------------------ combined selectivity (offline)
    comb = {}
    for key in ("tau_fpr10", "tau_fpr05"):
        tau = frozen["d1"]["V"][key]
        sel = sV >= tau
        selA, selP = ia1[(y1 == 1) & sel], ia1[(y1 == 0) & sel]
        uA = unit(models["d2"]["V"].predict(F["V"][selA])) if len(selA) else np.zeros((0, 4))
        cA = d2_targets["V"][1][selA]
        cosA = (uA * cA).sum(1) if len(selA) else np.zeros(0)
        gA = eps * hnorm[selA] * (d2_targets["V"][0][selA] * uA).sum(1) if len(selA) else np.zeros(0)
        uP = unit(models["d2"]["V"].predict(F["V"][selP])) if len(selP) else np.zeros((0, 4))
        dP = eps * hnorm[selP] * ((q[selP] @ data["V"]) * uP).sum(1) if len(selP) else np.zeros(0)
        comb[key] = {"threshold": tau, "A_selected": [int(len(selA)), int((y1 == 1).sum())],
                     "P_selected": [int(len(selP)), int((y1 == 0).sum())],
                     "selected_A_cos_pos": int((cosA > 0).sum()), "selected_A_cos_neg": int((cosA < 0).sum()),
                     "selected_A_mean_gain_nats": float(gA.mean()) if len(gA) else None,
                     "selected_P_mean_correct_margin_change_nats": float(dP.mean()) if len(dP) else None,
                     "selected_P_frac_predicted_damage": float((dP < 0).mean()) if len(dP) else None,
                     "selected_A_by_dialogue": dict(collections.Counter(dlg[selA].tolist())),
                     "selected_A_by_category": dict(collections.Counter(meta[r]["category"] for r in selA)),
                     "selected_P_by_lang": dict(collections.Counter(meta[r]["lang"] for r in selP))}

    # ------------------------------------------------------------------ decision rules
    bsV = d1["bootstrap"]
    core = {"auroc": d1["V"]["auroc"] >= G["d1_auroc_min"] and (bsV["auroc_V"]["ci95"] or [0])[0] > 0.5,
            "beyond_random": d1["V"]["auroc"] > max(d1[s]["auroc"] for s in seeds)
            and (bsV["auroc_V_minus_mean_random"]["ci95"] or [0])[0] > 0,
            "recall_fpr10": d1["V"]["tau_fpr10"]["recall"] >= G["d1_recall_at_fpr10_min"]
            and d1["V"]["tau_fpr10"]["recall"] > max(d1[s]["tau_fpr10"]["recall"] for s in seeds),
            "lodo": all(v is not None and v >= G["d1_lodo_auroc_min"] for v in d1["lodo_auroc_V"].values())}
    pv = d1["position"]["V"]
    support = pv["supported_A"] >= G["d1_position_support_min_a"] and len(pv["bins_used"]) >= G["d1_position_support_min_bins"]
    pos_pass = support and pv["stratified_auroc"] is not None and pv["stratified_auroc"] >= G["d1_position_stratified_auroc_min"] \
        and d1["V+NUIS"]["auroc"] - d1["NUIS"]["auroc"] >= G["d1_incremental_over_nuisance_min"]
    d1_status = ("PASS" if pos_pass else "FAIL" if support else "CONFOUNDED_UNRESOLVED") if all(core.values()) else "FAIL"
    d2c = {"beats_const": d2["V"]["mean_cos"] > d2["CONST"]["mean_cos"]
           and (d2["bootstrap"]["mean_cos_V_minus_CONST"]["ci95"] or [0])[0] > 0,
           "beats_random": d2["V"]["mean_cos"] > d2["random_max_seed_mean_cos"],
           "positive_fraction": d2["V"]["frac_pos"] >= G["d2_positive_fraction_min"],
           "lodo": all(v > 0 for v in d2["lodo_mean_cos_V_minus_CONST"].values()),
           "cs_beats_const": (d2["breakdown"]["CS"]["mean_cos_V"] or -9) > (d2["breakdown"]["CS"]["mean_cos_CONST"] or 9)}
    d2_pass = all(d2c.values())
    v = verdict(integrity_ok, d1_status, d2_pass)
    common = {"utc": now(), **fz, "frozen_models_sha256": frozen_sha, "git": git_state()}
    write_json(out("d1_error_separability.json"), {"schema": "reft_p1d_d1_v1", **common, "primary_A_vs_P": d1,
                                                  "secondary_B_vs_P": d1B})
    write_json(out("d2_direction_predictability.json"), {"schema": "reft_p1d_d2_v1", **common, "primary_A": d2,
                                                        "secondary_B": d2B})
    write_json(out("combined_selectivity.json"), {"schema": "reft_p1d_combined_v1", **common, **comb})
    ev = {"schema": "reft_p1d_evaluation_v1", **common, "verdict": v, "integrity_ok": integrity_ok, "integrity": checks,
          "d1_core": core, "d1_position_support": support, "d1_position_pass": pos_pass, "d1_status": d1_status,
          "d2_criteria": d2c, "d2_pass": d2_pass, "seconds": time.time() - t0,
          "note": "exploratory; does not replace the sealed P1 verdict (P1_SUBSPACE_WEAK_OR_NARROW)"}
    write_json(out("evaluation.json"), ev)
    write_json(out("manifest.json"), {"schema": "reft_p1d_manifest_v1", **common, "env": environment(), "source_hashes": hashes,
                                      "outputs": {f: sha(out(f)) for f in ("frozen_models.json", "d1_error_separability.json",
                                                                           "d2_direction_predictability.json",
                                                                           "combined_selectivity.json", "evaluation.json")},
                                      "verdict": v, "cpu_seconds": time.time() - t0})
    print(json.dumps({k: ev[k] for k in ("verdict", "integrity_ok", "d1_core", "d1_position_support", "d1_position_pass",
                                         "d1_status", "d2_criteria", "d2_pass", "seconds")}, indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "audit", "run"])
    a = ap.parse_args(argv)
    {"freeze": run_freeze, "audit": run_audit, "run": run_main}[a.cmd]()


if __name__ == "__main__":
    main()
