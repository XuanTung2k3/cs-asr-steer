#!/usr/bin/env python
"""P1-C -- held-out training-audit feasibility of the frozen basis (audit partition only).

  leverage -- GPU: oracle local diagnostics. For each audit correction row (A, B) the projected-gradient update
              u_S = eps*||h||*P_S q/||P_S q|| for S in {V, R1..R5, PCA4, SCRIPT1} and the unprojected q (FULL ceiling);
              for each preservation row (P) the predefined perturbations +-eps*||h||*s_j of every basis/control vector,
              plus the mean-correction direction V*cbar (random analogue R*cbar_R). Applied at the row's site position
              only, in an otherwise unchanged teacher-forced float32 pass.
  analyze  -- CPU: Test 1 retention, Test 2 leverage, Test 3 preservation, Test 4 breakdown, dialogue-cluster
              bootstrap, frozen P1 verdict (P0P1_PROTOCOL.md section 7).

The reference-derived q is used only to estimate local controllability. This is not a deployable controller.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import art, sdir, cfg, read_json, read_jsonl, write_json  # noqa: E402
from reft.provenance import check_freeze, cublas_consistent, git_state, loaded_cuda_libs, now, sha, slurm, weights_hash  # noqa: E402

CHUNK = 48


def load_bases():
    import torch
    bm = read_json(sdir("basis_manifest.json"))
    assert bm["checks"]["ok"], "basis checks failed"
    assert sha(sdir("basis_rank4.pt")) == bm["basis_sha256"] and sha(sdir("controls.pt")) == bm["controls_sha256"]
    B = torch.load(sdir("basis_rank4.pt"))
    K = torch.load(sdir("controls.pt"))
    subs = {"V": B["V"], **{f"R{s}": R for s, R in K["random"].items()}, "PCA4": K["pca4"], "SCRIPT1": K["script1"]}
    cbar = {"V": B["V"] @ B["cbar"], **{f"R{s}": K["random"][s] @ c for s, c in K["cbar_random"].items()}}
    return subs, cbar, bm


# ====================================================================== GPU: local leverage and preservation

def perturbations(row, q, h, subs, cbar, eps_list):
    """(tag, eps, unit direction) list for one row; FULL only for correction rows."""
    import torch
    out = []
    hn = float(h.norm())
    if row["source"] in ("A", "B"):
        for name, S in subs.items():
            p = S @ (S.T @ q)
            if float(p.norm()) == 0:
                continue
            for e in eps_list:
                out.append((name, e, e * hn * p / p.norm()))
        for e in eps_list:
            out.append(("FULL", e, e * hn * q / q.norm()))
    else:
        for name, S in subs.items():
            for j in range(S.shape[1]):
                for sg in (1, -1):
                    for e in eps_list:
                        out.append((f"{name}/{j}{'+' if sg > 0 else '-'}", e, sg * e * hn * S[:, j]))
        for name, c in cbar.items():
            for e in eps_list:
                out.append((f"{name}/cbar", e, e * hn * c / c.norm()))
    return out


def run_leverage():
    import torch
    from reft.baseline_auto import load_model
    from reft.gradient_capture import P0, encode, masked, suppress_masks, tf_logits, to_float32
    C = cfg()
    P1 = C["p1"]
    fz = check_freeze(cfg()["p1"]["freeze_stage"])
    subs, cbar, bm = load_bases()
    meta = read_jsonl(sdir("gradients/rows.jsonl"))
    T = torch.load(sdir("gradients/tensors.pt"))
    panel = {r["utterance_id"]: r for r in read_json(art("p1/panel.json"))["rows"]}
    gm = read_json(sdir("gradient_manifest.json"))
    assert sha(Path(gm["decode_path"])) == gm["decode_sha256"]
    dec = {d["id"]: d for d in read_jsonl(Path(gm["decode_path"]))}
    tg = read_json(art("p1/targets.json"))["targets"]
    audit = [m for m in meta if m["partition"] == "audit"]
    b, env = load_model()
    w0 = weights_hash(b.model)
    to_float32(b)
    masks = suppress_masks(b, b.device)
    dev = b.device
    # directions are built on CPU in float64 (cuBLAS level-1/2 SIGFPE on this MIG slice, amendment r3)
    subs_d = {k: v.double() for k, v in subs.items()}
    cbar_d = {k: v.double() for k, v in cbar.items()}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    by_utt = collections.defaultdict(list)
    for m in audit:
        by_utt[m["id"]].append(m)
    corr, pres = [], []
    for uid, rows in by_utt.items():
        enc = encode(b, panel[uid]["audio_path"])
        ids = tg[uid]["target_ids"]
        Xfull = torch.tensor([dec[uid]["prompt"] + ids[:-1]], device=dev)
        for m in rows:
            q, h = T["q"][m["row"]].double(), T["h"][m["row"]].double()
            pos, t = m["pos"], m["t"]
            X = Xfull[:, :pos + 1]
            items = [("BASE", 0.0, torch.zeros_like(q))] + perturbations(m, q, h, subs_d, cbar_d, P1["epsilons"])
            res = {}
            for i in range(0, len(items), CHUNK):
                ch = items[i:i + CHUNK]
                Xb = X.expand(len(ch), -1).contiguous()
                delta = torch.zeros(len(ch), X.shape[1], b.d_model, device=dev)
                for j, (_, _, u) in enumerate(ch):
                    delta[j, pos] = u.to(dev, torch.float32)
                with torch.no_grad():
                    lg, _ = tf_logits(b, enc, Xb, delta=delta)
                z = lg[:, pos].float()
                for j, (tag, e, _) in enumerate(ch):
                    zz = masked(z[j], t, masks)
                    c, k = m["correct"], m["competitor"]
                    top = int(zz.argmax())
                    other = zz.clone()
                    other[c] = -torch.inf
                    gb = float(zz[c] - other.max())  # -inf when the correct token is a Whisper-suppressed token
                    res[f"{tag}@{e}"] = {"m": float(z[j, c] - z[j, k]), "correct_argmax": top == c,
                                         "gold_vs_best_other": gb if gb == gb and abs(gb) != float("inf") else None,
                                         "correct_suppressed": bool(zz[c] == -torch.inf)}
            base = res["BASE@0.0"]
            rec = {"row": m["row"], "id": uid, "dialogue": m["dialogue"], "source": m["source"], "category": m["category"],
                   "stratum": m["stratum"], "lang": m["lang"], "m0_capture": m["m0"], "m0_eval": base["m"],
                   "m0_reproduced": abs(base["m"] - m["m0"]) <= 1e-3 + 1e-3 * abs(m["m0"]), "h_norm": m["h_norm"],
                   "results": res}
            (corr if m["source"] in ("A", "B") else pres).append(rec)
    w1 = weights_hash(b.model)
    libs = loaded_cuda_libs()
    assert cublas_consistent(libs), libs
    common = {"utc": now(), **fz, "git": git_state(), "slurm": slurm(), "basis_sha256": bm["basis_sha256"],
              "epsilons": P1["epsilons"], "primary_epsilon": P1["primary_epsilon"]}
    write_json(sdir("local_leverage.json"), {"schema": "reft_p1_local_leverage_v1", **common, "rows": corr})
    write_json(sdir("preservation_audit.json"), {"schema": "reft_p1_preservation_v1", **common, "rows": pres})
    man = {"schema": "reft_p1_eval_gpu_v1", **common, "env": env, "cuda_libs": libs, "weights_before": w0, "weights_after_bf16_cast": w1,
           "weights_unchanged": w0 == w1, "seconds": time.time() - t0, "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
           "n_correction_rows": len(corr), "n_preservation_rows": len(pres),
           "m0_reproduced": sum(r["m0_reproduced"] for r in corr + pres)}
    write_json(sdir("manifest_eval.json"), man)
    print(json.dumps({k: v for k, v in man.items() if k != "env"}, indent=1, default=str))


# ====================================================================== CPU: analysis and verdict

def boot_ci(rows, fn, reps, seed):
    """Dialogue-cluster bootstrap: 2.5% / 97.5% quantiles of fn(resampled rows)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["dialogue"]].append(r)
    dl = sorted(by)
    vals = []
    for _ in range(reps):
        pick = rng.choice(dl, len(dl))
        rs = [r for d in pick for r in by[d]]
        v = fn(rs)
        if v is not None:
            vals.append(v)
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] if vals else None


def mean(v):
    v = list(v)
    return sum(v) / len(v) if v else None


def run_analyze():
    import numpy as np
    import torch
    C = cfg()
    P1, GT = C["p1"], C["p1_gate"]
    BS = C["evaluation"]["bootstrap"]
    fz = check_freeze(cfg()["p1"]["freeze_stage"])
    subs, cbar, bm = load_bases()
    seeds = [f"R{s}" for s in P1["random_seeds"]]
    meta = read_jsonl(sdir("gradients/rows.jsonl"))
    T = torch.load(sdir("gradients/tensors.pt"))
    Q = T["q"].double()
    rho = {}
    for name, S in subs.items():
        rho[name] = (((Q @ S) ** 2).sum(1) / (Q ** 2).sum(1)).tolist()
    lev = read_json(sdir("local_leverage.json"))["rows"]
    prs = read_json(sdir("preservation_audit.json"))["rows"]
    eps = P1["primary_epsilon"]
    k, d = P1["rank"], Q.shape[1]
    thr = GT["retention_min_multiple_of_random_expectation"] * k / d

    def dm(r, tag, e=eps):
        return r["results"][f"{tag}@{e}"]["m"] - r["m0_eval"]

    for r in lev:
        r["rho"] = {n: rho[n][r["row"]] for n in subs}

    def summarize(rows, e=eps):
        if not rows:
            return {"n": 0}
        names = list(subs) + ["FULL"]
        out = {"n": len(rows), "dialogues": len({r["dialogue"] for r in rows}), "utterances": len({r["id"] for r in rows}),
               "m0_median": float(np.median([r["m0_eval"] for r in rows])),
               "m0_negative": sum(r["m0_eval"] < 0 for r in rows)}
        out["retention"] = {n: {"mean": mean(r["rho"][n] for r in rows), "median": float(np.median([r["rho"][n] for r in rows])),
                                "q10": float(np.quantile([r["rho"][n] for r in rows], 0.1)),
                                "q90": float(np.quantile([r["rho"][n] for r in rows], 0.9))} for n in subs}
        out["retention"]["random_expectation"] = k / d
        out["retention"]["random_max_seed_mean"] = max(out["retention"][s]["mean"] for s in seeds)
        out["retention"]["random_mean_of_seeds"] = mean(out["retention"][s]["mean"] for s in seeds)
        diff = sorted((r["rho"]["V"] - mean(r["rho"][s] for s in seeds) for r in rows), reverse=True)
        out["retention"]["top5_share_of_V_minus_random"] = sum(diff[:5]) / sum(diff) if sum(diff) > 0 else None
        out["leverage"] = {}
        for ee in P1["epsilons"]:
            L = {}
            for n in names:
                x = [dm(r, n, ee) for r in rows]
                neg = [r for r in rows if r["m0_eval"] <= 0]
                notarg = [r for r in rows if not r["results"]["BASE@0.0"]["correct_argmax"]]
                L[n] = {"mean_dm": mean(x), "median_dm": float(np.median(x)), "frac_improved": mean(v > 0 for v in x),
                        "correct_beats_competitor_among_m0_le_0": (sum(r["results"][f"{n}@{ee}"]["m"] > 0 for r in neg), len(neg)),
                        "correct_becomes_argmax": (sum(r["results"][f"{n}@{ee}"]["correct_argmax"] for r in notarg), len(notarg))}
            L["random_max_seed_mean_dm"] = max(L[s]["mean_dm"] for s in seeds)
            L["random_mean_of_seeds_dm"] = mean(L[s]["mean_dm"] for s in seeds)
            L["V_share_of_FULL"] = L["V"]["mean_dm"] / L["FULL"]["mean_dm"] if L["FULL"]["mean_dm"] else None
            out["leverage"][str(ee)] = L
        # bootstrap (dialogue clusters): V minus max-seed random
        f_rho = lambda rs: mean(r["rho"]["V"] for r in rs) - max(mean(r["rho"][s] for r in rs) for s in seeds)  # noqa: E731
        f_dm = lambda rs: mean(dm(r, "V") for r in rs) - max(mean(dm(r, s) for r in rs) for s in seeds)  # noqa: E731
        out["bootstrap"] = {"rho_V_minus_maxseed": {"point": f_rho(rows), "ci95": boot_ci(rows, f_rho, BS["reps"], BS["seed"])},
                            "dm_V_minus_maxseed": {"point": f_dm(rows), "ci95": boot_ci(rows, f_dm, BS["reps"], BS["seed"])},
                            "rho_V_mean": {"ci95": boot_ci(rows, lambda rs: mean(r["rho"]["V"] for r in rs), BS["reps"], BS["seed"])},
                            "dm_V_mean": {"ci95": boot_ci(rows, lambda rs: mean(dm(r, "V") for r in rs), BS["reps"], BS["seed"])}}
        by = collections.defaultdict(list)
        for r in rows:
            by[r["dialogue"]].append(r)
        out["per_dialogue"] = {dd: {"n": len(rr), "rho_V": mean(r["rho"]["V"] for r in rr),
                                    "rho_R_mean": mean(mean(r["rho"][s] for s in seeds) for r in rr),
                                    "dm_V": mean(dm(r, "V") for r in rr), "dm_R_mean": mean(mean(dm(r, s) for s in seeds) for r in rr)}
                               for dd, rr in sorted(by.items())}
        return out

    A = [r for r in lev if r["source"] == "A"]
    Bq = [r for r in lev if r["source"] == "B"]
    TRANS = ("translation_like", "zh_to_latin")
    groups = {"A_all": A, "A_CS": [r for r in A if r["stratum"] == "CS"],
              "A_CS_non_translation": [r for r in A if r["stratum"] == "CS" and r["category"] not in TRANS],
              "A_CS_english": [r for r in A if r["stratum"] == "CS" and r["lang"] == "EN"],
              "A_EN_ONLY": [r for r in A if r["stratum"] == "EN_ONLY"], "A_ZH_ONLY": [r for r in A if r["stratum"] == "ZH_ONLY"],
              "B_all": Bq, "B_CS": [r for r in Bq if r["stratum"] == "CS"]}
    for cat in sorted({r["category"] for r in A}):
        groups[f"A_cat:{cat}"] = [r for r in A if r["category"] == cat]
    for cat in sorted({r["category"] for r in Bq}):
        groups[f"B_cat:{cat}"] = [r for r in Bq if r["category"] == cat]
    S = {g: summarize(rs) for g, rs in groups.items()}

    # generalization: construction (in-sample) vs audit A rows
    con_a = [m["row"] for m in meta if m["partition"] == "construction" and m["source"] == "A"]
    in_sample = mean(rho["V"][i] for i in con_a)
    gen_ratio = S["A_all"]["retention"]["V"]["mean"] / in_sample if in_sample else None

    # preservation (Test 3)
    pres_names = list(subs)
    P = {}
    for ee in P1["epsilons"]:
        out = {}
        for lang in ("ALL", "EN", "ZH"):
            rr = [r for r in prs if lang == "ALL" or r["lang"] == lang]
            if not rr:
                continue
            o = {"n": len(rr), "dialogues": len({r["dialogue"] for r in rr})}
            for n in pres_names:
                keys = [kk for kk in rr[0]["results"] if kk.startswith(f"{n}/") and kk.endswith(f"@{ee}") and "cbar" not in kk]
                flip = [any(not r["results"][kk]["correct_argmax"] for kk in keys) for r in rr]
                worst = [min(r["results"][kk]["gold_vs_best_other"] - r["results"]["BASE@0.0"]["gold_vs_best_other"] for kk in keys) for r in rr]
                o[n] = {"worst_of_dirs_flip_fraction": mean(flip), "n_dirs": len(keys), "mean_worst_margin_change": mean(worst)}
            for n in cbar:
                kk = f"{n}/cbar@{ee}"
                o[f"{n}_cbar"] = {"flip_fraction": mean(not r["results"][kk]["correct_argmax"] for r in rr),
                                  "mean_margin_change": mean(r["results"][kk]["gold_vs_best_other"] - r["results"]["BASE@0.0"]["gold_vs_best_other"] for r in rr)}
            o["random_mean_of_seeds_flip"] = mean(o[s]["worst_of_dirs_flip_fraction"] for s in seeds)
            o["random_max_seed_flip"] = max(o[s]["worst_of_dirs_flip_fraction"] for s in seeds)
            o["base_margin_median"] = float(np.median([r["results"]["BASE@0.0"]["gold_vs_best_other"] for r in rr]))
            out[lang] = o
        P[str(ee)] = out
    pres_rho = {}
    for lang in ("ALL", "EN", "ZH"):
        rows_ = [r["row"] for r in prs if lang == "ALL" or r["lang"] == lang]
        if rows_:
            pres_rho[lang] = {n: mean(rho[n][i] for i in rows_) for n in subs}

    # integrity
    smokes = sorted(sdir().glob("smoke_*.json"), key=lambda p: p.stat().st_mtime)
    cap = read_json(sdir("manifest_capture.json"))
    ev = read_json(sdir("manifest_eval.json"))
    integ = {"smoke_all_pass": bool(smokes) and read_json(smokes[-1])["all_pass"], "capture_weights_unchanged": cap["weights_unchanged"],
             "capture_weights_expected": cap["weights_equal_expected"], "eval_weights_unchanged": ev["weights_unchanged"],
             "basis_checks_ok": bm["checks"]["ok"], "m0_reproduced_all": ev["m0_reproduced"] == len(lev) + len(prs),
             "audio_sha_ok": cap["audio_sha_ok"] == cap["n_panel"]}
    a = S["A_all"]
    cs = S["A_CS"]
    cov = {"audit_A_rows": a["n"], "audit_A_dialogues": a.get("dialogues", 0), "audit_A_CS_rows": cs["n"]}
    coverage_ok = cov["audit_A_rows"] >= GT["min_audit_a_examples"] and cov["audit_A_dialogues"] >= GT["min_audit_a_dialogues"] \
        and cov["audit_A_CS_rows"] >= GT["min_audit_cs_a_examples"]
    gates = {"integrity": all(integ.values()), "coverage": coverage_ok}
    if coverage_ok:
        pe = str(eps)
        La, Lc = a["leverage"][pe], cs["leverage"][pe]
        gates["R_retention"] = a["retention"]["V"]["mean"] >= thr and a["bootstrap"]["rho_V_minus_maxseed"]["ci95"][0] > 0
        gates["L_leverage"] = La["V"]["mean_dm"] > 0 and a["bootstrap"]["dm_V_minus_maxseed"]["ci95"][0] > 0 \
            and La["V"]["frac_improved"] >= GT["leverage_min_fraction_improved"]
        cs_ok = cs["retention"]["V"]["mean"] >= thr and cs["retention"]["V"]["mean"] > cs["retention"]["random_max_seed_mean"] \
            and Lc["V"]["mean_dm"] > Lc["random_max_seed_mean_dm"] and Lc["V"]["frac_improved"] >= GT["leverage_min_fraction_improved"]
        if cs["dialogues"] >= 4:
            cs_ok = cs_ok and cs["bootstrap"]["dm_V_minus_maxseed"]["ci95"][0] > 0
        gates["CS_signal"] = cs_ok
        nt = S["A_CS_non_translation"]
        gates["not_narrow_non_translation"] = nt["n"] >= GT["narrow_min_non_translation_cs"] and \
            nt["leverage"][pe]["V"]["mean_dm"] > nt["leverage"][pe]["random_max_seed_mean_dm"]
        gates["not_narrow_language_direction"] = Lc["SCRIPT1"]["mean_dm"] < GT["language_direction_dominance"] * Lc["V"]["mean_dm"]
        gates["generalization"] = gen_ratio is not None and gen_ratio >= GT["generalization_min_ratio"]
        pp = P[pe]["ALL"]
        gates["preservation_not_concerning"] = pp["V"]["worst_of_dirs_flip_fraction"] <= max(
            GT["preservation_flip_abs"], GT["preservation_flip_vs_random"] * pp["random_mean_of_seeds_flip"])
    if not (gates["integrity"] and gates["coverage"]):
        verdict = "P1_INCONCLUSIVE_ENGINEERING_OR_DATA"
    elif not (gates["R_retention"] and gates["L_leverage"]):
        verdict = "P1_NO_USEFUL_DIRECTIONAL_SIGNAL"
    elif all(gates[g] for g in ("CS_signal", "not_narrow_non_translation", "not_narrow_language_direction",
                                "generalization", "preservation_not_concerning")):
        verdict = "P1_SUBSPACE_FEASIBLE"
    else:
        verdict = "P1_SUBSPACE_WEAK_OR_NARROW"
    out = {"schema": "reft_p1_evaluation_v1", "utc": now(), **fz, "git": git_state(), "verdict": verdict, "gates": gates,
           "integrity": integ, "coverage": cov, "thresholds": {**GT, "retention_threshold": thr, "primary_epsilon": eps},
           "generalization": {"construction_A_mean_rho_V": in_sample, "audit_A_mean_rho_V": a["retention"]["V"]["mean"] if a["n"] else None,
                              "ratio": gen_ratio},
           "groups": S, "preservation": P, "preservation_gradient_retention": pres_rho,
           "basis": {"sha256": bm["basis_sha256"], "policy": bm["policy"], "cos_V_with_script1": bm["cos_V_with_script1"]},
           "claim_boundary": "oracle local diagnostics on held-out training-audit rows; no free-decoding or deployable claim"}
    write_json(sdir("evaluation.json"), out)
    write_json(sdir("manifest.json"), {"schema": "reft_p1_manifest_v1", "utc": now(), **fz, "git": git_state(),
                                         "verdict": verdict, "layer": P1["layer"], "files": {str(f): sha(f) for f in (
                                             art("p1/data_audit.json"), art("p1/panel.json"), Path(read_json(sdir("gradient_manifest.json"))["decode_path"]),
                                             *[sdir(x) for x in ("gradient_manifest.json", "gradients/rows.jsonl", "gradients/tensors.pt",
                                                                 "basis_rank4.pt", "basis_manifest.json", "controls.pt", "svd_spectrum.json",
                                                                 "local_leverage.json", "preservation_audit.json", "evaluation.json",
                                                                 "manifest_capture.json", "manifest_eval.json")])}})
    brief = {g: {"n": S[g]["n"], "dialogues": S[g].get("dialogues"),
                 "rho_V": S[g]["retention"]["V"]["mean"] if S[g]["n"] else None,
                 "rho_Rmax": S[g]["retention"]["random_max_seed_mean"] if S[g]["n"] else None,
                 "dm_V": S[g]["leverage"][str(eps)]["V"]["mean_dm"] if S[g]["n"] else None,
                 "dm_Rmax": S[g]["leverage"][str(eps)]["random_max_seed_mean_dm"] if S[g]["n"] else None,
                 "dm_SCRIPT1": S[g]["leverage"][str(eps)]["SCRIPT1"]["mean_dm"] if S[g]["n"] else None,
                 "dm_FULL": S[g]["leverage"][str(eps)]["FULL"]["mean_dm"] if S[g]["n"] else None} for g in S}
    print(json.dumps({"verdict": verdict, "gates": gates, "integrity": integ, "coverage": cov,
                      "generalization": out["generalization"], "brief": brief,
                      "preservation_primary": P[str(eps)]}, indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["leverage", "analyze"])
    a = ap.parse_args(argv)
    {"leverage": run_leverage, "analyze": run_analyze}[a.cmd]()


if __name__ == "__main__":
    main()
