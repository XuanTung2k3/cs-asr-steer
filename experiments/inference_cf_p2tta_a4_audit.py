#!/usr/bin/env python
"""Independent P2-TTA-A4 auditor. Does NOT import inference_cf_p2tta_a4_analyze, csasr.inference_cf.script_safe_tta,
soft_auto_tta or episodic_tta. Uses the canonical tokenizer partition primitive, the independent A3/TTA0 auditor
helpers and locked canonical metric primitives; owns class assignment, safe-teacher selection, D_E/D_M/D_ANCHOR,
R_E, rho, safety, rescue, benefit and label.

prerun -> PASS_TO_P2_TTA_A4 | BLOCK_BEFORE_P2_TTA_A4
live   -> ``live_safe_kl_check`` (runner, first row, theta0, no optimizer update)
post   -> P2_TTA_A4_AUDIT: PASS | BLOCK
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

import experiments.inference_cf_p2tta0_audit as tau
import experiments.inference_cf_p2tta_a3_audit as a3au

FREEZE = "37441eb"
CONFIG = "configs/inference_cf/p2_tta_a4.json"
PANEL = "docs/inference_cf/P2_TTA_A3_PANEL.json"
BASE = ROOT / "results/inference_cf/p2tta_a4"
PLAN_REL = "results/inference_cf/p2tta_a4/plan_sealed.json"


def _git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    return _git("ls-files", rel) == rel and tau.blob_sha("HEAD", rel) == sha(ROOT / rel)


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


def own_classes(y, partition) -> list[str]:
    E, M = frozenset(partition["embedded_ids"]), frozenset(partition["matrix_ids"])
    return [("E" if t in E else "M" if t in M else "O") for t in y]


# ---- live first-row check ---------------------------------------------------------------------------------------

def live_safe_kl_check(model, names, encoded, cA, cB, y, partition, *, suppress, begin, tokenizer) -> dict:
    import torch
    import torch.nn.functional as F
    params = dict(model.named_parameters())
    masters = [torch.nn.Parameter(params[n].detach().float().clone()) for n in names]
    dev = masters[0].device
    valid = torch.tensor(tau.own_valid(y, suppress, begin, tokenizer), dtype=torch.bool, device=dev)
    isE = torch.tensor([c == "E" for c in own_classes(y, partition)], dtype=torch.bool, device=dev)
    T = len(y)

    def masked(logits):
        q = logits[0].float()[3:3 + T]
        ban = torch.zeros_like(q, dtype=torch.bool)
        if suppress:
            ban[:, list(suppress)] = True
        if begin and T:
            ban[0, list(begin)] = True
        return q.masked_fill(ban, float("-inf")), ban
    with torch.no_grad():
        la_, ban = masked(model(encoder_outputs=encoded, decoder_input_ids=torch.tensor([list(cA) + list(y)], device=dev), use_cache=False).logits)
        lb_, _ = masked(model(encoder_outputs=encoded, decoder_input_ids=torch.tensor([list(cB) + list(y)], device=dev), use_cache=False).logits)
        target = torch.where(isE[:, None], torch.log_softmax(la_, -1), torch.log_softmax(lb_, -1))
    keep = ~ban
    with torch.enable_grad():
        out = torch.func.functional_call(model, {n: m.to(torch.bfloat16) for n, m in zip(names, masters)}, args=(),
                                         kwargs={"encoder_outputs": encoded, "decoder_input_ids": torch.tensor([list(cB) + list(y)], device=dev),
                                                 "use_cache": False}, strict=False, tie_weights=True)
        sl, _ = masked(out.logits)
        logp = torch.log_softmax(sl, -1)
        kl = F.kl_div(torch.where(keep, logp, torch.zeros_like(logp)), torch.where(keep, target, torch.zeros_like(target)),
                      reduction="none", log_target=True)
        kl = torch.where(keep, kl, torch.zeros_like(kl)).sum(-1)
        noop = list(cA) == list(cB) or int((valid & isE).sum()) == 0
        loss = sum((m * 0).sum() for m in masters) if noop else kl[valid].mean()
        loss.backward()
    g = torch.cat([(m.grad if m.grad is not None else torch.zeros_like(m)).detach().flatten() for m in masters])
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("live audit contaminated model gradients")
    return {"auditor_loss": float(loss), "valid": int(valid.sum()), "_grad": g}


# ---- prerun -----------------------------------------------------------------------------------------------------

def cmd_prerun(args) -> dict:
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    c = cfg()
    a3c = json.loads((ROOT / "configs/inference_cf/p2_tta_a3.json").read_text())
    checks = {}
    for rel in ("docs/inference_cf/P2_TTA_A4_SPEC.md", "docs/inference_cf/P2_TTA_A4_CLAUDE_DESIGN.md", CONFIG):
        checks[f"frozen:{rel}"] = sha(ROOT / rel) == tau.blob_sha(FREEZE, rel)
    checks["anchors"] = all(sha(ROOT / p) == h for p, h in c["source_sha256"].items())
    panel = json.loads((ROOT / PANEL).read_text())
    checks["panel_is_A3_panel"] = (sha(ROOT / PANEL) == c["panel"]["byte_sha256"] == a3c["panel"]["byte_sha256"]
                                   == "6422741b91e64ff007458b7a04d2f559e8ed8ffb592a087a538adf63632b2206"
                                   and [(r["utterance_id"], r["dialogue_id"], r["group"]) for r in panel["rows"]] == a3au.own_panel(a3c))
    checks["actuator_inherited"] = c["trainables"] == a3c["trainables"] and c["optimization"] == a3c["optimization"] and \
        c["optimization"]["lr"] == 1e-3 and c["optimization"]["steps"] == 2 and len(c["trainables"]["parameters"]) == 194
    checks["safety_copied"] = c["safety"] == a3c["safety"]
    a3ev = json.loads((ROOT / "results/inference_cf/p2tta_a3/evaluation.json").read_text())
    checks["rescue_reference_is_sealed_A3"] = all(c["partial_rescue"]["A3_reference"][k] == a3ev["points"]["A3"][k]
                                                  for k in ("zh_cer_increase", "zh_retention", "outside_harm_rate"))
    checks["thresholds"] = (c["mechanism"]["en_rows_decreased_min"] == 9 == math.ceil(0.75 * 11) and c["mechanism"]["en_median_R_E_min"] == 0.25
                            and c["mechanism"]["rho_anchor_max"] == 0.25 and c["benefit_vs_A2"] == {"poi_slack": 2, "mer_max_worse": 0.005,
                                                                                                  "no_new_severe_truncation": True})
    tok = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True).tokenizer
    gen = GenerationConfig.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True)
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    part = tokenizer_partition(tok)
    checks["partition"] = (part["hash"] == c["partition"]["hash"] and not (set(part["embedded_ids"]) & set(part["matrix_ids"]))
                           and len(part["embedded_ids"]) + len(part["matrix_ids"]) + len(part["ambiguous_ids"]) == len(tok))
    plan = json.loads((ROOT / PLAN_REL).read_text())
    checks["plan_committed"] = committed(PLAN_REL) and plan["plan_hash"] == tau.canon_digest({k: v for k, v in plan.items() if k != "plan_hash"}) \
        and all(plan["checks"].values()) and plan["references_used"] is False and plan["ids"] == c["panel"]["ids"]
    a3plan = json.loads((ROOT / "results/inference_cf/p2tta_a3/plan_sealed.json").read_text())
    a3seal = json.loads((ROOT / "results/inference_cf/p2tta_a3/output_seal.json").read_text())
    ok, counts = True, {"E": 0, "M": 0, "O": 0}
    elig = 0
    for i, (r, r3) in enumerate(zip(plan["rows"], a3plan["rows"])):
        rel = f"results/inference_cf/p2tta_a3/run1/rows/{i:02d}.json"
        a3row = json.loads((ROOT / rel).read_text())
        cls = own_classes(r["y_A"], part)
        mk = tau.own_valid(r["y_A"], sup, beg, tok)
        ok &= (all(r[k] == r3[k] for k in ("utterance_id", "group", "y_B", "y_A", "y_A_valid_mask", "A2", "d_BA", "d_A2A"))
               and r["classes"] == cls and r["y_A_valid_mask"] == mk and a3seal["files"][rel] == sha(ROOT / rel)
               and r["A3"]["tokens"] == a3row["A3"]["tokens"] and r["a3_lang_id"] == a3row["auto_condition"]["lang_id"])
        if r["group"] == "D":
            for cl, m_ in zip(cls, mk):
                if m_:
                    counts[cl] += 1
            elig += any(cl == "E" and m_ for cl, m_ in zip(cls, mk))
    checks["teachers_reuse_classes"] = bool(ok)
    checks["pre_outcome_counts"] = {"EN": counts["E"], "MATRIX": counts["M"], "OTHER": counts["O"]} == c["token_counts_pre_outcome"]["D"] and \
        elig == c["token_counts_pre_outcome"]["eligible_D_rows"] == 11
    src = (ROOT / "experiments/inference_cf_p2tta_a4.py").read_text()
    code = "\n".join(ast.get_source_segment(src, n) or "" for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in ("cmd_run", "cmd_prepare"))
    checks["runner_reference_steering_free"] = not any(t in code for t in ("load_references", "inference_cf_p2_evaluate", "corpus_metrics",
                                                                          "ReadoutDirection", "native_lid", "_edit_hook", "seq_decode", "num_beams=",
                                                                          "do_sample=True", "clip_grad", "lr_scheduler", "GradScaler", "autocast"))
    checks["final_decode_forced"] = "dec = forced_decode(bundle, enc_inf, CB, max_new_tokens=MAX_NEW)" in src
    mod = (ROOT / "src/csasr/inference_cf/script_safe_tta.py").read_text()
    checks["optimizer_inherited_in_code"] = "torch.optim.AdamW(list(masters.values()), **OPTIM)" in mod and "from csasr.inference_cf.episodic_tta import" in mod
    checks["no_outcome"] = not (BASE / "run1" / "rows").exists()
    asrc = (ROOT / "experiments/inference_cf_p2tta_a4_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(_a4_analyze|script_safe_tta|soft_auto_tta|episodic_tta)", asrc, re.M) is None
    checks["tests_present"] = (ROOT / "tests/test_inference_cf_p2tta_a4.py").exists()
    v = "PASS_TO_P2_TTA_A4" if all(checks.values()) else "BLOCK_BEFORE_P2_TTA_A4"
    return {"schema": "p2_tta_a4_prerun_audit_v1", "verdict": v, "checks": checks, "git_commit": _git("rev-parse", "HEAD")}


# ---- post -------------------------------------------------------------------------------------------------------

def cmd_post(args) -> dict:
    import torch
    from transformers import WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from experiments.inference_cf_p2_evaluate import load_references
    c = cfg()
    S, G = c["safety"], c["safety"]["float_guard"]
    run = ROOT / args.run
    ana = json.loads((ROOT / args.analysis).read_text())
    m = json.loads((run / "manifest.json").read_text())
    plan = json.loads((ROOT / PLAN_REL).read_text())
    seal = json.loads((BASE / "output_seal.json").read_text())
    rt = json.loads((run / "runtime.json").read_text())
    checks = {"manifest_self": m["manifest_hash"] == tau.canon_digest({k: v for k, v in m.items() if k != "manifest_hash"}),
              "sources_at_commit": all("sha256:" + (tau.blob_sha(m["git_commit"], p) or "") == h for p, h in m["sources"].items()),
              "seal_committed_unchanged": committed("results/inference_cf/p2tta_a4/output_seal.json") and all(sha(ROOT / p) == h for p, h in seal["files"].items())
              and seal["manifest_hash"] == m["manifest_hash"],
              "analysis_bound": ana["manifest_hash"] == m["manifest_hash"] and ana["output_seal_hash"] == seal["seal_hash"],
              "panel": m["ids"] == c["panel"]["ids"] == plan["ids"] and sha(ROOT / PANEL) == c["panel"]["byte_sha256"],
              "runtime": rt["status"] == "completed" and not rt.get("invalid") and rt["reset_final_ok"] and rt["nonln_unchanged"]
              and rt["nonln_hash_start"] == rt["nonln_hash_end"] and rt["model_grads_none"]}
    tok = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True).tokenizer
    part = tokenizer_partition(tok)
    sup, beg = plan["suppression"]["suppress"], plan["suppression"]["begin"]
    with np.load(run / "theta0_ln_fp32.npz") as z:
        th0 = [z[f"p{i:03d}"] for i in range(194)]
    checks["theta0_hash"] = tau.bf16_hash(th0) == rt["theta0_ln_hash"]
    rows = [json.loads((run / f"rows/{i:02d}.json").read_text()) for i in range(24)]
    P = {r["utterance_id"]: r for r in plan["rows"]}
    bad = {"rows": [], "diag": [], "update": [], "reset": [], "output": [], "distance": []}
    mine = []
    tol = c["mechanism"]["theta0_anchor_abs_tol"]
    for i, (r, u) in enumerate(zip(rows, m["ids"])):
        p, o = P[u], r["A4"]
        if not (r["status"] == "ok" and r["identity"] == u and r["theta0_forced"]["tokens_equal_S0"] and r["auto_condition"]["lang_id"] == p["a3_lang_id"]):
            bad["rows"].append(u)
        cls = own_classes(p["y_A"], part)
        mk = tau.own_valid(p["y_A"], sup, beg, tok)
        E = [mm and cl == "E" for cl, mm in zip(cls, mk)]
        NE = [mm and cl != "E" for cl, mm in zip(cls, mk)]
        Mx = [mm and cl == "M" for cl, mm in zip(cls, mk)]
        noop = r["auto_condition"]["cA"] == [50258, 50260, 50360, 50364] or not any(E) or not any(mk)
        mean = lambda xs, sel: float(np.mean(np.array([x for x, s_ in zip(xs, sel) if s_], dtype=np.float64))) if any(sel) else None
        dE = [mean(o["positions"][k]["kl_A"], E) for k in range(3)]
        dA = [mean(o["positions"][k]["kl_B"], NE) for k in range(3)]
        dM = [mean(o["positions"][k]["kl_B"], Mx) for k in range(3)]
        safe_kl = [[a if e else b for a, b, e in zip(o["positions"][k]["kl_A"], o["positions"][k]["kl_B"], E)] for k in range(3)]
        Ls = [mean(safe_kl[k], mk) for k in range(3)]
        close = lambda a, b: (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= 1e-5)
        if not all(close(a, b) for a, b in zip(dE, o["d_E"])) or not all(close(a, b) for a, b in zip(dA, o["d_anchor"])) or \
                not all(close(a, b) for a, b in zip(dM, o["d_M"])) or o["noop"] != noop or \
                (not noop and not all(close(a, b) for a, b in zip(Ls, o["losses"]))) or (noop and o["losses"] != [0.0, 0.0, 0.0]) or \
                (dA[0] is not None and abs(dA[0]) > tol) or o["steps"] != 2 or not all(o["finite"]) or \
                (o["n_E"], o["n_M"], o["n_nonE"]) != (sum(E), sum(Mx), sum(NE)):
            bad["diag"].append(u)
        with np.load(run / f"rows/{i:02d}_final_masters_fp32.npz") as z:
            mast = [z[f"A4_p{j:03d}"] for j in range(194)]
        ml2 = math.sqrt(sum(float(((ma.astype(np.float32) - t0).astype(np.float64) ** 2).sum()) for ma, t0 in zip(mast, th0)))
        eff = [torch.from_numpy(ma).to(torch.bfloat16).float().numpy() - t0 for ma, t0 in zip(mast, th0)]
        if abs(ml2 - o["master_delta_l2"]) > 1e-9 * max(1, ml2) or int(sum(int((e != 0).sum()) for e in eff)) != o["effective_changed_scalars"] \
                or (noop and ml2 != 0):
            bad["update"].append(u)
        if not (o["reset_ok"] and o["end_hash"] == rt["theta0_ln_hash"] and o["other_versions_unchanged"]):
            bad["reset"].append(u)
        if o["length"] != len(o["tokens"]) or (o["terminated"] == "cap") != (len(o["tokens"]) == 200) or \
                o["text"] != tok.decode(o["tokens"], skip_special_tokens=True) or (noop and o["tokens"] != p["y_B"]):
            bad["output"].append(u)
        dd = (tau.own_lev(o["tokens"], p["y_A"]), tau.own_lev(o["tokens"], p["y_B"]), tau.own_lev(o["tokens"], p["A2"]["tokens"]),
              tau.own_lev(o["tokens"], p["A3"]["tokens"]))
        if dd != (o["d_A4_AUTO"], o["d_A4_B0"], o["d_A4_A2"], o["d_A4_A3"]):
            bad["distance"].append(u)
        mine.append({"g": p["group"], "dE": dE, "dA": dA, "nE": sum(E), "nNE": sum(NE)})
    for k, v in bad.items():
        checks[f"{k}_recomputed"] = not v
    la = rows[0]["live_audit"]
    checks["live_audit"] = abs(la["primary_loss"] - la["auditor_loss"]) <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, 0.02 * la["auditor_grad_l2"])
    cnt = rt["counters"]
    checks["compute"] = cnt["A4"]["optimizer_steps"] == 48 and cnt["A4"]["backwards"] == 48 and cnt["audit"]["backwards"] == 1 and cnt["final_decodes"] == 24
    el = [x for x in mine if x["g"] == "D" and x["nE"] > 0]
    dec = sum(x["dE"][2] < x["dE"][0] for x in el)
    RE = [(x["dE"][0] - x["dE"][2]) / max(x["dE"][0], 1e-12) for x in el]
    en_pass = len(el) == 11 and dec >= 9 and float(np.median(RE)) >= 0.25 - 1e-12
    Gtw = sum(x["nE"] * (x["dE"][0] - x["dE"][2]) for x in el) / sum(x["nE"] for x in el)
    DA2 = sum(x["nNE"] * x["dA"][2] for x in el if x["dA"][2] is not None) / sum(x["nNE"] for x in el if x["dA"][2] is not None)
    rho = DA2 / max(Gtw, 1e-12)
    anc_pass = math.isfinite(rho) and rho <= c["mechanism"]["rho_anchor_max"] + 1e-12
    ag = ana["mechanics"]
    checks["mechanics_agree"] = (ag["en_transfer"]["pass"] == en_pass and ag["en_transfer"]["rows_decreased"] == dec
                                 and abs(ag["en_transfer"]["median_R_E"] - float(np.median(RE))) <= 1e-6
                                 and ag["anchor"]["pass"] == anc_pass and abs(ag["anchor"]["rho_anchor"] - rho) <= 1e-6 * max(1, abs(rho)))
    refs = load_references()
    ids = m["ids"]
    R = [refs[u]["reference"] for u in ids]
    H = {"B0_FORCED": [P[u]["y_B_text"] for u in ids], "B0_AUTO": [P[u]["y_A_text"] for u in ids], "A2": [P[u]["A2"]["text"] for u in ids],
         "A3": [P[u]["A3"]["text"] for u in ids], "A4": [r["A4"]["text"] for r in rows]}
    C = {k: np.array([tau.counts(a, b) for a, b in zip(R, H[k])], dtype=float) for k in H}
    rate = lambda c_, n, d: c_[:, n].sum() / c_[:, d].sum()
    met = {k: {"pier": rate(C[k], 0, 1), "mer": rate(C[k], 2, 3), "zh_cer": rate(C[k], 4, 5), "en_wer": rate(C[k], 6, 7),
               "poi_errors": int(C[k][:, 0].sum())} for k in C}
    checks["metrics_agree"] = max(abs(met[k][n] - ana["metrics"][k][n]) for k in met for n in ("pier", "mer", "zh_cer", "en_wer")) <= 1e-12 and \
        all(met[k]["poi_errors"] == ana["metrics"][k]["num_poi_errors"] for k in met)
    tr = tau.transitions(R, H["B0_FORCED"], H["A4"])
    tb, ta = [P[u]["y_B_terminated"] for u in ids], [r["A4"]["terminated"] for r in rows]
    added = sum(a == "cap" and b != "cap" for a, b in zip(ta, tb))
    sev = sum(len(P[u]["y_B"]) >= 10 and r["A4"]["terminated"] == "eos" and len(r["A4"]["tokens"]) <= math.floor(0.5 * len(P[u]["y_B"])) for r, u in zip(rows, ids))
    zinc = met["A4"]["zh_cer"] - met["B0_FORCED"]["zh_cer"]
    safe = {"MER": met["A4"]["mer"] - met["B0_FORCED"]["mer"] <= S["max_MER_increase"] + G, "ZH_CER": zinc <= S["max_ZH_CER_increase"] + G,
            "ZH_ret": tr["zr"] is None or tr["zr"] >= S["min_matrix_ZH_retention"] - G, "EN_ret": tr["er"] is None or tr["er"] >= S["min_embedded_EN_retention"] - G,
            "outside": tr["ohr"] is None or tr["ohr"] <= S["max_outside_POI_harm_rate"] + G,
            "corruption": tr["corr_rate"] is None or tr["corr_rate"] <= S["max_POI_corruption_rate"] + G,
            "caps": added <= S["max_additional_caps"], "severe": sev <= S["max_new_severe_truncations"]}
    pa = ana["points"]["A4"]
    checks["safety_points_agree"] = (tr["corrections"], tr["corruptions"]) == (pa["corrections"], pa["corruptions"]) and tr["zr"] == pa["zh_retention"] \
        and tr["er"] == pa["en_retention"] and tr["ohr"] == pa["outside_harm_rate"] and added == pa["added_caps"] and sev == pa["new_severe_truncations"]
    B = c["benefit_vs_A2"]
    ben = met["A4"]["poi_errors"] <= met["A2"]["poi_errors"] + B["poi_slack"] and met["A4"]["mer"] - met["A2"]["mer"] <= B["mer_max_worse"] + 1e-12 and sev == 0
    a3r = c["partial_rescue"]["A3_reference"]
    frac = {"zh_cer": (a3r["zh_cer_increase"] - zinc) / (a3r["zh_cer_increase"] - S["max_ZH_CER_increase"]),
            "retention": (tr["zr"] - a3r["zh_retention"]) / (S["min_matrix_ZH_retention"] - a3r["zh_retention"]),
            "outside": (a3r["outside_harm_rate"] - tr["ohr"]) / (a3r["outside_harm_rate"] - S["max_outside_POI_harm_rate"])}
    checks["partial_rescue_agrees"] = all(abs(frac[k] - ana["partial_matrix_rescue"]["fraction_closed"][k]) <= 1e-9 for k in frac)
    valid = all(checks.values())
    lab = ("P2_TTA_A4_INVALID" if not valid else "P2_TTA_A4_ENGLISH_TRANSFER_WEAK" if not en_pass else
           "P2_TTA_A4_SHARED_PARAMETER_INTERFERENCE" if not anc_pass else "P2_TTA_A4_SEQUENCE_SAFETY_NOT_RESCUED" if not all(safe.values())
           else "P2_TTA_A4_SAFE_BUT_NO_GAIN" if not ben else "P2_TTA_A4_PROMISING")
    checks["label_agrees"] = lab == ana["label"]
    v = "P2_TTA_A4_AUDIT: PASS" if all(checks.values()) else "P2_TTA_A4_AUDIT: BLOCK"
    return {"schema": "p2_tta_a4_post_audit_v1", "verdict": v, "label": lab, "checks": checks,
            "independent": {"metrics": met, "en": {"rows_decreased": dec, "median_R_E": float(np.median(RE)), "pass": en_pass},
                            "anchor": {"G_E_tw": Gtw, "D_ANCHOR2_tw": DA2, "rho": rho, "pass": anc_pass}, "safety": safe,
                            "rescue_fraction": frac, "benefit": ben},
            "failures": {k: v_[:10] for k, v_ in bad.items()}, "git_commit": _git("rev-parse", "HEAD")}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prerun").add_argument("--out", required=True)
    p = sub.add_parser("post")
    p.add_argument("--run", required=True)
    p.add_argument("--analysis", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"prerun": cmd_prerun, "post": cmd_post}[args.cmd](args)
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("audit output exists; never overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, sort_keys=True, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    print(json.dumps({k: res[k] for k in res if k in ("verdict", "label")}, indent=1))
    bad = {k: v for k, v in res["checks"].items() if not v}
    if bad:
        print(json.dumps(bad, indent=1))


if __name__ == "__main__":
    main()
