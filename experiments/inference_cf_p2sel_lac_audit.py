#!/usr/bin/env python
"""Independent P2-SEL-LAC auditor. Does NOT import inference_cf_p2sel_lac_analyze, any other analysis module, or
lexical_compatibility: candidates (own sorted argmax), W*, waveform masks and invariants (recomputed from the
source audio on CPU), feature-adapter identity, S_lex/F_lex, groups, bootstrap and label are re-implemented.

prerun -> PASS_TO_P2_SEL_LAC_LAC0 | BLOCK_BEFORE_P2_SEL_LAC_LAC0
lac0   -> P2_SEL_LAC_AUDIT: PASS (LAC0)
final  -> P2_SEL_LAC_AUDIT: PASS
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

from experiments.inference_cf_p2dir_audit import bf16, canon, fhash, git_blob_hash

FREEZE = "be460ce871ea3fda859e8a51a022de3148d29e18"
CONFIG = "configs/inference_cf/p2_sel_lac.json"
BASE = ROOT / "results/inference_cf/p2sel_lac"
SEAL = BASE / "candidates_sealed.json"
P2DIR = ROOT / "results/inference_cf/p2dir/exp1_run1"
T0 = ROOT / "results/inference_cf/p2sel_t/t0_run1"
MODEL = Path("/mnt/data/tungnx/whisper-large-v3")
FORBIDDEN = ("target_ids", "competitor", "stratum", "group", "positions.json", "timing", "oracle", "reference", "autograd", "readout")


def _git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def cfg():
    return json.loads((ROOT / CONFIG).read_text())


def surface_clean() -> dict:
    res = {}
    for rel, names in (("experiments/inference_cf_p2sel_lac.py", ("lac0_utterance", "cmd_seal")),
                       ("src/csasr/inference_cf/lexical_compatibility.py", ("select_candidates", "hard_mask", "lexical"))):
        src = (ROOT / rel).read_text()
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.FunctionDef) and n.name in names:
                res[n.name] = not any(f in ast.get_source_segment(src, n) for f in FORBIDDEN)
    return {"ok": len(res) == 5 and all(res.values()), "parts": res}


def cand(z, ids):
    best, bid = -np.inf, None
    for i in sorted(int(x) for x in ids):
        if z[i] > best:
            best, bid = z[i], i
    return bid


def partition():
    from transformers import WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    return tokenizer_partition(WhisperProcessor.from_pretrained(MODEL, local_files_only=True).tokenizer)


def cmd_prerun(args) -> dict:
    import torch
    from types import SimpleNamespace
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.models.whisper import batch_model_inputs, load_audio
    c = cfg()
    checks, notes = {}, {}
    for rel in ("docs/inference_cf/P2_SEL_LAC_SPEC.md", "docs/inference_cf/P2_SEL_LAC_CODEX_DESIGN.md", CONFIG):
        checks[f"frozen:{rel}"] = fhash(ROOT / rel) == git_blob_hash(FREEZE, rel)
    for rel, h in c["source_sha256"].items():
        checks[f"anchor:{rel}"] = fhash(ROOT / rel) == "sha256:" + h.replace("sha256:", "")
    pa = {"e": json.loads((ROOT / "results/inference_cf/p2sel_e/final_audit.json").read_text()),
          "t": json.loads((ROOT / "results/inference_cf/p2sel_t/final_audit.json").read_text()),
          "xa": json.loads((ROOT / "results/inference_cf/p2sel_xa/final_audit.json").read_text()),
          "xa0": json.loads((ROOT / "results/inference_cf/p2sel_xa/xa0_run1_audit.json").read_text())}
    checks["parents_terminal_pass"] = all(v["verdict"].endswith("PASS") for v in pa.values()) and pa["xa0"]["label"] == "P2_SEL_XA_INVALID"
    xa_rows = json.loads((ROOT / "results/inference_cf/p2sel_xa/xa0_run1_analysis.json").read_text())["rows"]
    checks["xa0_live_baseline_identity_180"] = len(xa_rows) == 180 and all(
        r["checks"]["logits_sealed_none"] and r["checks"]["r_sealed_hb"] for r in xa_rows)
    sealed = json.loads((P2DIR / "directions_sealed.json").read_text())
    checks["p2dir_seal"] = sealed["status"] == "SEALED" and all(fhash(P2DIR / r) == h for r, h in sealed["files"].items())
    part = partition()
    checks["partition"] = part["hash"] == c["candidates"]["partition_hash"] and len(part["embedded_ids"]) == 37858 and len(part["matrix_ids"]) == 1667
    gen = GenerationConfig.from_pretrained(MODEL, local_files_only=True)
    sup = set(gen.suppress_tokens or []) | set(gen.begin_suppress_tokens or [])
    checks["suppression_disjoint"] = not sup & (set(part["embedded_ids"]) | set(part["matrix_ids"]))
    con = json.loads((ROOT / "results/inference_cf/p2dir/construction_population.json").read_text())
    pos = json.loads((ROOT / "results/inference_cf/p2rj/positions.json").read_text())
    checks["population_180"] = [(p["utterance_id"], p["t"]) for p in pos["positions"]] == [(p["utterance_id"], p["t"]) for p in con["positions"]] and len(pos["positions"]) == 180
    # candidate seal: tracked, committed, independently reproduced
    seal = json.loads(SEAL.read_text())
    rel = str(SEAL.relative_to(ROOT))
    checks["seal_committed"] = _git("ls-files", rel) == rel and fhash(SEAL) == git_blob_hash("HEAD", rel)
    checks["seal_self_hash"] = canon({k: v for k, v in seal.items() if k != "seal_hash"}) == seal["seal_hash"]
    cidx = {u: i for i, u in enumerate(con["utterances"])}
    bad = []
    for r in seal["rows"]:
        i = cidx[r["utterance_id"]]
        with np.load(P2DIR / f"rows/{i:03d}_logits.npz") as z:
            v = bf16(z[f"t{r['t']}_none"])
        tp = json.loads((T0 / f"rows/{i:03d}.json").read_text())["positions"][str(r["t"])]
        if (cand(v, part["embedded_ids"]), cand(v, part["matrix_ids"])) != (r["c_E"], r["c_M"]) or r["W_star"] != tp["bounds"][tp["j"]] \
                or r["j"] != tp["j"] or float(v[r["c_E"]]) != r["z_E"]:
            bad.append((r["utterance_id"], r["t"]))
    checks["seal_reproduced_180"] = len(seal["rows"]) == 180 and not bad
    # CPU feature-adapter identity on all 80 original waveforms
    proc = WhisperProcessor.from_pretrained(MODEL, local_files_only=True)
    bun = SimpleNamespace(processor=proc, sample_rate=16000, device="cpu", dtype=torch.bfloat16)
    audio = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p0_r2/inference_panel.json").read_text())["rows"]}
    fbad = []
    for uid in con["utterances"]:
        ref = batch_model_inputs(bun, [audio[uid]["audio_path"]])
        f = proc.feature_extractor([load_audio(audio[uid]["audio_path"], 16000)], sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        if not (torch.equal(ref["input_features"], f.input_features.to("cpu", torch.bfloat16)) and torch.equal(ref["attention_mask"], f.attention_mask)):
            fbad.append(uid)
    checks["feature_adapter_identity_80"] = not fbad
    checks["mini_panel"] = fhash(ROOT / "docs/inference_cf/P2_SEL_MINI_PANEL.json") == "sha256:266ea7ea6328c0e6b68f554cb48085fc41359ade86c214defbfb9bff004815f5"
    checks["no_LAC_outcome"] = not any(x.name.startswith(("lac0", "lac1", "lac2")) for x in BASE.iterdir())
    checks["surface_reference_free"] = surface_clean()["ok"]
    asrc = (ROOT / "experiments/inference_cf_p2sel_lac_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(_analyze|lexical_compatibility)", asrc, re.M) is None
    checks["tests_present"] = (ROOT / "tests/test_inference_cf_p2sel_lac.py").exists()
    th = c["LAC0"]["thresholds"]
    checks["constants"] = (abs(th["EN_TP_S_min_nat"] - math.log(17 / 3)) < 1e-15 and abs(th["ZH_FP_S_max_nat"] - math.log(11 / 9)) < 1e-15
                           and c["compute"]["LAC0_masked_counterfactual_evaluations_max"] == 180)
    notes["seal_failures"], notes["feature_failures"] = bad[:5], fbad[:5]
    v = "PASS_TO_P2_SEL_LAC_LAC0" if all(checks.values()) else "BLOCK_BEFORE_P2_SEL_LAC_LAC0"
    return {"schema": "p2_sel_lac_prerun_audit_v1", "verdict": v, "checks": checks, "notes": notes, "git_commit": _git("rev-parse", "HEAD")}


def cmd_lac0(args) -> dict:
    from csasr.models.whisper import load_audio
    c = cfg()
    ana = json.loads((ROOT / args.analysis).read_text())
    run = BASE / "lac0_run1"
    man = json.loads((run / "manifest.json").read_text())
    pos = json.loads((ROOT / "results/inference_cf/p2rj/positions.json").read_text())
    con = json.loads((ROOT / "results/inference_cf/p2dir/construction_population.json").read_text())
    seal = {(r["utterance_id"], r["t"]): r for r in json.loads(SEAL.read_text())["rows"]}
    sel = {(r["utterance_id"], r["t"]): r["gate"]["E"] for r in json.loads((ROOT / "results/inference_cf/p2sel/s1_run1_analysis.json").read_text())["per_position"]}
    audio = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p0_r2/inference_panel.json").read_text())["rows"]}
    part = partition()
    cidx = {u: i for i, u in enumerate(con["utterances"])}
    checks = {"manifest_sources_at_commit": all(git_blob_hash(man["git_commit"], p) == h for p, h in man["sources"].items()),
              "analysis_manifest": ana["manifest_hash"] == man["manifest_hash"],
              "seal_unchanged": fhash(SEAL) == git_blob_hash(man["git_commit"], str(SEAL.relative_to(ROOT)))}
    an_rows = {(r["utterance_id"], r["t"]): r for r in ana["rows"]}
    rows, fails, wav = [], [], {}
    for p in pos["positions"]:
        uid, t = p["utterance_id"], int(p["t"])
        i = cidx[uid]
        s = seal[(uid, t)]
        x_row = json.loads((run / f"rows/{i:03d}.json").read_text())["positions"][str(t)]
        if uid not in wav:
            wav = {uid: load_audio(audio[uid]["audio_path"], 16000)}
        x = wav[uid]
        a, b = s["W_star"]
        heard = min(len(x), 480000)
        xm = x.copy()
        xm[a:b] = np.float32(0.0)
        sh = lambda arr: "sha256:" + hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
        e = float((x[a:b].astype(np.float64) ** 2).sum())
        ok = (0 <= a < b <= heard and sh(x) == x_row["x_sha256"] and sh(xm) == x_row["x_mask_sha256"]
              and abs(e - x_row["local_energy"]) <= 1e-9 * max(1.0, e) and (e == 0 or not np.array_equal(xm, x)))
        with np.load(P2DIR / f"rows/{i:03d}_logits.npz") as z:
            z0 = bf16(z[f"t{t}_none"]).astype(np.float64)
        if x_row.get("noop"):
            z1 = z0
        else:
            with np.load(run / f"rows/{i:03d}_logits.npz") as z:
                z1 = bf16(z[f"t{t}_masked"]).astype(np.float64)
        ok = ok and (cand(z0, part["embedded_ids"]), cand(z0, part["matrix_ids"])) == (s["c_E"], s["c_M"]) and np.isfinite(z1).all()
        cE, cM = s["c_E"], s["c_M"]
        S = (z0[cE] - z0[cM]) - (z1[cE] - z1[cM])
        ok = ok and abs(S - ((z0[cE] - z1[cE]) - (z0[cM] - z1[cM]))) <= 1e-10 and abs(S - an_rows[(uid, t)]["S_lex"]) <= 1e-10
        if not x_row.get("noop"):
            ok = ok and x_row["fed_equals_input_ids"] and x_row["query_matches"] and len(s["input_ids"]) == 4 + t
        if not ok:
            fails.append((uid, t))
        es = sel[(uid, t)]
        g = {"EN-confusion": "EN_TP" if es > 0 else "EN_FN", "ZH-correct": "ZH_FP" if es > 0 else "ZH_TN"}.get(p["stratum"], "EN_CORRECT")
        rows.append({"d": p["dialogue_id"], "g": g, "S": S})
    checks["row_reconstruction"] = not fails
    G = {k: [r for r in rows if r["g"] == k] for k in ("EN_TP", "EN_FN", "ZH_TN", "ZH_FP", "EN_CORRECT")}
    checks["groups"] = {k: len(v) for k, v in G.items()} == c["population"]["historical_groups"]
    sp = lambda rs: len({r["d"] for r in rs})
    hiS, loS = math.log(17 / 3), math.log(11 / 9)
    tps = [r for r in G["EN_TP"] if r["S"] >= hiS]
    fps = [r for r in G["ZH_FP"] if r["S"] <= loS]
    fns = [r for r in G["EN_FN"] if r["S"] >= hiS]
    keys = sorted({r["d"] for r in rows})
    idx = np.random.default_rng(240924).integers(0, len(keys), size=(10000, len(keys)))

    def per(rs):
        acc = {}
        for r in rs:
            acc.setdefault(r["d"], []).append(r["S"])
        return {k: sum(v) / len(v) for k, v in acc.items()}
    A, B = per(G["EN_TP"]), per(G["ZH_FP"])
    ds = []
    for draw in idx:
        ks = [keys[j] for j in draw]
        va, vb = [A[k] for k in ks if k in A], [B[k] for k in ks if k in B]
        if va and vb:
            ds.append(sum(va) / len(va) - sum(vb) / len(vb))
    pt = sum(A.values()) / len(A) - sum(B.values()) / len(B)
    l80 = float(np.quantile(ds, 0.2))
    valid = checks["row_reconstruction"] and checks["groups"] and len(ds) >= 9900 and all(ana["validity"].values())
    if not valid:
        lab = "P2_SEL_LAC_INVALID"
    elif len(tps) >= 34 and sp(tps) >= 3 and len(fps) >= 4 and sp(fps) >= 3 and pt >= 0.5 and l80 > 0:
        lab = "P2_SEL_LAC_LEXICAL_COMPATIBILITY_SUPPORTED"
    elif len(fns) >= 9 and sp(fns) >= 3:
        lab = "P2_SEL_LAC_RECALL_ONLY"
    else:
        lab = "P2_SEL_LAC_NOT_DISCRIMINATIVE"
    d = ana["decision"]
    checks["statistics_agree"] = abs(pt - d["S_TP_minus_FP"]["point"]) <= 1e-8 and abs(l80 - d["S_TP_minus_FP"]["lower80"]) <= 1e-8 \
        and len(ds) == d["S_TP_minus_FP"]["valid_draws"]
    checks["counts_agree"] = (len(tps), len(fps), len(fns)) == (d["counts"]["tp_strong"], d["counts"]["fp_weak"], d["counts"]["fn_strong"])
    checks["label_agrees"] = lab == ana["label"]
    return {"schema": "p2_sel_lac_lac0_audit_v1", "stage": "LAC0", "label": lab,
            "verdict": "P2_SEL_LAC_AUDIT: PASS" if all(checks.values()) else "P2_SEL_LAC_AUDIT: BLOCK", "checks": checks,
            "independent": {"tp_strong": len(tps), "fp_weak": len(fps), "fn_strong": len(fns), "S_TP_minus_FP": pt, "lower80": l80,
                            "valid_draws": len(ds)}, "row_failures": fails[:10]}


def cmd_final(args) -> dict:
    c = cfg()
    checks = {}
    for rel in ("docs/inference_cf/P2_SEL_LAC_SPEC.md", "docs/inference_cf/P2_SEL_LAC_CODEX_DESIGN.md", CONFIG):
        checks[f"unchanged:{rel}"] = fhash(ROOT / rel) == git_blob_hash(FREEZE, rel)
    for rel, h in c["source_sha256"].items():
        checks[f"anchor:{rel}"] = fhash(ROOT / rel) == "sha256:" + h.replace("sha256:", "")
    entries = sorted(x.name for x in BASE.iterdir())
    a0 = json.loads((BASE / "lac0_run1_audit.json").read_text())
    checks["lac0_audit_pass"] = a0["verdict"] == "P2_SEL_LAC_AUDIT: PASS"
    checks["lac1_only_if_supported"] = not any(e.startswith(("lac1", "lac2")) for e in entries) or a0["label"] == "P2_SEL_LAC_LEXICAL_COMPATIBILITY_SUPPORTED"
    rt = json.loads((BASE / "lac0_run1/runtime.json").read_text())["counters"]
    checks["lac0_forward_only"] = rt["autograd_calls"] == 0 and rt["readout_calls"] == 0 and rt["lid_calls"] == 0 and rt["steering_calls"] == 0 \
        and rt["counterfactual_evaluations"] <= 180
    m = json.loads((BASE / "lac0_run1/manifest.json").read_text())
    checks["firewall"] = m["role"] == "D-dev-select" and m["firewall"]["role"] == "D-dev-select"
    checks["single_attempt"] = len(list((BASE / "lac0_run1").glob("slurm-*.out"))) == 1
    checks["surface_reference_free"] = surface_clean()["ok"]
    checks["no_p3_full300"] = not any(("p3" in e.lower()) or ("300" in e) for e in entries)
    return {"schema": "p2_sel_lac_final_audit_v1", "verdict": "P2_SEL_LAC_AUDIT: PASS" if all(checks.values()) else "P2_SEL_LAC_AUDIT: BLOCK",
            "lac0_label": a0["label"], "entries": entries, "checks": checks, "git_commit": _git("rev-parse", "HEAD")}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("prerun", "final"):
        sub.add_parser(n).add_argument("--out", required=True)
    x = sub.add_parser("lac0")
    x.add_argument("--analysis", required=True)
    x.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"prerun": cmd_prerun, "lac0": cmd_lac0, "final": cmd_final}[args.cmd](args)
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("audit output exists; never overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, sort_keys=True, indent=1, default=str) + "\n")
    print(json.dumps({k: res[k] for k in res if k in ("verdict", "label", "lac0_label")}, indent=1))
    print(json.dumps({k: v for k, v in res["checks"].items() if not v}, indent=1))


if __name__ == "__main__":
    main()
