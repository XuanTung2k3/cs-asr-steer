#!/usr/bin/env python
"""Independent P2-SEQ auditor. Does NOT import inference_cf_p2seq_analyze (nor any other analysis/bootstrap/decision
module). It may use the locked canonical metric primitives (pier / mer / retention / unit_status) but aggregates
counts, draws, safety predicates and the label itself; edit energy is recomputed by exact bf16 hook emulation.

prerun -> PASS_TO_P2_SEQ | BLOCK_BEFORE_P2_SEQ
post   -> P2_SEQ_AUDIT: PASS | P2_SEQ_AUDIT: BLOCK
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

from experiments.inference_cf_p2dir_audit import canon, fhash, git_blob_hash

FREEZE = "660a6197ca2ea49f6df296e92a751800f82e240e"
CONFIG = "configs/inference_cf/p2_seq.json"
BASE = ROOT / "results/inference_cf/p2seq"


def _git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def runner_clean() -> dict:
    src = (ROOT / "experiments/inference_cf_p2seq.py").read_text()
    tree = ast.parse(src)
    res = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name in ("seq_decode", "run"):
            seg = ast.get_source_segment(src, n)
            res[n.name] = not any(f in seg for f in ("load_references", "reference", "evaluation", "optimizer", "adapt",
                                                     "entropy", "backward(", "target_ids", "pier", "corpus_mer"))
    return {"ok": len(res) == 2 and all(res.values()), "parts": res}


def cmd_prerun(args) -> dict:
    c = json.loads((ROOT / CONFIG).read_text())
    checks, notes = {}, {}
    for rel in ("docs/inference_cf/P2_SEQ_SPEC.md", "docs/inference_cf/P2_SEQ_CODEX_DESIGN.md", CONFIG):
        checks[f"frozen:{rel}"] = fhash(ROOT / rel) == git_blob_hash(FREEZE, rel)
    for rel, h in c["source_sha256"].items():
        checks[f"anchor:{rel}"] = sha(ROOT / rel) == h
    panel = json.loads((ROOT / c["panel"]["path"]).read_text())
    ids = [r["utterance_id"] for r in panel["rows"]]
    per = {}
    for r in panel["rows"]:
        per[r["dialogue_id"]] = per.get(r["dialogue_id"], 0) + 1
    checks["panel"] = sha(ROOT / c["panel"]["path"]) == c["panel"]["sha256"] and ids == c["panel"]["ids"] and len(set(ids)) == 100 \
        and len(per) == 20 and set(per.values()) == {5}
    reuse = json.loads((ROOT / "results/inference_cf/p2seq/reuse_audit.json").read_text())
    rel = "results/inference_cf/p2seq/reuse_audit.json"
    checks["reuse_seal_committed"] = _git("ls-files", rel) == rel and fhash(ROOT / rel) == git_blob_hash("HEAD", rel) \
        and reuse["reuse_hash"] == canon({k: v for k, v in reuse.items() if k != "reuse_hash"}) and reuse["outcomes_computed"] is False
    from csasr.utils.hashing import sha256_file
    hist_start = json.loads((ROOT / "results/inference_cf/p2_A_r1_L16/runtime.json").read_text())["start_unix"]
    sealed = {r["utterance_id"]: r for r in c["reuse"]["rows"]}
    hp = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p2_A_r1_L16/panel.json").read_text())["rows"]}
    rows_ok = True
    for u in ids:
        s = sealed[u]
        row = json.loads((ROOT / s["path"]).read_text())
        rows_ok &= (sha(ROOT / s["path"]) == s["sha256"] and row["identity"] == u and row["status"] == "ok"
                    and "B0_AUTO" in row["systems"] and sha256_file(hp[u]["audio_path"], max_bytes=65536) == hp[u]["audio_sha256"]
                    and Path(hp[u]["audio_path"]).stat().st_mtime < hist_start)
    checks["auto_rows_and_audio"] = bool(rows_ok)
    hman = json.loads((ROOT / "results/inference_cf/p2_A_r1_L16/manifest.json").read_text())
    checks["historical_sources_unchanged"] = all("sha256:" + sha(p) == h for p, h in hman["sources"].items())
    r2m = json.loads((ROOT / "results/inference_cf/p0_r2/manifest.json").read_text())
    import platform
    import torch
    import transformers
    env = (platform.python_version(), torch.__version__, transformers.__version__)
    checks["environment"] = env == ("3.11.9", "2.10.0+cu128", "4.57.6") and \
        (r2m["environment"]["python"], r2m["environment"]["torch"], r2m["environment"]["transformers"]) == env
    hrt = json.loads((ROOT / "results/inference_cf/p2_A_r1_L16/runtime.json").read_text())
    site = Path("/home/tungnx/miniconda3/envs/acl1/lib/python3.11/site-packages")
    checks["libraries_predate_historical"] = all(max(f.stat().st_mtime for f in d.iterdir()) < hrt["start_unix"]
                                                 for pat in ("torch-*.dist-info", "transformers-*.dist-info", "tokenizers-*.dist-info")
                                                 for d in site.glob(pat))
    checks["model_predates_historical"] = max(f.stat().st_mtime for f in Path("/mnt/data/tungnx/whisper-large-v3").iterdir()) < hrt["start_unix"]
    checks["reuse_decision_consistent"] = reuse["decision_AUTO"] == ("REUSE_AUTO_ALL_100" if all(
        checks[k] for k in ("auto_rows_and_audio", "historical_sources_unchanged", "environment", "libraries_predate_historical",
                            "model_predates_historical")) else "COMPUTE_AUTO_ALL_100")
    checks["no_outcome"] = not any(x.name.startswith("run") for x in BASE.iterdir())
    d_cfg = (ROOT / "configs/data/cs_dialogue.yaml").read_text()
    checks["audio_fingerprint_definition"] = "audio_hash_bytes: 65536" in d_cfg
    checks["runner_reference_and_tta_free"] = runner_clean()["ok"]
    src = (ROOT / "experiments/inference_cf_p2seq.py").read_text()
    checks["alpha_restricted_0_2"] = 'if alpha not in (0.0, ALPHA_STEER)' in src and "ALPHA_STEER = 2.0" in src and "LAYER = 16" in src \
        and "MAX_NEW = 200" in src
    asrc = (ROOT / "experiments/inference_cf_p2seq_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*_analyze", asrc, re.M) is None
    checks["tests_present"] = (ROOT / "tests/test_inference_cf_p2seq.py").exists()
    notes["auto_decision"] = reuse["decision_AUTO"]
    v = "PASS_TO_P2_SEQ" if all(checks.values()) else "BLOCK_BEFORE_P2_SEQ"
    return {"schema": "p2_seq_prerun_audit_v1", "verdict": v, "checks": checks, "notes": notes, "git_commit": _git("rev-parse", "HEAD")}


def counts(ref, hyp):
    from csasr.evaluation.mer import corpus_mer
    from csasr.evaluation.pier import pier
    p, m = pier([ref], [hyp]), corpus_mer([ref], [hyp])
    return [p["num_poi_errors"], p["num_poi"], m["substitutions"] + m["deletions"] + m["insertions"], m["num_ref_tokens"],
            round(m["zh_cer"] * m["num_zh_ref"]) if m["num_zh_ref"] else 0, m["num_zh_ref"],
            round(m["en_wer"] * m["num_en_ref"]) if m["num_en_ref"] else 0, m["num_en_ref"]]


def cmd_post(args) -> dict:
    from csasr.evaluation.normalization import EN, ZH, normalize_text, segment_units, tag_unit
    from csasr.evaluation.pier import evaluate_pois, unit_status
    from experiments.inference_cf_p2_evaluate import load_references
    from experiments.inference_cf_p2sel_audit import hook_edit_emulation
    run = BASE / "run1"
    ana = json.loads((ROOT / args.analysis).read_text())
    m = json.loads((run / "manifest.json").read_text())
    reuse = json.loads((BASE / "reuse_audit.json").read_text())
    checks = {"manifest_self": m["manifest_hash"] == canon({k: v for k, v in m.items() if k != "manifest_hash"}),
              "sources_at_commit": all(git_blob_hash(m["git_commit"], p) == h for p, h in m["sources"].items()),
              "analysis_manifest": ana["manifest_hash"] == m["manifest_hash"],
              "panel": sha(ROOT / "docs/inference_cf/P2_SEL_MINI_PANEL.json") == m["panel_sha256"] and len(m["ids"]) == 100}
    refs = load_references()
    ids = m["ids"]
    rows = [json.loads((run / f"rows/{i:03d}.json").read_text()) for i in range(100)]
    checks["rows_ok"] = all(r["status"] == "ok" and r["identity"] == u and {"S0", "S2"} <= set(r["systems"]) for r, u in zip(rows, ids))
    auto = {}
    for x in reuse["rows"]:
        if reuse["decision_AUTO"] == "REUSE_AUTO_ALL_100":
            checks.setdefault("auto_rows_hash", True)
            checks["auto_rows_hash"] &= sha(ROOT / x["row"]) == x["row_sha256"]
            auto[x["utterance_id"]] = json.loads((ROOT / x["row"]).read_text())["systems"]["B0_AUTO"]
    if reuse["decision_AUTO"] != "REUSE_AUTO_ALL_100":
        auto = {u: r["systems"]["S1"] for r, u in zip(rows, ids)}
    # step-level integrity and energy
    bad, efail, unattr = [], [], []
    nedit = 0
    for i, (r, u) in enumerate(zip(rows, ids)):
        s0, s2 = r["systems"]["S0"], r["systems"]["S2"]
        with np.load(run / f"rows/{i:03d}.npz") as z:
            V = {k: z[k] for k in z.files}
        ok = all(st["S_bitwise_B"] and st["next"] == st["unsteered_next"] and not st["d2_called"] and not st["edited"] for st in s0["steps"])
        edits = [st for st in s2["steps"] if st["edited"]]
        fe = edits[0]["t"] if edits else None
        ok &= fe == s2["first_edit_t"] and all(st["S_bitwise_B"] for st in s2["steps"] if fe is None or st["t"] < fe)
        ok &= all(st["d2_logits_bitwise_B"] and st["d2_site_bitwise_B"] for st in s2["steps"] if st["d2_called"])
        ok &= all(st.get("noedit_site_identity", True) for st in s0["steps"] + s2["steps"] if not st["edited"])
        ok &= all((st["edited"] == (st["d2_called"] and st["dir"] == "ok" and st["dose"] > 0)) for st in s2["steps"]) and \
            all(st["query"] >= 4 for st in edits) and s0["lineage_ok"] and s2["lineage_ok"]
        for st in edits:
            nedit += 1
            pre, post, d = V[f"S2_t{st['t']}_pre"], V[f"S2_t{st['t']}_post"], V[f"S2_t{st['t']}_d2"]
            emu = hook_edit_emulation(pre, d, st["g"])
            if abs(emu / st["edit_norm"] - 1) > 1e-5 or abs(float(np.linalg.norm(post.astype(np.float64) - pre.astype(np.float64))) - st["edit_norm"]) > 5e-3 * max(1.0, st["edit_norm"]):
                efail.append((u, st["t"]))
            if abs(st["g"] - st["E"] * st["R_B"]) > 1e-12 or abs(float(np.linalg.norm(d.astype(np.float64))) - 1) > 2e-6:
                efail.append((u, st["t"], "gate_or_unit"))
        if s2["tokens"] != s0["tokens"] or s2["terminated"] != s0["terminated"]:
            k = next((t for t, (a, b) in enumerate(zip(s2["tokens"], s0["tokens"])) if a != b), min(len(s2["tokens"]), len(s0["tokens"])))
            if fe is None or fe > k:
                unattr.append(u)
        if not ok:
            bad.append(u)
    checks["step_integrity"] = not bad
    checks["energy_emulation_and_gate"] = not efail
    checks["divergences_attributed"] = not unattr
    rt = json.loads((run / "runtime.json").read_text())
    checks["runtime"] = rt["status"] == "completed" and rt["counters"]["S0"]["d2_calls"] == 0 and \
        rt["counters"]["S2"]["d2_calls"] == sum(st["d2_called"] for r in rows for st in r["systems"]["S2"]["steps"])
    # independent metrics
    R = [refs[u]["reference"] for u in ids]
    D = [refs[u]["dialogue_id"] for u in ids]
    H = {"S0": [r["systems"]["S0"]["text"] for r in rows], "S2": [r["systems"]["S2"]["text"] for r in rows], "S1": [auto[u]["text"] for u in ids]}
    C = {k: np.array([counts(a, b) for a, b in zip(R, H[k])], dtype=float) for k in H}
    rate = lambda c, n, d: c[:, n].sum() / c[:, d].sum()
    mine = {k: {"pier": rate(C[k], 0, 1), "mer": rate(C[k], 2, 3), "zh_cer": rate(C[k], 4, 5), "en_wer": rate(C[k], 6, 7),
                "poi_errors": int(C[k][:, 0].sum())} for k in C}
    diffs = [abs(mine[k][n] - ana["metrics"][k][n]) for k in mine for n in ("pier", "mer", "zh_cer", "en_wer")]
    checks["metrics_agree"] = max(diffs) <= 1e-12 and all(mine[k]["poi_errors"] == ana["metrics"][k]["num_poi_errors"] for k in mine)
    # POI transitions / retention / outside harm (own aggregation)
    corr = corrupt = bc = 0
    zk = zt = ek = et = oh = ob = 0
    for r, b, s in zip(R, H["S0"], H["S2"]):
        bb = {p.poi_index: p.correct for p in evaluate_pois(r, b)}
        ss = {p.poi_index: p.correct for p in evaluate_pois(r, s)}
        for k, v in bb.items():
            bc += v
            corr += (not v) and ss.get(k, False)
            corrupt += v and not ss.get(k, False)
        units = segment_units(normalize_text(r))
        tags = [tag_unit(x) for x in units]
        is_cs = EN in tags and ZH in tags
        ub, us = unit_status(r, b), unit_status(r, s)
        for i, tg in enumerate(tags):
            okb, oks = ub.get(i, (False,))[0], us.get(i, (False,))[0]
            if okb and is_cs and tg == ZH:
                zt += 1
                zk += oks
            if okb and is_cs and tg == EN:
                et += 1
                ek += oks
            if tg != EN and okb:
                ob += 1
                oh += not oks
    checks["transition_identity"] = corr - corrupt == mine["S0"]["poi_errors"] - mine["S2"]["poi_errors"] and \
        (corr, corrupt) == (ana["transitions"]["corrections"], ana["transitions"]["corruptions"])
    zr, er_ = (zk / zt if zt else None), (ek / et if et else None)
    ohr = oh / ob if ob else None
    checks["retention_outside_agree"] = (zr == ana["retention"]["matrix_zh"]["rate"] and er_ == ana["retention"]["embedded_en"]["rate"]
                                         and ohr == ana["outside"]["harm_rate"])
    # own bootstrap
    keys = sorted(set(D))
    grp = {k: [i for i, d in enumerate(D) if d == k] for k in keys}
    rng = np.random.default_rng(240924)
    draws = [sum((grp[keys[j]] for j in rng.integers(0, 20, 20)), []) for _ in range(2000)]
    bdiff = []
    for comp in ("S0", "S1"):
        for n, (a, b) in {"pier": (0, 1), "mer": (2, 3), "zh_cer": (4, 5), "en_wer": (6, 7)}.items():
            vals = [C["S2"][ix][:, a].sum() / C["S2"][ix][:, b].sum() - C[comp][ix][:, a].sum() / C[comp][ix][:, b].sum()
                    for ix in draws if C["S2"][ix][:, b].sum() and C[comp][ix][:, b].sum()]
            ref_ = ana["bootstrap"][f"S2_minus_{comp}"][n]
            bdiff += [abs(np.quantile(vals, .025) - ref_["ci95"][0]), abs(np.quantile(vals, .975) - ref_["ci95"][1])]
            if comp == "S0":
                checks.setdefault("valid_draws", True)
                checks["valid_draws"] &= len(vals) >= 1980
    checks["bootstrap_agrees"] = max(bdiff) <= 1e-9
    # own label
    caps = {k: sum((r["systems"][k]["terminated"] if k != "S1" else auto[u]["terminated"]) == "cap" for r, u in zip(rows, ids))
            for k in ("S0", "S2", "S1")}
    g = 1e-12
    safety = (mine["S2"]["mer"] - mine["S0"]["mer"] <= 0.005 + g and mine["S2"]["zh_cer"] - mine["S0"]["zh_cer"] <= 0.005 + g
              and (zr is None or zr >= 0.99 - g) and (er_ is None or er_ >= 0.95 - g)
              and (bc == 0 or corrupt / bc <= 0.05 + g) and (ohr is None or ohr <= 0.01 + g) and caps["S2"] - caps["S0"] <= 1)
    benefit = (mine["S0"]["poi_errors"] - mine["S2"]["poi_errors"] >= 5) and (mine["S0"]["pier"] - mine["S2"]["pier"] >= 0.005 - g)
    valid = all(checks.values())
    lab = "P2_SEQ_INVALID" if not valid else ("P2_SEQ_SEQUENCE_DAMAGE" if not safety else
                                              ("P2_SEQ_PROMISING" if benefit else "P2_SEQ_NO_USEFUL_GAIN"))
    checks["label_agrees"] = lab == ana["label"]
    return {"schema": "p2_seq_post_audit_v1", "verdict": "P2_SEQ_AUDIT: PASS" if all(checks.values()) else "P2_SEQ_AUDIT: BLOCK",
            "label": lab, "checks": checks, "independent": {"metrics": mine, "corrections": corr, "corruptions": corrupt,
                                                            "zh_retention": zr, "en_retention": er_, "outside_harm_rate": ohr,
                                                            "caps": caps, "edits": nedit},
            "failures": {"steps": bad[:10], "energy": efail[:10], "unattributed": unattr[:10]}, "git_commit": _git("rev-parse", "HEAD")}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prerun").add_argument("--out", required=True)
    p = sub.add_parser("post")
    p.add_argument("--analysis", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"prerun": cmd_prerun, "post": cmd_post}[args.cmd](args)
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("audit output exists; never overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, sort_keys=True, indent=1, default=str) + "\n")
    print(json.dumps({k: res[k] for k in res if k in ("verdict", "label")}, indent=1))
    print(json.dumps({k: v for k, v in res["checks"].items() if not v}, indent=1))


if __name__ == "__main__":
    main()
