#!/usr/bin/env python
"""Independent P2-TTA0 auditor. Does NOT import inference_cf_p2tta0_analyze, csasr.inference_cf.episodic_tta, or
any other analysis/bootstrap/decision module. It may use the locked canonical metric primitives (pier / mer /
normalization / unit_status) but owns its counts, retention/outside-harm aggregation, safety/useful predicates,
confirmation-bias test, labels, selection and bootstrap.

prerun -> PASS_TO_P2_TTA0 | BLOCK_BEFORE_P2_TTA0
live   -> ``live_objective_check`` (imported by the runner, FIRST utterance only, theta0, no optimizer update): an
          independent full-vocabulary masked formula (Categorical entropy / cross_entropy) and backward on fresh
          fp32 masters, compared with the primary L0 / step-0 gradient.
post   -> P2_TTA0_AUDIT: PASS | P2_TTA0_AUDIT: BLOCK
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

FREEZE = "eb0da2a06cdffba4395a58215458d682bbba94e9"
CONFIG = "configs/inference_cf/p2_tta0.json"
PANEL = "docs/inference_cf/P2_TTA0_PANEL20.json"
BASE = ROOT / "results/inference_cf/p2tta0"
SEAL = BASE / "pseudo_sealed.json"
G = 1e-12


def _git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def blob_sha(commit: str, rel: str) -> str | None:
    r = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=ROOT, capture_output=True)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def canon_digest(obj) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                                                 allow_nan=False).encode()).hexdigest()


def fhash(p) -> str:
    return "sha256:" + sha(p)


def own_valid(y, suppress, begin, tokenizer) -> list[bool]:
    sp = set(int(i) for i in tokenizer.all_special_ids)
    eos = int(tokenizer.eos_token_id)
    s, b = set(suppress), set(begin)
    return [(t < eos and t not in sp) and t not in s and not (j == 0 and t in b) for j, t in enumerate(y)]


# ---- live first-utterance objective/gradient check ---------------------------------------------------------

def live_objective_check(model, names, encoded, prompt, y, kind, *, suppress, begin, tokenizer) -> dict:
    """Independent formula at theta0: full-vocabulary logits with suppressed IDs masked to -inf, entropy via
    torch.distributions.Categorical (A1) or cross_entropy (A2) over the independently derived valid set; one backward
    to fresh fp32 masters. No optimizer, no resident parameter change."""
    import torch
    import torch.nn.functional as F
    params = dict(model.named_parameters())
    masters = [torch.nn.Parameter(params[n].detach().float().clone()) for n in names]
    dev = masters[0].device
    ids = torch.tensor([list(prompt) + list(y)], device=dev)
    valid = torch.tensor(own_valid(y, suppress, begin, tokenizer), dtype=torch.bool, device=dev)
    with torch.enable_grad():
        out = torch.func.functional_call(model, {n: m.to(torch.bfloat16) for n, m in zip(names, masters)}, args=(),
                                         kwargs={"encoder_outputs": encoded, "decoder_input_ids": ids, "use_cache": False},
                                         strict=False, tie_weights=True)
        q = out.logits[0].float()[len(prompt) - 1:len(prompt) - 1 + len(y)]
        ban = torch.zeros_like(q, dtype=torch.bool)
        if suppress:
            ban[:, list(suppress)] = True
        if begin and len(y):
            ban[0, list(begin)] = True
        q = q.masked_fill(ban, float("-inf"))
        if int(valid.sum()) == 0:
            loss = sum((m * 0).sum() for m in masters)
        elif kind == "A1":
            loss = torch.distributions.Categorical(logits=q[valid]).entropy().mean()
        else:
            tgt = torch.tensor(list(y), device=dev)[valid]
            loss = F.cross_entropy(q[valid], tgt, reduction="mean")
        loss.backward()
    g = torch.cat([(m.grad if m.grad is not None else torch.zeros_like(m)).detach().flatten() for m in masters])
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("live audit contaminated model gradients")
    return {"auditor_loss": float(loss), "valid": int(valid.sum()), "_grad": g}


# ---- prerun ---------------------------------------------------------------------------------------------------

def runner_clean() -> dict:
    res = {}
    for rel in ("experiments/inference_cf_p2tta0.py", "src/csasr/inference_cf/episodic_tta.py"):
        src = (ROOT / rel).read_text()
        code = "\n".join(ast.get_source_segment(src, n) or "" for n in ast.parse(src).body
                         if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)))
        res[rel] = not any(f in code for f in ("load_references", "inference_cf_p2_evaluate", "corpus_mer", "corpus_metrics",
                                               "ReadoutDirection", "native_lid", "_edit_hook", "seq_decode", "cached_decode",
                                               "do_sample=True", "num_beams=", "clip_grad", "lr_scheduler", "GradScaler", "autocast"))
    return {"ok": all(res.values()), "parts": res}


def cmd_prerun(args) -> dict:
    import torch
    from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor
    c = json.loads((ROOT / CONFIG).read_text())
    checks, notes = {}, {}
    for rel in ("docs/inference_cf/P2_TTA0_SPEC.md", "docs/inference_cf/P2_TTA0_CODEX_DESIGN.md", CONFIG, PANEL):
        checks[f"frozen:{rel}"] = sha(ROOT / rel) == blob_sha(FREEZE, rel)
    for rel, h in c["source_sha256"].items():
        checks[f"anchor:{rel}"] = sha(ROOT / rel) == h
    panel = json.loads((ROOT / PANEL).read_text())
    parent = json.loads((ROOT / panel["parent"]).read_text())
    firsts, seen = [], set()
    for r in parent["rows"]:
        if r["dialogue_id"] not in seen:
            seen.add(r["dialogue_id"])
            firsts.append(r["utterance_id"])
    ids = [r["utterance_id"] for r in panel["rows"]]
    checks["panel"] = sha(ROOT / PANEL) == c["panel"]["byte_sha256"] == "3c752d7a2e247a77f05956839e3aa396ae04818fea7440f74b87778af5fa9eb4" \
        and ids == firsts == c["panel"]["ids"] and len({r["dialogue_id"] for r in panel["rows"]}) == 20 and len(ids) == 20
    model_dir = "/mnt/data/tungnx/whisper-large-v3"
    with torch.device("meta"):
        mdl = WhisperForConditionalGeneration(WhisperConfig.from_pretrained(model_dir, local_files_only=True))
    ln = []
    for mn, mod in mdl.model.decoder.named_modules():
        if isinstance(mod, torch.nn.LayerNorm):
            ln += [(f"model.decoder.{mn}.weight", list(mod.weight.shape)), (f"model.decoder.{mn}.bias", list(mod.bias.shape))]
    total = sum(p.numel() for p in mdl.parameters())
    checks["trainables_exact"] = [(p["name"], p["shape"]) for p in c["trainables"]["parameters"]] == ln and len(ln) == 194 \
        and sum(math.prod(s) for _, s in ln) == 248320 == c["trainables"]["scalar_count"] and total == c["trainables"]["total_unique_model_scalars"] \
        and abs(248320 / total - c["trainables"]["fraction"]) < 1e-15 and not any(n.startswith("model.encoder") for n, _ in ln)
    notes["trainables"] = {"tensors": len(ln), "scalars": 248320, "total": total}
    o = c["optimization"]
    esrc = (ROOT / "src/csasr/inference_cf/episodic_tta.py").read_text()
    checks["optimizer_frozen"] = (o["optimizer"] == "torch.optim.AdamW" and o["lr"] == 0.001 and o["steps"] == 2 and
                                  'OPTIM = {"lr": 1e-3, "weight_decay": 0.0, "betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": False, "foreach": False,\n'
                                  '         "fused": False, "maximize": False}' in esrc and "STEPS = 2" in esrc
                                  and "torch.optim.AdamW(list(masters.values()), **OPTIM)" in esrc
                                  and "strict=False, tie_weights=True" in esrc and "use_cache=False" in esrc)
    rel = "results/inference_cf/p2tta0/pseudo_sealed.json"
    seal = json.loads(SEAL.read_text())
    checks["seal_committed"] = _git("ls-files", rel) == rel and sha(SEAL) == blob_sha("HEAD", rel) and \
        seal["seal_hash"] == canon_digest({k: v for k, v in seal.items() if k != "seal_hash"}) and \
        seal["outcomes_computed"] is False and seal["references_used"] is False and all(seal["checks"].values())
    proc = WhisperProcessor.from_pretrained(model_dir, local_files_only=True)
    gen = GenerationConfig.from_pretrained(model_dir, local_files_only=True)
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    checks["suppression"] = seal["suppression"] == {"suppress": sup, "begin": beg} and seal["eos"] == proc.tokenizer.eos_token_id
    tok_ok = True
    for spec, s in zip(c["reuse"]["rows"], seal["rows"]):
        fr, ar = json.loads((ROOT / spec["forced_row"]).read_text()), json.loads((ROOT / spec["auto_row"]).read_text())
        tok_ok &= (s["utterance_id"] == spec["utterance_id"] and sha(ROOT / spec["forced_row"]) == spec["forced_row_sha256"]
                   and sha(ROOT / spec["auto_row"]) == spec["auto_row_sha256"] and s["y_B"] == fr["systems"]["S0"]["tokens"]
                   and s["y_A"] == ar["systems"]["B0_AUTO"]["tokens"] and s["y_B_terminated"] == fr["systems"]["S0"]["terminated"]
                   and s["y_A_terminated"] == ar["systems"]["B0_AUTO"]["terminated"]
                   and s["y_B_valid_mask"] == own_valid(s["y_B"], sup, beg, proc.tokenizer)
                   and s["y_A_valid_mask"] == own_valid(s["y_A"], sup, beg, proc.tokenizer)
                   and s["audio_full_sha256"] == sha(s["audio_path"]))
    checks["teachers_exact_and_masks"] = bool(tok_ok) and [s["utterance_id"] for s in seal["rows"]] == ids
    seq = json.loads((ROOT / "results/inference_cf/p2seq/run1_audit.json").read_text())
    reuse = json.loads((ROOT / "results/inference_cf/p2seq/reuse_audit.json").read_text())
    checks["reuse_proofs"] = seq["verdict"] == "P2_SEQ_AUDIT: PASS" and reuse["decision_AUTO"] == "REUSE_AUTO_ALL_100"
    import platform
    import transformers
    checks["environment"] = (platform.python_version(), torch.__version__, transformers.__version__) == ("3.11.9", "2.10.0+cu128", "4.57.6")
    checks["no_outcome"] = not any(x.name.startswith("run") for x in BASE.iterdir())
    checks["runner_reference_steering_free"] = runner_clean()["ok"]
    asrc = (ROOT / "experiments/inference_cf_p2tta0_audit.py").read_text()
    checks["auditor_independent"] = re.search(r"^\s*(from|import)\s+\S*(_analyze|episodic_tta)", asrc, re.M) is None
    checks["tests_present"] = (ROOT / "tests/test_inference_cf_p2tta0.py").exists()
    v = "PASS_TO_P2_TTA0" if all(checks.values()) else "BLOCK_BEFORE_P2_TTA0"
    return {"schema": "p2_tta0_prerun_audit_v1", "verdict": v, "checks": checks, "notes": notes, "git_commit": _git("rev-parse", "HEAD")}


# ---- post -----------------------------------------------------------------------------------------------------

def counts(ref, hyp):
    from csasr.evaluation.mer import corpus_mer
    from csasr.evaluation.pier import pier
    p, m = pier([ref], [hyp]), corpus_mer([ref], [hyp])
    return [p["num_poi_errors"], p["num_poi"], m["substitutions"] + m["deletions"] + m["insertions"], m["num_ref_tokens"],
            round(m["zh_cer"] * m["num_zh_ref"]) if m["num_zh_ref"] else 0, m["num_zh_ref"],
            round(m["en_wer"] * m["num_en_ref"]) if m["num_en_ref"] else 0, m["num_en_ref"]]


def own_lev(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)]


def bf16_hash(arrs) -> str:
    import torch
    h = hashlib.sha256()
    for a in arrs:
        x = torch.from_numpy(np.ascontiguousarray(a)).to(torch.bfloat16)
        h.update(str(x.dtype).encode() + str(tuple(x.shape)).encode())
        h.update(x.view(torch.uint8).numpy().tobytes())
    return "sha256:" + h.hexdigest()


def transitions(R, B, S):
    from csasr.evaluation.normalization import EN, ZH, normalize_text, segment_units, tag_unit
    from csasr.evaluation.pier import evaluate_pois, unit_status
    corr = corrupt = bc = zk = zt = ek = et = oh = ob = 0
    per = []
    for r, b, s in zip(R, B, S):
        bb = {p.poi_index: p.correct for p in evaluate_pois(r, b)}
        ss = {p.poi_index: p.correct for p in evaluate_pois(r, s)}
        u = [0] * 8
        for k, v in bb.items():
            bc += v
            corr += (not v) and ss.get(k, False)
            corrupt += v and not ss.get(k, False)
            u[4] += v and not ss.get(k, False)
            u[5] += v
        units = segment_units(normalize_text(r))
        tags = [tag_unit(x) for x in units]
        is_cs = EN in tags and ZH in tags
        ub, us = unit_status(r, b), unit_status(r, s)
        for i, tg in enumerate(tags):
            okb, oks = ub.get(i, (False,))[0], us.get(i, (False,))[0]
            if okb and is_cs and tg == ZH:
                zt += 1
                zk += oks
                u[0] += oks
                u[1] += 1
            if okb and is_cs and tg == EN:
                et += 1
                ek += oks
                u[2] += oks
                u[3] += 1
            if tg != EN and okb:
                ob += 1
                oh += not oks
                u[6] += not oks
                u[7] += 1
        per.append(u)
    return {"corrections": corr, "corruptions": corrupt, "baseline_correct_poi": bc, "zr": zk / zt if zt else None,
            "er": ek / et if et else None, "ohr": oh / ob if ob else None, "corr_rate": corrupt / bc if bc else None,
            "per": np.array(per, dtype=float)}


def cmd_post(args) -> dict:
    import torch
    from transformers import WhisperProcessor
    from experiments.inference_cf_p2_evaluate import load_references
    run = ROOT / args.run
    ana = json.loads((ROOT / args.analysis).read_text())
    c = json.loads((ROOT / CONFIG).read_text())
    TH = c["decision"]["thresholds"]
    m = json.loads((run / "manifest.json").read_text())
    seal = json.loads(SEAL.read_text())
    rt = json.loads((run / "runtime.json").read_text())
    checks = {"manifest_self": m["manifest_hash"] == canon_digest({k: v for k, v in m.items() if k != "manifest_hash"}),
              "sources_at_commit": all("sha256:" + (blob_sha(m["git_commit"], p) or "") == h and
                                       (fhash(ROOT / p) == h or (getattr(args, "rel_grad_tol", None) is not None
                                                                  and (p.endswith(("_analyze.py", "_audit.py")) or p.startswith("tests/"))))
                                       for p, h in m["sources"].items()),
              "analysis_manifest": ana["manifest_hash"] == m["manifest_hash"],
              "panel": sha(ROOT / PANEL) == c["panel"]["byte_sha256"] == m["panel_sha256"].split(":")[-1] and m["ids"] == c["panel"]["ids"],
              "trainable_names": m["trainables"] == [p["name"] for p in c["trainables"]["parameters"]] and len(m["trainables"]) == 194,
              "seal": seal["seal_hash"] == m["seal_hash"] == canon_digest({k: v for k, v in seal.items() if k != "seal_hash"}),
              "runtime_completed": rt["status"] == "completed" and (not rt.get("invalid") if getattr(args, "rel_grad_tol", None) is None else
                                                                    all("live audit" in x for x in rt.get("invalid", []))),
              "nonln_and_final_reset": bool(rt["nonln_unchanged"]) and rt["nonln_hash_start"] == rt["nonln_hash_end"]
              and bool(rt["reset_final_ok"]) and bool(rt["model_grads_none"])}
    ids = m["ids"]
    rows = [json.loads((run / f"rows/{i:02d}.json").read_text()) for i in range(20)]
    checks["rows_ok"] = all(r["status"] == "ok" and r["identity"] == u and set(r["objectives"]) == {"A1", "A2"} for r, u in zip(rows, ids))
    S = {s["utterance_id"]: s for s in seal["rows"]}
    proc = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True)
    tok = proc.tokenizer
    sup, beg = seal["suppression"]["suppress"], seal["suppression"]["begin"]
    # theta0 identity from saved fp32 theta0 values
    with np.load(run / "theta0_ln_fp32.npz") as z:
        th0 = [z[f"p{i:03d}"] for i in range(194)]
    checks["theta0_npz_hash"] = fhash(run / "theta0_ln_fp32.npz") == rt["theta0_ln_fp32_npz_sha256"]
    checks["theta0_bf16_hash"] = bf16_hash(th0) == rt["theta0_ln_hash"]
    checks["theta0_decode_equals_S0"] = all(r["theta0_decode"]["tokens_equal_S0"] and r["theta0_decode"]["terminated_equal_S0"] for r in rows)
    la = rows[0].get("live_audit", {})
    rtol = getattr(args, "rel_grad_tol", None) or 1e-3          # original TTA0 1e-3; P2-TTA0-R funnel 0.02
    checks["live_audit"] = all(k in la and abs(la[k]["primary_loss"] - la[k]["auditor_loss"]) <= 1e-5
                               and la[k]["grad_diff_l2"] <= max(1e-8, rtol * la[k]["auditor_grad_l2"])
                               and abs(la[k]["primary_loss"] - rows[0]["objectives"][k]["losses"][0]) == 0 for k in ("A1", "A2"))
    upd_bad, loss_bad, reset_bad, out_bad = [], [], [], []
    mean_upd = {"A1": [], "A2": []}
    ent = {"theta0": [0.0, 0], "A1": [0.0, 0], "A2": [0.0, 0]}
    for i, (r, u) in enumerate(zip(rows, ids)):
        s = S[u]
        if fhash(run / f"rows/{i:02d}_final_masters_fp32.npz") != r["final_masters_npz_sha256"]:
            upd_bad.append((u, "npz"))
            continue
        with np.load(run / f"rows/{i:02d}_final_masters_fp32.npz") as z:
            fm = {k: z[k] for k in z.files}
        for kind, y in (("A1", s["y_B"]), ("A2", s["y_A"])):
            o = r["objectives"][kind]
            mk = own_valid(y, sup, beg, tok)
            # updates
            mast = [fm[f"{kind}_p{j:03d}"] for j in range(194)]
            dl = [ma.astype(np.float32) - t0.astype(np.float32) for ma, t0 in zip(mast, th0)]
            ml2 = math.sqrt(sum(float((d.astype(np.float64) ** 2).sum()) for d in dl))
            base = math.sqrt(sum(float((t.astype(np.float64) ** 2).sum()) for t in th0))
            eff = [torch.from_numpy(ma).to(torch.bfloat16).float().numpy() - t0 for ma, t0 in zip(mast, th0)]
            el2 = math.sqrt(sum(float((e.astype(np.float64) ** 2).sum()) for e in eff))
            nchg = int(sum(int((e != 0).sum()) for e in eff))
            if abs(ml2 - o["master_delta_l2"]) > 1e-9 * max(1, ml2) or abs(el2 - o["effective_delta_l2"]) > 1e-9 * max(1, el2) \
                    or nchg != o["effective_changed_scalars"] or abs(o["master_delta_rel"] - ml2 / (base + 1e-12)) > 1e-9:
                upd_bad.append((u, kind))
            mean_upd[kind].append(ml2)
            # losses from saved per-position terms
            if o["steps"] != 2 or len(o["losses"]) != 3 or len(o["grad_l2"]) != 2 or not all(o["finite"]) or o["valid"] != sum(mk):
                loss_bad.append((u, kind, "schedule"))
            for k3 in range(3):
                pos = o["positions"][k3]
                if kind == "A1":
                    v = [e for e, ok in zip(pos["entropy"], mk) if ok]
                    L = float(np.mean(np.array(v, dtype=np.float64))) if v else 0.0
                else:
                    v = [-t for t, ok in zip(pos["target_logprob"], mk) if ok]
                    L = float(np.mean(np.array(v, dtype=np.float64))) if v else 0.0
                if abs(L - o["losses"][k3]) > 1e-5:
                    loss_bad.append((u, kind, k3, L, o["losses"][k3]))
            if not (o["reset_ok"] and o["start_hash"] == rt["theta0_ln_hash"] == o["end_hash"] and o["other_versions_unchanged"]):
                reset_bad.append((u, kind))
            # outputs
            if o["length"] != len(o["tokens"]) or (o["terminated"] == "cap") != (len(o["tokens"]) == 200) or \
                    o["text"] != tok.decode(o["tokens"], skip_special_tokens=True) or \
                    o["vs_B0_FORCED"]["equal"] != (o["tokens"] == s["y_B"]) or o["vs_B0_FORCED"]["levenshtein"] != own_lev(o["tokens"], s["y_B"]) or \
                    o["vs_B0_AUTO"]["levenshtein"] != own_lev(o["tokens"], s["y_A"]):
                out_bad.append((u, kind))
            sev = len(s["y_B"]) >= 10 and o["terminated"] == "eos" and len(o["tokens"]) <= math.floor(0.5 * len(s["y_B"]))
            if sev != o["severe_truncation"]:
                out_bad.append((u, kind, "severe"))
        # common-y_B entropy (A1 theta0 and theta2; A2 theta2), own sums over the y_B valid set
        mB = own_valid(s["y_B"], sup, beg, tok)
        for key, src, th in (("theta0", "A1", "theta0"), ("A1", "A1", "theta2"), ("A2", "A2", "theta2")):
            e = r["objectives"][src]["common_y_B"][th]["entropy"]
            vv = [x for x, ok in zip(e, mB) if ok]
            ent[key][0] += float(np.sum(np.array(vv, dtype=np.float64)))
            ent[key][1] += len(vv)
        if r["objectives"]["A1"]["positions"][0]["entropy"] != r["objectives"]["A1"]["common_y_B"]["theta0"]["entropy"] or \
                r["objectives"]["A1"]["positions"][2]["entropy"] != r["objectives"]["A1"]["common_y_B"]["theta2"]["entropy"]:
            loss_bad.append((u, "common_y_B_A1_not_loss_forward"))
    checks["updates_recomputed"] = not upd_bad
    checks["losses_recomputed"] = not loss_bad
    checks["resets"] = not reset_bad
    checks["outputs"] = not out_bad
    cnt = rt["counters"]
    checks["compute_counts"] = (cnt["A1"]["backwards"] + cnt["A2"]["backwards"] + cnt["audit"]["backwards"] == 82
                                and cnt["A1"]["optimizer_steps"] + cnt["A2"]["optimizer_steps"] == 80
                                and cnt["A1"]["teacher_forwards"] + cnt["A2"]["teacher_forwards"] + cnt["audit"]["forwards"] == 142
                                and cnt["A1"]["final_decodes"] + cnt["A2"]["final_decodes"] == 40 and cnt["theta0_integrity_decodes"] == 20
                                and cnt["encoder_passes"] == 20)
    # independent metrics
    refs = load_references()
    R = [refs[u]["reference"] for u in ids]
    D = [refs[u]["dialogue_id"] for u in ids]
    checks["dialogues"] = len(set(D)) == 20 and D == [S[u]["dialogue_id"] for u in ids]
    H = {"B0_FORCED": [S[u]["y_B_text"] for u in ids], "B0_AUTO": [S[u]["y_A_text"] for u in ids],
         "A1": [r["objectives"]["A1"]["text"] for r in rows], "A2": [r["objectives"]["A2"]["text"] for r in rows]}
    TERM = {"B0_FORCED": [S[u]["y_B_terminated"] for u in ids], "A1": [r["objectives"]["A1"]["terminated"] for r in rows],
            "A2": [r["objectives"]["A2"]["terminated"] for r in rows]}
    C = {k: np.array([counts(a, b) for a, b in zip(R, H[k])], dtype=float) for k in H}
    rate = lambda c_, n, d: c_[:, n].sum() / c_[:, d].sum()
    mine = {k: {"pier": rate(C[k], 0, 1), "mer": rate(C[k], 2, 3), "zh_cer": rate(C[k], 4, 5), "en_wer": rate(C[k], 6, 7),
                "poi_errors": int(C[k][:, 0].sum())} for k in C}
    checks["metrics_agree"] = max(abs(mine[k][n] - ana["metrics"][k][n]) for k in mine for n in ("pier", "mer", "zh_cer", "en_wer")) <= 1e-12 \
        and all(mine[k]["poi_errors"] == ana["metrics"][k]["num_poi_errors"] for k in mine)
    checks["populations_nonempty"] = C["B0_FORCED"][:, 1].sum() > 0 and C["B0_FORCED"][:, 5].sum() > 0 and C["B0_FORCED"][:, 7].sum() > 0
    labels, safety_all, useful_all, cbs = {}, {}, {}, {}
    rng = np.random.default_rng(240924)
    keys = sorted(set(D))
    pos = {k: D.index(k) for k in keys}
    drs = [[pos[keys[j]] for j in rng.integers(0, 20, 20)] for _ in range(2000)]
    bdiff = []
    for kind in ("A1", "A2"):
        tr = transitions(R, H["B0_FORCED"], H[kind])
        p = {"mer_inc": mine[kind]["mer"] - mine["B0_FORCED"]["mer"], "zh_inc": mine[kind]["zh_cer"] - mine["B0_FORCED"]["zh_cer"],
             "net": mine["B0_FORCED"]["poi_errors"] - mine[kind]["poi_errors"]}
        added_caps = sum(a == "cap" and b != "cap" for a, b in zip(TERM[kind], TERM["B0_FORCED"]))
        sev = sum(len(S[u]["y_B"]) >= 10 and r["objectives"][kind]["terminated"] == "eos"
                  and len(r["objectives"][kind]["tokens"]) <= math.floor(0.5 * len(S[u]["y_B"])) for r, u in zip(rows, ids))
        safety = {"MER": p["mer_inc"] <= TH["max_MER_increase"] + G, "ZH_CER": p["zh_inc"] <= TH["max_ZH_CER_increase"] + G,
                  "ZH_ret": tr["zr"] is None or tr["zr"] >= TH["min_matrix_ZH_retention"] - G,
                  "EN_ret": tr["er"] is None or tr["er"] >= TH["min_embedded_EN_retention"] - G,
                  "outside": tr["ohr"] is None or tr["ohr"] <= TH["max_outside_POI_harm_rate"] + G,
                  "corruption": tr["corr_rate"] is None or tr["corr_rate"] <= TH["max_POI_corruption_rate"] + G,
                  "caps": added_caps <= TH["max_additional_caps"], "severe": sev <= TH["max_new_severe_truncations"]}
        d_mix = C[kind][:, 2] - C["B0_FORCED"][:, 2]
        imp, deg = int((d_mix < 0).sum()), int((d_mix > 0).sum())
        useful = p["net"] >= TH["min_net_POI_error_reduction"] or (imp >= TH["alternative_min_MER_improved_utterances"] and imp > deg)
        ap = ana["objectives"][kind]["point"]
        checks[f"{kind}_points_agree"] = (tr["corrections"], tr["corruptions"]) == (ana["objectives"][kind]["transitions"]["corrections"],
                                                                                   ana["objectives"][kind]["transitions"]["corruptions"]) \
            and tr["zr"] == ap["zh_retention"] and tr["er"] == ap["en_retention"] and tr["ohr"] == ap["outside_harm_rate"] \
            and added_caps == ap["added_caps"] and sev == ap["new_severe_truncations"] and (imp, deg) == (ap["improved_utterances"], ap["degraded_utterances"])
        cb = False
        if kind == "A1":
            h0 = ent["theta0"][0] / ent["theta0"][1] if ent["theta0"][1] else None
            h2 = ent["A1"][0] / ent["A1"][1] if ent["A1"][1] else None
            cb = bool(h0 is not None and h2 is not None and h0 > 0 and (h0 - h2) / h0 >= TH["EM_entropy_relative_reduction_min"] - G
                      and not all(safety.values()))
            cbs = {"theta0": h0, "theta2": h2, "flag": cb}
            ac = ana["objectives"]["A1"]["confirmation_bias"]
            checks["entropy_agree"] = h0 is not None and abs(h0 - ac["theta0"]) <= 1e-9 and abs(h2 - ac["theta2"]) <= 1e-9
        ok_stage = True
        names = ("TTA0_EM_VIABLE", "TTA0_EM_NOT_VIABLE", "TTA0_EM_CONFIRMATION_BIAS") if kind == "A1" else ("TTA0_AC_VIABLE", "TTA0_AC_NOT_VIABLE")
        labels[kind] = names[2] if cb else (names[0] if useful and all(safety.values()) else names[1])
        safety_all[kind], useful_all[kind] = safety, useful
        # own bootstrap agreement
        for n, (a, b) in {"pier": (0, 1), "mer": (2, 3), "zh_cer": (4, 5), "en_wer": (6, 7)}.items():
            for base_ in ("B0_FORCED", "B0_AUTO"):
                vals = [C[kind][ix, a].sum() / C[kind][ix, b].sum() - C[base_][ix, a].sum() / C[base_][ix, b].sum()
                        for ix in drs if C[kind][ix, b].sum() and C[base_][ix, b].sum()]
                ref_ = ana["objectives"][kind]["bootstrap"][f"{kind}_minus_{base_}"][n]
                if ref_["ci95"] is None:
                    bdiff.append(0.0 if not vals else 1.0)
                else:
                    bdiff += [abs(np.quantile(vals, .025) - ref_["ci95"][0]), abs(np.quantile(vals, .975) - ref_["ci95"][1])]
    checks["bootstrap_agrees"] = max(bdiff) <= 1e-9
    valid = all(checks.values())
    if not valid:
        labels = {k: "P2_TTA0_INVALID" for k in labels}
    viable = [k for k in ("A1", "A2") if labels[k].endswith("_VIABLE") and "NOT" not in labels[k]]
    if not valid:
        stage, sel = "P2_TTA0_INVALID", None
    elif not viable:
        stage, sel = "P2_TTA0_NO_VIABLE_OBJECTIVE", None
    elif len(viable) == 1:
        stage, sel = "P2_TTA0_OBJECTIVE_SELECTED", viable[0]
    else:
        stage, sel = "P2_TTA0_OBJECTIVE_SELECTED", "A2"
        for a, b in ((mine["A1"]["mer"], mine["A2"]["mer"]), (mine["A1"]["pier"], mine["A2"]["pier"]),
                     (mine["A1"]["zh_cer"], mine["A2"]["zh_cer"]), (float(np.mean(mean_upd["A1"])), float(np.mean(mean_upd["A2"])))):
            if abs(a - b) > G:
                sel = "A1" if a < b else "A2"
                break
    checks["labels_agree"] = labels == ana["labels"]
    checks["selection_agrees"] = stage == ana["label"] and sel == ana["selected"]
    verdict = "P2_TTA0_AUDIT: PASS" if all(checks.values()) else "P2_TTA0_AUDIT: BLOCK"
    return {"schema": "p2_tta0_post_audit_v1", "verdict": verdict, "label": stage, "selected": sel, "labels": labels,
            "checks": checks, "independent": {"metrics": mine, "safety": safety_all, "useful": useful_all, "confirmation_bias": cbs,
                                              "mean_master_update_l2": {k: float(np.mean(v)) if v else None for k, v in mean_upd.items()}},
            "failures": {"updates": upd_bad[:10], "losses": loss_bad[:10], "resets": reset_bad[:10], "outputs": out_bad[:10]},
            "git_commit": _git("rev-parse", "HEAD")}


def cmd_invalid(args) -> dict:
    """Reference-free independent confirmation of a technically INVALID run: recomputes the frozen first-row live
    loss/gradient criterion from stored scalars, theta0 identity, updates and losses from saved terms/masters, resets,
    output well-formedness, compute counts and source provenance, then derives the stage label itself
    (frozen precedence: any technical failure => P2_TTA0_INVALID). Loads no reference; computes no metric."""
    run = ROOT / args.run
    rec = json.loads((ROOT / args.record).read_text())
    m = json.loads((run / "manifest.json").read_text())
    seal = json.loads(SEAL.read_text())
    rt = json.loads((run / "runtime.json").read_text())
    c = json.loads((ROOT / CONFIG).read_text())
    runtime_sources = [p for p in m["sources"] if not p.endswith(("_analyze.py", "_audit.py"))]
    checks = {"manifest_self": m["manifest_hash"] == canon_digest({k: v for k, v in m.items() if k != "manifest_hash"}),
              "sources_blobs_at_manifest_commit": all("sha256:" + (blob_sha(m["git_commit"], p) or "") == h for p, h in m["sources"].items()),
              "runtime_sources_unchanged": all(fhash(ROOT / p) == m["sources"][p] for p in runtime_sources),
              "record_manifest": rec["manifest_hash"] == m["manifest_hash"] and rec["references_loaded"] is False
              and rec["canonical_metrics_computed"] is False,
              "panel": sha(ROOT / PANEL) == c["panel"]["byte_sha256"] and m["ids"] == c["panel"]["ids"],
              "trainable_names": m["trainables"] == [p["name"] for p in c["trainables"]["parameters"]],
              "seal": seal["seal_hash"] == m["seal_hash"] == canon_digest({k: v for k, v in seal.items() if k != "seal_hash"})}
    ids = m["ids"]
    rows = [json.loads((run / f"rows/{i:02d}.json").read_text()) for i in range(20)]
    eng = {"rows_complete": all(r["status"] == "ok" and r["identity"] == u and set(r["objectives"]) == {"A1", "A2"} for r, u in zip(rows, ids)),
           "runtime_completed": rt["status"] == "completed",
           "nonln_and_final_reset": bool(rt["nonln_unchanged"]) and rt["nonln_hash_start"] == rt["nonln_hash_end"]
           and bool(rt["reset_final_ok"]) and bool(rt["model_grads_none"]),
           "theta0_decode_equals_S0": all(r["theta0_decode"]["tokens_equal_S0"] and r["theta0_decode"]["terminated_equal_S0"] for r in rows)}
    with np.load(run / "theta0_ln_fp32.npz") as z:
        th0 = [z[f"p{i:03d}"] for i in range(194)]
    eng["theta0_bf16_hash"] = bf16_hash(th0) == rt["theta0_ln_hash"]
    S = {s_["utterance_id"]: s_ for s_ in seal["rows"]}
    from transformers import WhisperProcessor
    tok = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True).tokenizer
    sup, beg = seal["suppression"]["suppress"], seal["suppression"]["begin"]
    bad = {"updates": [], "losses": [], "resets": [], "outputs": []}
    for i, (r, u) in enumerate(zip(rows, ids)):
        s_ = S[u]
        with np.load(run / f"rows/{i:02d}_final_masters_fp32.npz") as z:
            fm = {k: z[k] for k in z.files}
        if fhash(run / f"rows/{i:02d}_final_masters_fp32.npz") != r["final_masters_npz_sha256"]:
            bad["updates"].append((u, "npz"))
        for kind, y in (("A1", s_["y_B"]), ("A2", s_["y_A"])):
            o = r["objectives"][kind]
            mk = own_valid(y, sup, beg, tok)
            mast = [fm[f"{kind}_p{j:03d}"] for j in range(194)]
            ml2 = math.sqrt(sum(float(((ma.astype(np.float32) - t0) .astype(np.float64) ** 2).sum()) for ma, t0 in zip(mast, th0)))
            import torch
            eff = [torch.from_numpy(ma).to(torch.bfloat16).float().numpy() - t0 for ma, t0 in zip(mast, th0)]
            el2 = math.sqrt(sum(float((e.astype(np.float64) ** 2).sum()) for e in eff))
            if abs(ml2 - o["master_delta_l2"]) > 1e-9 * max(1, ml2) or abs(el2 - o["effective_delta_l2"]) > 1e-9 * max(1, el2) \
                    or int(sum(int((e != 0).sum()) for e in eff)) != o["effective_changed_scalars"]:
                bad["updates"].append((u, kind))
            if o["steps"] != 2 or len(o["losses"]) != 3 or len(o["grad_l2"]) != 2 or not all(o["finite"]) or o["valid"] != sum(mk) \
                    or not all(math.isfinite(x) for x in o["losses"] + o["grad_l2"]):
                bad["losses"].append((u, kind, "schedule"))
            for k3 in range(3):
                pos = o["positions"][k3]
                v = [e for e, ok in zip(pos["entropy"], mk) if ok] if kind == "A1" else [-t for t, ok in zip(pos["target_logprob"], mk) if ok]
                L = float(np.mean(np.array(v, dtype=np.float64))) if v else 0.0
                if abs(L - o["losses"][k3]) > 1e-5:
                    bad["losses"].append((u, kind, k3))
            if not (o["reset_ok"] and o["start_hash"] == rt["theta0_ln_hash"] == o["end_hash"] and o["other_versions_unchanged"]):
                bad["resets"].append((u, kind))
            if o["length"] != len(o["tokens"]) or (o["terminated"] == "cap") != (len(o["tokens"]) == 200) or \
                    o["text"] != tok.decode(o["tokens"], skip_special_tokens=True) or o["vs_B0_FORCED"]["levenshtein"] != own_lev(o["tokens"], s_["y_B"]) \
                    or o["vs_B0_AUTO"]["levenshtein"] != own_lev(o["tokens"], s_["y_A"]):
                bad["outputs"].append((u, kind))
    eng.update({f"{k}_recomputed": not v for k, v in bad.items()})
    cnt = rt["counters"]
    eng["compute_counts"] = (cnt["A1"]["backwards"] + cnt["A2"]["backwards"] + cnt["audit"]["backwards"] == 82
                             and cnt["A1"]["optimizer_steps"] + cnt["A2"]["optimizer_steps"] == 80
                             and cnt["A1"]["teacher_forwards"] + cnt["A2"]["teacher_forwards"] + cnt["audit"]["forwards"] == 142
                             and cnt["A1"]["final_decodes"] + cnt["A2"]["final_decodes"] == 40 and cnt["theta0_integrity_decodes"] == 20)
    # frozen first-row live objective/gradient criterion, recomputed from stored scalars
    la = rows[0]["live_audit"]
    live = {}
    for k in ("A1", "A2"):
        x = la[k]
        live[k] = {"loss_abs_diff": abs(x["primary_loss"] - x["auditor_loss"]), "loss_ok": abs(x["primary_loss"] - x["auditor_loss"]) <= 1e-5,
                   "grad_rel": x["grad_diff_l2"] / x["auditor_grad_l2"], "grad_ok": x["grad_diff_l2"] <= max(1e-8, 1e-3 * x["auditor_grad_l2"]),
                   "primary_loss_is_L0": x["primary_loss"] == rows[0]["objectives"][k]["losses"][0]}
    eng["live_objective_gradient_check"] = all(v["loss_ok"] and v["grad_ok"] for v in live.values())
    own_label = "P2_TTA0_INVALID" if not all(eng.values()) else "NOT_INVALID"
    checks["label_agrees"] = own_label == rec["label"] == "P2_TTA0_INVALID"
    checks["failed_checks_agree"] = sorted(k for k, v in eng.items() if not v) == ["live_objective_gradient_check"] and \
        sorted(rec["failed_checks"]) == ["live_audit_pass", "no_runtime_invalid"] and \
        all("live audit" in x for x in rt.get("invalid", [])) and rec["runtime_invalid"] == rt.get("invalid")
    verdict = "P2_TTA0_AUDIT: PASS" if all(checks.values()) else "P2_TTA0_AUDIT: BLOCK"
    return {"schema": "p2_tta0_invalid_audit_v1", "verdict": verdict, "label": own_label, "selected": None, "labels": {"A1": own_label, "A2": own_label},
            "checks": checks, "engineering": eng, "live": live, "failures": {k: v[:10] for k, v in bad.items()},
            "references_loaded": False, "git_commit": _git("rev-parse", "HEAD")}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prerun").add_argument("--out", required=True)
    p = sub.add_parser("post")
    p.add_argument("--run", required=True)
    p.add_argument("--analysis", required=True)
    p.add_argument("--out", required=True)
    q = sub.add_parser("invalid")
    q.add_argument("--run", required=True)
    q.add_argument("--record", required=True)
    q.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"prerun": cmd_prerun, "post": cmd_post, "invalid": cmd_invalid}[args.cmd](args)
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("audit output exists; never overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, sort_keys=True, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    print(json.dumps({k: res[k] for k in res if k in ("verdict", "label", "selected", "labels")}, indent=1))
    if res["verdict"] not in ("PASS_TO_P2_TTA0", "P2_TTA0_AUDIT: PASS"):
        print(json.dumps({k: v for k, v in res["checks"].items() if not v}, indent=1))


if __name__ == "__main__":
    main()
