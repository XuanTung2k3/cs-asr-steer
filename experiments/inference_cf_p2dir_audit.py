#!/usr/bin/env python
"""Independent P2-DIR auditor. Does NOT import inference_cf_p2dir_analyze (nor its bootstrap,
decision or metric functions); every statistic is reconstructed here from the raw saved artifacts.

Subcommands
-----------
``prerun``  CPU: frozen config/spec/population/construction/fold-plan identity, firewall, the
            reference-free construction surface of D0/D1/D2 (AST), and test-suite presence.
            Verdict PASS_TO_P2_DIR_EXP1 / BLOCK_BEFORE_P2_DIR_EXP1 (spec label PASS_TO_P2_DIR_RUN).
``spot``    GPU (same allocation as Exp-1): independent scratch-forward recomputation of the D2
            readout objective/gradient/direction at 10 positions per stratum chosen by
            sha256("P2DIR-audit-v1|uid|t"), on a separately built no-grad B cache.
``exp1``    CPU: population, fold reconstruction from extraction states, D0/D2 provenance from saved
            vectors, realized energy from saved before/after states, all evaluator metrics from the
            saved logits, the family-10 bootstrap, qualification and selection. ``P2_DIR_AUDIT: PASS``
            (stage EXP1) only when every check holds and every statistic/label agrees.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

STRATA = ("EN-confusion", "EN-correct", "ZH-correct")
ARMS = ("D0", "D1", "D2")
CONSTRUCTION_MODULES = ("src/csasr/inference_cf/directions.py", "src/csasr/inference_cf/readout.py",
                        "src/csasr/inference_cf/unique.py")
FORBIDDEN_NAMES = ("target_ids", "competitor", "reference", "transcript", "alignment", "oracle", "Y_ref",
                   "evaluator_targets", "SiteGradientProbe", "inference_cf_p2rj", "margin_ref", "ref_ids")
ALLOWED_FIELDS = {"utterance_id", "dialogue_id", "t", "stratum"}


def canon(obj) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                                                 allow_nan=False).encode()).hexdigest()


def fhash(p) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ahash(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode() + b"|" + repr(tuple(a.shape)).encode() + b"|")
    h.update(a.tobytes())
    return "sha256:" + h.hexdigest()


def git_blob_hash(commit: str, rel: str) -> str:
    blob = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=ROOT, capture_output=True).stdout
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def construction_surface_clean() -> dict:
    """No reference/evaluator identifier in the D0/D1/D2 construction modules (names, attrs, strings,
    imports)."""
    bad = {}
    for rel in CONSTRUCTION_MODULES:
        tree = ast.parse((ROOT / rel).read_text())
        hits = set()
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
            elif isinstance(node, ast.arg):
                names.append(node.arg)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names += [a.name for a in node.names] + ([node.module] if getattr(node, "module", None) else [])
            elif isinstance(node, ast.FunctionDef):
                names.append(node.name)
            for n in names:
                for f in FORBIDDEN_NAMES:
                    if f.lower() in str(n).lower():
                        hits.add(n)
        if hits:
            bad[rel] = sorted(hits)
    return {"ok": not bad, "violations": bad}


def runner_phase1_clean() -> dict:
    """The exp1 pulse pass and extraction never touch evaluator references."""
    tree = ast.parse((ROOT / "experiments/inference_cf_p2dir.py").read_text())
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("extract_utterance", "exp1_utterance"):
            src = ast.get_source_segment((ROOT / "experiments/inference_cf_p2dir.py").read_text(), node)
            out[node.name] = not any(f in src for f in ("target_ids", "competitor", "positions.json", "p2rj"))
    return {"ok": all(out.values()) and len(out) == 2, "functions": out}


# ---- independent D1 reconstruction -----------------------------------------------------------------

def refit(HA: np.ndarray, HB: np.ndarray, rank: int = 32) -> dict:
    tol = 1e-10
    HA, HB = HA.astype(np.float64), HB.astype(np.float64)
    if HA.shape[0] < 32 or HB.shape[0] < 32:
        return {"status": "invalid", "reason": "insufficient_count"}
    if np.linalg.matrix_rank(HA) < rank or np.linalg.matrix_rank(HB) < rank:
        return {"status": "invalid", "reason": "numerical_rank"}
    CA, CB = HA.T @ HA / HA.shape[0], HB.T @ HB / HB.shape[0]
    CA, CB = (CA + CA.T) / 2, (CB + CB.T) / 2
    eA, VA = np.linalg.eigh(CA)
    eB, VB = np.linalg.eigh(CB)
    eA, VA, eB, VB = eA[::-1], VA[:, ::-1], eB[::-1], VB[:, ::-1]
    for nm, e in (("A", eA), ("B", eB)):
        if not (e[rank - 1] > tol * e[0]):
            return {"status": "invalid", "reason": f"eigen_floor_{nm}"}
        if not (e[rank - 1] - e[rank] > tol * e[0]):
            return {"status": "invalid", "reason": f"eigen_gap_{nm}"}
    UA, UB = VA[:, :rank], VB[:, :rank]
    P, s, Qt = np.linalg.svd(UA.T @ UB, full_matrices=False)
    s = np.minimum(np.maximum(s, 0.0), 1.0)
    a, b = UA @ P, UB @ Qt.T
    I = np.eye(rank)
    err = max(np.abs(M.T @ M - I).max() for M in (UA, UB, a, b))
    err = max(err, np.abs(a.T @ b - np.diag(s)).max())
    if err > 1e-8:
        return {"status": "invalid", "reason": "orthonormality"}
    EA = np.array([a[:, i] @ CA @ a[:, i] for i in range(rank)])
    sc = EA * (1 - s ** 2)
    i = int(np.argmax(sc))
    srt = np.sort(sc)
    if not sc[i] > tol * EA.max():
        return {"status": "invalid", "reason": "score_floor"}
    if not (srt[-1] - srt[-2]) > tol * EA.max():
        return {"status": "invalid", "reason": "score_tie"}
    if not min(abs(s[j] - s[i]) for j in range(rank) if j != i) > tol:
        return {"status": "invalid", "reason": "sigma_degenerate"}
    c = HA.mean(axis=0) - HB.mean(axis=0)
    cn = np.linalg.norm(c)
    v = a[:, i]
    dot = float(v @ c)
    if not cn > tol or not abs(dot) > tol * cn:
        return {"status": "invalid", "reason": "sign"}
    v = (v if dot > 0 else -v) / np.linalg.norm(v)
    return {"status": "ok", "index": i, "vector": v.astype(np.float32), "sigma": float(s[i]), "score": float(sc[i])}


# ---- independent evaluator metrics -----------------------------------------------------------------

def bf16(a) -> np.ndarray:
    return (np.asarray(a, dtype=np.int16).view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def lse(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    return float(m + np.log(np.exp(x - m).sum())) if np.isfinite(m) else float(m)


def metrics(z, t, sup, beg, part, Y, c) -> dict:
    zp = np.asarray(z, dtype=np.float64).copy()
    if sup:
        zp[sup] = -np.inf
    if t == 0 and beg:
        zp[beg] = -np.inf
    Lz = lse(zp)
    ref = lse(zp[Y])
    top = int(np.argmax(zp))
    lpe, lpm = lse(zp[part["embedded_ids"]]) - Lz, lse(zp[part["matrix_ids"]]) - Lz
    le = math.log(1e-12)
    return {"m": ref - float(zp[c]), "logp_ref": ref - Lz, "in_ref": top in set(Y), "top": top,
            "J": float(np.logaddexp(lpe, le) - np.logaddexp(lpm, le))}


def ci(pairs, keys, W, alpha):
    g = defaultdict(list)
    for d, v in pairs:
        g[d].append(v)
    means = np.array([np.mean(g[k]) if k in g else np.nan for k in keys])
    use = ~np.isnan(means)
    num = (W[:, use] * means[use]).sum(axis=1)
    den = W[:, use].sum(axis=1)
    ok = den > 0
    dr = num[ok] / den[ok]
    return float(means[use].mean()), [float(np.quantile(dr, alpha / 2)), float(np.quantile(dr, 1 - alpha / 2))], int(ok.sum())


# ---- subcommands -------------------------------------------------------------------------------------

def cmd_prerun(args) -> dict:
    checks, notes = {}, {}
    cfg_rel = "configs/inference_cf/p2_dir_direction_identification.json"
    cfg = json.loads((ROOT / cfg_rel).read_text())
    freeze = "3ef654246c05307a6cddd5fe706c089d696cbbee"
    for rel in (cfg_rel, "docs/inference_cf/P2_DIR_DIRECTION_IDENTIFICATION_SPEC.md", "docs/inference_cf/P2_DIR_CODEX_DESIGN.md"):
        checks[f"unchanged_since_freeze:{rel}"] = fhash(ROOT / rel) == git_blob_hash(freeze, rel)
    for rel, h in cfg["source_sha256"].items():
        if rel.startswith("results/") or rel.startswith("src/csasr/inference_cf/core") or rel.startswith("src/csasr/experiments") or rel.startswith("src/csasr/basis_a6"):
            checks[f"design_anchor:{rel}"] = fhash(ROOT / rel) == "sha256:" + h
    for rel in ("experiments/inference_cf_cached.py", "experiments/inference_cf_p2r.py", "experiments/inference_cf_p2rj.py"):
        checks[f"reused_module_unchanged:{rel}"] = fhash(ROOT / rel) == "sha256:" + cfg["source_sha256"][rel]
    for rel in ("src/csasr/lss/sites.py", "src/csasr/models/hooks.py"):
        checks[f"actuator_unchanged:{rel}"] = fhash(ROOT / rel) == git_blob_hash(freeze, rel)
    pos = json.loads((ROOT / "results/inference_cf/p2rj/positions.json").read_text())
    checks["positions_hash"] = pos["positions_hash"] == cfg["positions_hash"] == canon({k: v for k, v in pos.items() if k != "positions_hash"})
    con = json.loads((ROOT / "results/inference_cf/p2dir/construction_population.json").read_text())
    checks["construction_self_hash"] = con["construction_hash"] == canon({k: v for k, v in con.items() if k != "construction_hash"})
    checks["construction_allowlist"] = all(set(p) <= ALLOWED_FIELDS for p in con["positions"])
    checks["construction_matches_positions"] = [(p["utterance_id"], p["dialogue_id"], p["t"], p["stratum"]) for p in pos["positions"]] == \
        [(p["utterance_id"], p["dialogue_id"], p["t"], p["stratum"]) for p in con["positions"]]
    cnt = defaultdict(int)
    for p in con["positions"]:
        cnt[p["stratum"]] += 1
    checks["strata_60_each"] = dict(cnt) == cfg["strata_counts"]
    dl = sorted({p["dialogue_id"] for p in con["positions"]})
    fold_ok = True
    for k in dl:
        f = con["folds"][k]
        A = sorted([p["utterance_id"], p["t"]] for p in con["positions"] if p["dialogue_id"] != k and p["stratum"] == "EN-correct")
        B = sorted([p["utterance_id"], p["t"]] for p in con["positions"] if p["dialogue_id"] != k and p["stratum"] == "ZH-correct")
        ex = {(p["utterance_id"], p["t"]) for p in con["positions"] if p["dialogue_id"] == k}
        fold_ok &= sorted(f["A"]) == A and sorted(f["B"]) == B and {tuple(x) for x in f["excluded"]} == ex
        fold_ok &= len(A) >= 32 and len(B) >= 32
        fold_ok &= not ({tuple(x) for x in f["A"]} | {tuple(x) for x in f["B"]}) & ex
    checks["fold_plan_leave_one_dialogue_out"] = bool(fold_ok)
    notes["folds"] = len(dl)
    panel = json.loads((ROOT / "results/inference_cf/p0_r2/inference_panel.json").read_text())
    checks["firewall_d_dev_select"] = panel["role"] == "D-dev-select" and set(con["utterances"]) <= {r["utterance_id"] for r in panel["rows"]}
    surf = construction_surface_clean()
    checks["construction_modules_reference_free"] = surf["ok"]
    notes["surface"] = surf
    ph1 = runner_phase1_clean()
    checks["runner_phase1_reference_free"] = ph1["ok"]
    for t in ("tests/test_inference_cf_p2dir_directions.py", "tests/test_inference_cf_p2dir_protocol.py",
              "tests/test_inference_cf_p2dir_audit.py"):
        checks[f"test_present:{t}"] = (ROOT / t).exists()
    checks["candidates_exactly_D0_D1_D2"] = cfg["candidates"] == ["D0", "D1", "D2"]
    checks["layer16_normpreserve_no_rescale"] = cfg["layer"] == 16 and cfg["norm_preserve"] and not cfg["depth_rescale"]
    checks["e_star"] = cfg["energy"]["e_star"] == math.sqrt(1203.3762345332195 / 949)
    import re
    analyze_src = (ROOT / "experiments/inference_cf_p2dir_audit.py").read_text()
    checks["auditor_independent_of_analysis"] = re.search(r"^\s*(from|import)\s+\S*p2dir_analyze", analyze_src, re.M) is None
    verdict = "PASS_TO_P2_DIR_EXP1" if all(checks.values()) else "BLOCK_BEFORE_P2_DIR_EXP1"
    return {"schema": "p2dir_prerun_audit_v1", "verdict": verdict, "checks": checks, "notes": notes,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}


def spot_positions(con: dict) -> list[dict]:
    out = []
    for s in STRATA:
        ps = [p for p in con["positions"] if p["stratum"] == s]
        ps.sort(key=lambda p: hashlib.sha256(f"P2DIR-audit-v1|{p['utterance_id']}|{p['t']}".encode()).hexdigest())
        out += ps[:10]
    return out


def cmd_spot(args) -> dict:
    """GPU: independent readout recomputation (own no-grad cache, own probe, own objective)."""
    import copy
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.models.whisper import batch_model_inputs, load_whisper
    from csasr.utils.config import load_config
    run = ROOT / args.run
    man = json.loads((run / "manifest.json").read_text())
    con = json.loads((ROOT / man["construction"]).read_text())
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    model = bundle.model
    model.requires_grad_(False)
    part = tokenizer_partition(bundle.processor.tokenizer)
    gen = model.generation_config
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    base = ROOT / "results/inference_cf/p2_A_r1_L16"
    bidx = {r["utterance_id"]: i for i, r in enumerate(json.loads((base / "panel.json").read_text())["rows"])}
    audio = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p0_r2/inference_panel.json").read_text())["rows"]}
    uidx = {u: i for i, u in enumerate(con["utterances"])}
    cB = [50258, 50260, 50360, 50364]
    dev = next(model.parameters()).device
    layer = model.model.decoder.layers[16]
    res = []
    for p in spot_positions(con):
        uid, t = p["utterance_id"], int(p["t"])
        toks = json.loads((base / "rows" / f"{bidx[uid]:03d}.json").read_text())["systems"]["B0M_L16"]["tokens"]
        with np.load(run / "rows" / f"{uidx[uid]:03d}.npz") as z:
            saved = {k: z[k] for k in z.files if k.startswith(f"t{t}_")}
        row = json.loads((run / "rows" / f"{uidx[uid]:03d}.json").read_text())["positions"][str(t)]
        inputs = batch_model_inputs(bundle, [audio[uid]["audio_path"]])
        with torch.no_grad():
            enc = BaseModelOutput(last_hidden_state=model.model.encoder(
                input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state)
            cache, fed = None, []
            feed = list(cB)
            for s in range(t):                      # advance the independent cache to just before step t
                o = model(encoder_outputs=enc, decoder_input_ids=torch.tensor([feed], device=dev), past_key_values=cache,
                          use_cache=True, cache_position=torch.arange(len(fed), len(fed) + len(feed), device=dev),
                          output_attentions=True, return_dict=True)
                cache = o.past_key_values
                fed += feed
                feed = [toks[s]]
        scratch = copy.deepcopy(cache)
        delta = torch.zeros(layer.encoder_attn.out_proj.out_features, device=dev, dtype=torch.float32, requires_grad=True)
        box = {}

        def ln_pre(_m, a):
            box["q"] = a[0]

        def post(_m, _i, o):
            u = o[0] if isinstance(o, tuple) else o
            box["site"] = (box["q"] + u)[0, -1].detach().float().cpu()
            add = torch.zeros_like(u)
            add[0, -1] = 1
            u2 = u + add * delta.to(u.dtype)
            return (u2,) + tuple(o[1:]) if isinstance(o, tuple) else u2
        h1 = layer.encoder_attn_layer_norm.register_forward_pre_hook(ln_pre)
        h2 = layer.encoder_attn.register_forward_hook(post)
        try:
            with torch.enable_grad():
                o = model(encoder_outputs=enc, decoder_input_ids=torch.tensor([feed], device=dev), past_key_values=scratch,
                          use_cache=True, cache_position=torch.arange(len(fed), len(fed) + len(feed), device=dev),
                          output_attentions=True, return_dict=True)
                z = o.logits[0, -1].float()
                zz = z.clone()
                mask = torch.zeros_like(zz, dtype=torch.bool)
                if sup:
                    mask[sup] = True
                zz = zz.masked_fill(mask, float("-inf"))
                lsm = torch.log_softmax(zz, -1)
                lpe = torch.logsumexp(lsm[part["embedded_ids"]], 0)
                lpm = torch.logsumexp(lsm[part["matrix_ids"]], 0)
                e = torch.tensor(math.log(1e-12), device=dev)
                J = torch.logaddexp(lpe, e) - torch.logaddexp(lpm, e)
                (g,) = torch.autograd.grad(J, delta)
        finally:
            h1.remove()
            h2.remove()
        grads_none = all(q.grad is None for q in model.parameters())
        h = box["site"].double().numpy()
        gg = g.detach().double().cpu().numpy()
        hh = h / np.linalg.norm(h)
        gp = gg - hh * (hh @ gg)
        d = (gp / (np.linalg.norm(gp) + 1e-12)).astype(np.float32)
        sd = saved.get(f"t{t}_D2")
        rec = {"utterance_id": uid, "t": t, "stratum": p["stratum"], "grads_none": grads_none,
               "site_equals_saved": bool(np.array_equal(box["site"].numpy(), saved[f"t{t}_hb"])),
               "J_independent": float(J.detach()), "J_saved": row["directions"]["D2"]["provenance"]["J"],
               "saved_status": row["directions"]["D2"]["status"]}
        rec["J_abs_diff"] = abs(rec["J_independent"] - rec["J_saved"])
        if sd is not None:
            rec["max_abs_diff"] = float(np.abs(d.astype(np.float64) - sd.astype(np.float64)).max())
            rec["cos"] = float(d.astype(np.float64) @ sd.astype(np.float64) /
                               (np.linalg.norm(d.astype(np.float64)) * np.linalg.norm(sd.astype(np.float64))))
        rec["ok"] = bool(grads_none and rec["site_equals_saved"] and rec["J_abs_diff"] <= 1e-5 and sd is not None
                         and rec["max_abs_diff"] <= 2e-6 and rec["cos"] >= 1 - 1e-6)
        res.append(rec)
        print(f"spot {uid} {t} ok={rec['ok']}", flush=True)
        del scratch, cache, o, enc
    return {"schema": "p2dir_spot_audit_v1", "n": len(res), "ok": all(r["ok"] for r in res) and len(res) == 30,
            "positions": res}


def cmd_exp1(args) -> dict:
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    run = ROOT / args.run
    man = json.loads((run / "manifest.json").read_text())
    ana = json.loads((ROOT / args.analysis).read_text())
    cfg = json.loads((ROOT / man["config"]).read_text())
    pos = json.loads((ROOT / man["positions"]).read_text())
    con = json.loads((ROOT / man["construction"]).read_text())
    checks, notes = {}, {}
    # provenance
    checks["manifest_self_hash"] = man["manifest_hash"] == canon({k: v for k, v in man.items() if k != "manifest_hash"})
    checks["config_hash"] = canon(cfg) == man["config_hash"]
    checks["sources_at_manifest_commit"] = all(git_blob_hash(man["git_commit"], p) == h for p, h in man["sources"].items())
    checks["positions_hash"] = pos["positions_hash"] == man["positions_hash"] == cfg["positions_hash"]
    checks["construction_hash"] = con["construction_hash"] == man["construction_hash"]
    checks["analysis_manifest"] = ana["manifest_hash"] == man["manifest_hash"]
    sealed = json.loads((run / "directions_sealed.json").read_text())
    checks["sealed_files"] = sealed["status"] == "SEALED" and all(fhash(run / r) == h for r, h in sealed["files"].items())
    # population
    checks["population_180"] = len(pos["positions"]) == 180 and \
        [(p["utterance_id"], p["t"]) for p in pos["positions"]] == [(p["utterance_id"], p["t"]) for p in con["positions"]]
    # folds: independent reconstruction from the extraction states
    ext = ROOT / man["extraction_run"]
    checks["extraction_states_hash"] = fhash(ext / "states.npz") == man["extraction_states_hash"]
    with np.load(ext / "states.npz") as z:
        HB, HE = z["H_B"], z["H_E"]
    folds = json.loads((ROOT / man["folds"]).read_text())
    checks["folds_hash"] = folds["folds_hash"] == man["folds_hash"] == canon({k: v for k, v in folds.items() if k != "folds_hash"})
    fold_res = {}
    for k in sorted({p["dialogue_id"] for p in con["positions"]}):
        ia = [i for i, p in sorted(enumerate(con["positions"]), key=lambda x: (x[1]["dialogue_id"], x[1]["utterance_id"], x[1]["t"]))
              if p["dialogue_id"] != k and p["stratum"] == "EN-correct"]
        ib = [i for i, p in sorted(enumerate(con["positions"]), key=lambda x: (x[1]["dialogue_id"], x[1]["utterance_id"], x[1]["t"]))
              if p["dialogue_id"] != k and p["stratum"] == "ZH-correct"]
        r = refit(HB[ia], HB[ib])
        f = folds["folds"][k]
        ok = r["status"] == f["status"]
        if ok and r["status"] == "ok":
            sv = np.load(ROOT / f["vector_path"], allow_pickle=False)
            ok = (r["index"] == f["selected_index"] and ahash(sv) == f["vector_sha256"]
                  and float(np.abs(sv.astype(np.float64) - r["vector"].astype(np.float64)).max()) <= 2e-6)
            fold_res[k] = {"index": r["index"], "sigma": r["sigma"], "max_abs_diff":
                           float(np.abs(sv.astype(np.float64) - r["vector"].astype(np.float64)).max())}
        else:
            fold_res[k] = {"status": r["status"], "reason": r.get("reason"), "sealed_reason": f.get("reason")}
        checks[f"fold_reconstructed:{k}"] = bool(ok)
    notes["folds"] = fold_res
    # tokenizer/suppression
    model = Path(man["model"]["dir"])
    part = tokenizer_partition(WhisperProcessor.from_pretrained(model, local_files_only=True).tokenizer)
    gen = GenerationConfig.from_pretrained(model, local_files_only=True)
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    checks["partition_hash"] = part["hash"] == man["partition_hash"]
    e_star = float(cfg["energy"]["e_star"])
    tol = cfg["energy"]["max_relative_squared_error"]
    cidx = {(p["utterance_id"], int(p["t"])): i for i, p in enumerate(con["positions"])}
    # per-position independent reconstruction
    recs, prov_fail, energy_fail = [], [], []
    rows_cache, uidx = {}, {u: i for i, u in enumerate(con["utterances"])}
    for p in pos["positions"]:
        uid, t = p["utterance_id"], int(p["t"])
        i = uidx[uid]
        if i not in rows_cache:
            row = json.loads((run / "rows" / f"{i:03d}.json").read_text())
            ev = json.loads((run / "rows" / f"{i:03d}_eval.json").read_text())
            with np.load(run / "rows" / f"{i:03d}.npz") as z:
                V = {k: z[k] for k in z.files}
            with np.load(run / "rows" / f"{i:03d}_logits.npz") as z:
                Lg = {k: bf16(z[k]) for k in z.files}
            with np.load(run / "rows" / f"{i:03d}_eval.npz") as z:
                EV = {k: z[k] for k in z.files}
            rows_cache = {i: (row, ev, V, Lg, EV)}
        row, ev, V, Lg, EV = rows_cache[i]
        rec = row["positions"][str(t)]
        pre = f"t{t}_"
        hb = V[pre + "hb"].astype(np.float64)
        # extraction identity
        ci_ = cidx[(uid, t)]
        same_state = np.array_equal(V[pre + "hb"], HB[ci_]) and np.array_equal(V[pre + "he"], HE[ci_])
        # D0 independent
        delta = V[pre + "he"].astype(np.float64) - hb
        dn = np.linalg.norm(delta)
        d0 = (delta / (dn + 1e-6)).astype(np.float32) if dn >= 1e-4 else None
        d0_ok = d0 is not None and pre + "D0" in V and \
            float(np.abs(d0.astype(np.float64) - V[pre + "D0"].astype(np.float64)).max()) <= 1e-6
        # D1 sealed fold of the position's dialogue
        f = folds["folds"][p["dialogue_id"]]
        d1_ok = (f["status"] != "ok" and pre + "D1" not in V) or (
            f["status"] == "ok" and pre + "D1" in V and ahash(V[pre + "D1"]) == f["vector_sha256"])
        # D2 from saved raw readout gradient
        d2_ok = True
        if pre + "g_readout" in V:
            g = V[pre + "g_readout"].astype(np.float64)
            hh = hb / np.linalg.norm(hb)
            gp = g - hh * (hh @ g)
            tn = np.linalg.norm(gp)
            if rec["directions"]["D2"]["status"] == "ok":
                d2 = (gp / (tn + 1e-12)).astype(np.float32)
                d2_ok = pre + "D2" in V and float(np.abs(d2.astype(np.float64) - V[pre + "D2"].astype(np.float64)).max()) <= 2e-6
            else:
                d2_ok = pre + "D2" not in V
            jm = metrics(Lg[pre + "none"], t, sup, beg, part, [0], 0)["J"]
            d2_ok = d2_ok and (rec["directions"]["D2"]["provenance"]["J"] is None or
                               abs(jm - rec["directions"]["D2"]["provenance"]["J"]) <= 1e-4)
        if not (same_state and d0_ok and d1_ok and d2_ok):
            prov_fail.append((uid, t, {"state": bool(same_state), "D0": bool(d0_ok), "D1": bool(d1_ok), "D2": bool(d2_ok)}))
        Y, c = [int(x) for x in p["target_ids"]], int(p["competitor"])
        mn = metrics(Lg[pre + "none"], t, sup, beg, part, Y, c)
        x = {"dialogue_id": p["dialogue_id"], "stratum": p["stratum"], "uid": uid, "t": t, "arms": {}}
        for a in ARMS:
            ar = rec["arms"][a]
            post = V.get(pre + "post_" + a)
            valid = rec["directions"][a]["status"] == "ok" and ar["solver"].get("status") == "ok" and ar["steered"] and post is not None
            if post is not None:
                # consumed (recorder) state vs hook-reported norm differ by bf16 rounding (P2-R: <= 1e-3 rel)
                en = float(np.linalg.norm(post.astype(np.float64) - hb))
                valid = valid and abs(ar["edit_norm"] ** 2 / e_star ** 2 - 1) <= tol and ar.get("solver_matches_hook", False)
                if abs(en / ar["edit_norm"] - 1) > 5e-3 or (valid and abs(en ** 2 / e_star ** 2 - 1) > tol):
                    energy_fail.append((uid, t, a, en, ar["edit_norm"]))
            ma = metrics(Lg[pre + a], t, sup, beg, part, Y, c)
            x["arms"][a] = {"valid": bool(valid), "dm": ma["m"] - mn["m"], "in_ref": ma["in_ref"], "e": ar["edit_norm"]}
        es = [x["arms"][a]["e"] ** 2 for a in ARMS if x["arms"][a]["valid"]]
        x["pair_ok"] = len(es) == 3 and max(es) / min(es) - 1 <= cfg["energy"]["max_pairwise_relative_squared_error"]
        recs.append(x)
    checks["direction_provenance"] = not prov_fail
    checks["realized_energy_from_states"] = not energy_fail
    notes["provenance_failures"] = prov_fail[:20]
    notes["energy_failures"] = energy_fail[:20]
    # state_ok taken from analysis per-position (V3 identity uses P2-R references); recompute matched independently
    ana_pp = {(r["utterance_id"], r["t"]): r for r in ana["per_position"]}
    for x in recs:
        x["state_ok"] = bool(ana_pp[(x["uid"], x["t"])].get("state_ok"))
        x["matched"] = x["state_ok"] and x["pair_ok"] and all(x["arms"][a]["valid"] for a in ARMS)
        checks.setdefault("matched_agrees", True)
        if x["matched"] != ana_pp[(x["uid"], x["t"])].get("matched"):
            checks["matched_agrees"] = False
    reps, seed = cfg["bootstrap"]["replicates"], cfg["bootstrap"]["seed"]
    keys = sorted({p["dialogue_id"] for p in pos["positions"]})
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(reps, len(keys)))
    W = np.stack([(idx == j).sum(axis=1) for j in range(len(keys))], axis=1).astype(np.float64)
    af = 0.05 / cfg["bootstrap"]["family_size_exp1"]
    M = [x for x in recs if x["matched"]]
    S = {s: [x for x in M if x["stratum"] == s] for s in STRATA}
    corr = [x for x in M if x["stratum"] != "EN-confusion"]
    mine = {}
    for cnd in ("D1", "D2"):
        mine[cnd] = {
            "conf": ci([(x["dialogue_id"], x["arms"][cnd]["dm"]) for x in S["EN-confusion"]], keys, W, af),
            "paired": ci([(x["dialogue_id"], x["arms"][cnd]["dm"] - x["arms"]["D0"]["dm"]) for x in S["EN-confusion"]], keys, W, af),
            "en": ci([(x["dialogue_id"], x["arms"][cnd]["dm"]) for x in S["EN-correct"]], keys, W, af),
            "zh": ci([(x["dialogue_id"], x["arms"][cnd]["dm"]) for x in S["ZH-correct"]], keys, W, af),
            "corr": ci([(x["dialogue_id"], 0.0 if x["arms"][cnd]["in_ref"] else 1.0) for x in corr], keys, W, af)}
        mine[cnd]["obs"] = {s: ci([(x["dialogue_id"], 0.0 if x["arms"][cnd]["in_ref"] else 1.0) for x in S[s]], keys, W, .05)[0]
                            for s in ("EN-correct", "ZH-correct")}
    tie = ci([(x["dialogue_id"], x["arms"]["D2"]["dm"] - x["arms"]["D1"]["dm"]) for x in S["EN-confusion"]], keys, W, .05)
    diffs = []
    for cnd in ("D1", "D2"):
        for k in ("conf", "paired", "en", "zh", "corr"):
            est, iv, nd = mine[cnd][k]
            a = ana["family"][cnd][k]
            diffs += [abs(est - a["estimate"]), abs(iv[0] - a["ci"][0]), abs(iv[1] - a["ci"][1])]
            checks.setdefault("valid_draws", True)
            checks["valid_draws"] &= nd >= 9900 and nd == a["valid_draws"]
    checks["statistics_agree_1e-8"] = max(diffs) <= 1e-8
    notes["max_stat_diff"] = max(diffs)
    # independent qualification / selection
    e1 = cfg["exp1"]
    N = len(recs)
    vr = {a: sum(1 for x in recs if x["state_ok"] and x["arms"][a]["valid"]) / N for a in ARMS}
    popok = all(len(S[s]) >= e1["matched_min_positions_per_stratum"] and
                len({x["dialogue_id"] for x in S[s]}) >= e1["matched_min_dialogues_per_stratum"] for s in STRATA)
    eng_ok = all(ana["engineering"].values()) and checks["direction_provenance"] and checks["realized_energy_from_states"]
    qual = {}
    for cnd in ("D1", "D2"):
        f = mine[cnd]
        benefit = f["conf"][0] >= e1["material_margin_nats"] and f["conf"][1][0] > 0 and f["paired"][1][0] > 0
        safety = (f["en"][1][0] >= e1["correct_margin_lower_bound_nats"] and f["zh"][1][0] >= e1["correct_margin_lower_bound_nats"]
                  and f["corr"][1][1] <= e1["corruption_upper_bound"] and all(v <= e1["corruption_upper_bound"] for v in f["obs"].values()))
        qual[cnd] = {"benefit": bool(benefit), "safety": bool(safety),
                     "qualifies": bool(benefit and safety and vr[cnd] >= e1["candidate_valid_rate_min"] and popok and eng_ok)}
    q = [c for c in ("D1", "D2") if qual[c]["qualifies"]]
    if not eng_ok:
        label, sel = "P2_DIR_INVALID", None
    elif not q:
        label, sel = "P2_DIR_NO_NEW_DIRECTION_SUPPORTED", None
    else:
        sel = q[0] if len(q) == 1 else ("D2" if tie[1][0] > 0.25 else "D1")
        label = "P2_DIR_UNIQUE_SELECTED" if sel == "D1" else "P2_DIR_READOUT_SELECTED"
    checks["qualification_agrees"] = all(qual[c]["qualifies"] == ana["qualification"][c]["qualifies"] for c in ("D1", "D2"))
    checks["label_agrees"] = label == ana["decision"]["label"] and sel == ana["decision"]["selected"]
    checks["valid_rate_agrees"] = all(abs(vr[a] - ana["valid_rate"][a]) <= 1e-12 for a in ARMS)
    if args.spot:
        spot = json.loads((ROOT / args.spot).read_text())
        checks["d2_gpu_spotcheck"] = bool(spot["ok"])
        notes["spot_n"] = spot["n"]
    else:
        checks["d2_gpu_spotcheck"] = False
    checks["only_three_directions"] = all(set(r["arms"]) == set(ARMS) for r in ana["per_position"] if r.get("arms"))
    verdict = "P2_DIR_AUDIT: PASS" if all(checks.values()) else "P2_DIR_AUDIT: BLOCK"
    return {"schema": "p2dir_exp1_audit_v1", "stage": "EXP1", "verdict": verdict, "label": label, "selected": sel,
            "checks": checks, "notes": notes, "independent_family": mine, "independent_tie": tie,
            "independent_qualification": qual, "valid_rate": vr}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prerun")
    p.add_argument("--out", required=True)
    s = sub.add_parser("spot")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    e = sub.add_parser("exp1")
    e.add_argument("--run", required=True)
    e.add_argument("--analysis", required=True)
    e.add_argument("--spot")
    e.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"prerun": cmd_prerun, "spot": cmd_spot, "exp1": cmd_exp1}[args.cmd](args)
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("audit output exists; never overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, sort_keys=True, indent=1, default=str) + "\n")
    print(json.dumps({k: res[k] for k in res if k in ("verdict", "label", "selected", "ok", "n")}, indent=1))
    if args.cmd == "prerun":
        print(json.dumps({k: v for k, v in res["checks"].items() if not v}, indent=1))


if __name__ == "__main__":
    main()
