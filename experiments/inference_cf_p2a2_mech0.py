#!/usr/bin/env python
"""P2-A2-MECH0 runner (episodic A2 adaptation dynamics and termination mechanism). Contract: docs/inference_cf/P2_A2_MECH0_SPEC.md,
P2_A2_MECH0_CODEX_DESIGN.md, P2_A2_MECH0_PANEL.json, configs/inference_cf/p2_a2_mech0.json (freeze 9d89640).

prepare   (CPU) verify config/panel/source hashes, FULL300 identity/order/audio/teacher/AUTO/PATH5 B0+A2 fingerprints; recompute the
          token/termination strata, CONTROL40 and MECH_PANEL independently and require exact panel agreement; write plan_sealed.json
          (runtime inputs separated from diagnostic metadata). No reference.
manifest  (CPU) resolved manifest after committed PASS_TO_P2_A2_MECH0.
run       (GPU) ONE job, one resident model. Phase 1 FULL300: theta0 decode == PATH5 B0; the UNCHANGED historical A2 episode
          (`inference_cf_p2tta0.run_objective`) observed by a pure global optimizer step post-hook that clones the detached fp32
          masters to CPU after each original AdamW step (no source change to the A2 code); final A2 == PATH5 A2 (tokens/termination/
          text), effective state hash == PATH5, FIXED100 fp32 arrays == historical archives; norms; LN snapshots. Barrier 300/300.
          Phase 2 MECH_PANEL84: STEP1 and DROP_SELF/CROSS/POST free decodes; changed44: STEP0/1/2 first-divergence geometry +
          H3 continuation on the fixed historical prefix and DROP current-query actions.
seal      (CPU) immutable output seal (before references).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, digest, file_hash

SCHEMA = "p2_a2_mech0_v1"
CONFIG = "configs/inference_cf/p2_a2_mech0.json"
PANEL = "docs/inference_cf/P2_A2_MECH0_PANEL.json"
SPEC = "docs/inference_cf/P2_A2_MECH0_SPEC.md"
DESIGN = "docs/inference_cf/P2_A2_MECH0_CODEX_DESIGN.md"
P5_PLAN = "results/inference_cf/p2path5/plan_sealed.json"
P5_PANEL = "docs/inference_cf/P2_PATH5_PANEL.json"
FULL = "results/inference_cf/p2_A_r1_L16/panel.json"
FIXED = "docs/inference_cf/P2_SEL_MINI_PANEL.json"
ROLE = "/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/manifests/roles/role_D-dev-select.parquet"
BASE = "results/inference_cf/p2a2_mech0"
PLAN = f"{BASE}/plan_sealed.json"
CB = [50258, 50260, 50360, 50364]
EOS = 50257
MAX_NEW = 200
REL_GRAD_TOL = 0.02
FAMILIES = ("SELF", "CROSS", "POST")
DROPS = ("DROP_SELF", "DROP_CROSS", "DROP_POST")
STRATA = ("A2_SAME", "A2_DELTA", "EOS_RECOVERY", "EOS_REGRESSION", "CONTENT_DIVERGENCE", "OTHER_INVALID", "AUTO_SAME", "AUTO_DELTA",
          "AUTO_SAME__A2_SAME", "AUTO_SAME__A2_DELTA", "AUTO_DELTA__A2_SAME", "AUTO_DELTA__A2_DELTA", "MECH_CHANGED", "CONTROL40", "MECH_PANEL",
          "FULL300", "FIXED100", "NEW200")


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def committed(rel: str) -> bool:
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return git("ls-files", rel) == rel and hashlib.sha256(blob).hexdigest() == sha_file(ROOT / rel)


def row_fp(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def membership_hash(name: str, ids, dlg: dict) -> str:
    return row_fp({"partition": name, "rows": [{"utterance_id": u, "dialogue_id": dlg[u]} for u in ids]})


def stream(tokens, term):
    return [int(t) for t in tokens] + ([EOS] if term == "eos" else [])


# ---- pure stratum / state helpers (unit-tested) -------------------------------------------------------------------

def divergence(b0: dict, a2: dict) -> dict | None:
    """First decision-stream divergence of canonical B0 vs final A2 (logical EOS iff terminated=eos). None if identical."""
    sb, sa = stream(b0["tokens"], b0["terminated"]), stream(a2["tokens"], a2["terminated"])
    if sb == sa:
        return None
    k = next((i for i, (x, y) in enumerate(zip(sb, sa)) if x != y), None)
    if k is None:                                                   # strict prefix without EOS (cap) -> unavailable action
        return {"k": min(len(sb), len(sa)), "kind": "OTHER_INVALID"}
    c0, c2 = sb[k], sa[k]
    kind = ("EOS_RECOVERY" if c0 == EOS and c2 != EOS else "EOS_REGRESSION" if c2 == EOS and c0 != EOS else "CONTENT_DIVERGENCE")
    return {"k": k, "B0_action": c0, "A2_action": c2, "common_prefix_sha256": row_fp(list(b0["tokens"][:k])), "kind": kind,
            "H3_A2_content": [int(t) for t in a2["tokens"][k:k + 3]], "EOS_appended_only_for_comparison": True}


def strata(ids: list, dlg: dict, B0: dict, A2: dict, AUTO: dict, fixed: list) -> tuple[dict, dict]:
    """Reference-free strata from token arrays only, FULL300 canonical order."""
    div = {u: divergence(B0[u], A2[u]) for u in ids}
    auto_same = {u: stream(AUTO[u]["tokens"], AUTO[u]["terminated"]) == stream(B0[u]["tokens"], B0[u]["terminated"]) for u in ids}
    P = {k: [] for k in STRATA}
    for u in ids:
        d = div[u]
        a2k = "A2_SAME" if d is None else "A2_DELTA"
        P[a2k].append(u)
        if d is not None:
            P[d["kind"]].append(u)
        ak = "AUTO_SAME" if auto_same[u] else "AUTO_DELTA"
        P[ak].append(u)
        P[f"{ak}__{a2k}"].append(u)
    P["MECH_CHANGED"] = list(P["A2_DELTA"])
    per_d = {}
    for u in P["A2_SAME"]:
        per_d.setdefault(dlg[u], []).append(u)
    if sorted(per_d) != sorted(set(dlg[u] for u in ids)) or any(len(v) < 2 for v in per_d.values()):
        raise ValueError("CONTROL40 shortage: every dialogue needs >=2 A2_SAME rows (no refill)")
    ctrl = {u for v in per_d.values() for u in v[:2]}
    P["CONTROL40"] = [u for u in ids if u in ctrl]
    mp = set(P["MECH_CHANGED"]) | ctrl
    P["MECH_PANEL"] = [u for u in ids if u in mp]
    P["FULL300"] = list(ids)
    P["FIXED100"] = list(fixed)
    P["NEW200"] = [u for u in ids if u not in set(fixed)]
    return P, div


def family_partition(names: list, families: dict) -> dict:
    """Exact SELF/CROSS/POST partition of the 194 A2 trainables (union = all, pairwise disjoint)."""
    sets = {f: set(families[f]) for f in FAMILIES}
    allf = set().union(*sets.values())
    if allf != set(names) or sum(len(s) for s in sets.values()) != len(names) or any(sets[a] & sets[b] for a in FAMILIES for b in FAMILIES if a < b):
        raise ValueError("family partition is not exact")
    own = {"SELF": [n for n in names if ".self_attn_layer_norm." in n], "CROSS": [n for n in names if ".encoder_attn_layer_norm." in n],
           "POST": [n for n in names if ".final_layer_norm." in n or n.startswith("model.decoder.layer_norm.")]}
    if any(set(own[f]) != sets[f] for f in FAMILIES):
        raise ValueError("family membership does not follow module names")
    return {n: f for f in FAMILIES for n in families[f]}


def module_of(name: str) -> str:
    return name.rsplit(".", 1)[0]


def layer_of(name: str) -> str:
    parts = name.split(".")
    return f"layer{int(parts[3]):02d}" if parts[2] == "layers" else "decoder_final"


def delta_norms(theta0_f32: dict, state_f32: dict, names: list, fam: dict, *, effective: bool, theta0_bf16: dict | None = None) -> dict:
    """L2 norms of (state - theta0) per tensor/module/layer/family/total (float64 squared aggregation)."""
    import torch
    sq = {}
    for n in names:
        if effective:
            d = state_f32[n].to(torch.bfloat16).float() - theta0_bf16[n].float()
        else:
            d = state_f32[n].float() - theta0_f32[n]
        sq[n] = float((d.double() ** 2).sum())
    agg = lambda key: {k: math.sqrt(v) for k, v in _group(sq, key).items()}
    tot = math.sqrt(sum(sq.values()))
    base = math.sqrt(sum(float((theta0_f32[n].double() ** 2).sum()) for n in names))
    return {"total": tot, "relative": tot / (base + 1e-12), "family": agg(lambda n: fam[n]), "layer": agg(layer_of), "module": agg(module_of),
            "tensor": {n: math.sqrt(v) for n, v in sq.items()}}


def _group(sq: dict, key) -> dict:
    out = {}
    for n, v in sq.items():
        out[key(n)] = out.get(key(n), 0.0) + v
    return out


def drop_family(final_bf16: dict, theta0_bf16: dict, family_names) -> dict:
    """Exact final A2 bf16 state with one family replaced by theta0 bf16 bytes; all other tensors untouched."""
    fs = set(family_names)
    return {n: (theta0_bf16[n] if n in fs else final_bf16[n]) for n in final_bf16}


def action_class(a: int, b0_action: int, a2_action: int) -> str:
    return "A2" if a == a2_action else ("B0" if a == b0_action else "third")


@contextlib.contextmanager
def step_observer():
    """Pure observer: after every optimizer.step() clone the detached fp32 parameters (masters) to CPU. No RNG, forward,
    gradient, loss or optimizer mutation; removed on exit."""
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook
    snaps = []

    def hook(opt, args, kwargs):
        with torch.no_grad():
            snaps.append([p.detach().to("cpu", copy=True) for g in opt.param_groups for p in g["params"]])
    h = register_optimizer_step_post_hook(hook)
    try:
        yield snaps
    finally:
        h.remove()


def cmd_prepare(args) -> None:
    import experiments.inference_cf_p2tta0 as t0run
    import pyarrow.parquet as pq
    from transformers import GenerationConfig, WhisperProcessor
    import experiments.inference_cf_p2dir_prepare as prep
    from csasr.inference_cf.core_r2 import tokenizer_partition
    cfg = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    checks = {"panel_bytes": sha_file(ROOT / PANEL) == cfg["panel_byte_sha256"],
              "frozen_committed": all(committed(p) for p in (CONFIG, PANEL, SPEC, DESIGN)),
              "sources": all(sha_file(ROOT / p) == h for p, h in cfg["source_sha256"].items()),
              "panel_sources": all(sha_file(ROOT / p) == h for p, h in P["source_sha256"].items()),
              "instrumentation_baselines_unchanged": all(sha_file(ROOT / p) == h for p, h in cfg["instrumentation_provenance"]["original_source_baseline_sha256"].items()),
              "full300_parent": sha_file(ROOT / FULL) == cfg["FULL300_parent_byte_sha256"] == P["FULL300_source_parent"]["byte_sha256"]}
    plan5 = json.loads((ROOT / P5_PLAN).read_text())
    ids = plan5["ids"]
    fullp = json.loads((ROOT / FULL).read_text())
    fixed = [r["utterance_id"] for r in json.loads((ROOT / FIXED).read_text())["rows"]]
    role = pq.read_table(ROLE, columns=["utterance_id", "dialogue_id", "role"]).to_pylist()      # ID/dialogue/role columns ONLY
    rm = {}
    for x in role:
        if x["utterance_id"] in set(ids):
            rm.setdefault(x["utterance_id"], []).append((x["dialogue_id"], x["role"]))
    checks["role_manifest"] = set(rm) == set(ids) and all(len(v) == 1 and v[0][1] == "D-dev-select" for v in rm.values())
    dlg = {u: rm[u][0][0] for u in ids}
    checks["identity_order"] = (ids == [r["utterance_id"] for r in fullp["rows"]] == P["canonical_order"] == [r["utterance_id"] for r in P["rows"]]
                                and len(set(ids)) == 300 and len(set(dlg.values())) == 20 and [r["canonical_index"] for r in P["rows"]] == list(range(300))
                                and all(r["dialogue_id"] == dlg[r["utterance_id"]] for r in P["rows"]))
    B0, A2, AUTO, rows, bad = {}, {}, {}, [], []
    for i, (r, x5) in enumerate(zip(P["rows"], plan5["rows"])):
        u = r["utterance_id"]
        p5 = json.loads((ROOT / r["PATH5_output"]["path"]).read_text())
        ar = json.loads((ROOT / r["B0_AUTO"]["path"]).read_text())
        B0[u], A2[u], AUTO[u] = p5["theta0"], p5["A2_free"], ar["systems"]["B0_AUTO"]
        s = x5["runtime"]
        fine = (p5["identity"] == u and p5["status"] == "ok" and sha_file(ROOT / r["PATH5_output"]["path"]) == r["PATH5_output"]["byte_sha256"]
                and sha_file(ROOT / r["B0_AUTO"]["path"]) == r["B0_AUTO"]["byte_sha256"] and ar["identity"] == u
                and row_fp(AUTO[u]["tokens"]) == r["B0_AUTO"]["tokens_sha256"] and AUTO[u]["terminated"] == r["B0_AUTO"]["terminated"]
                and row_fp(s) == r["teacher"]["plan_runtime_row_sha256"] and row_fp(s["y_A"]) == r["teacher"]["y_A_token_sha256"]
                and row_fp(s["y_A_valid_mask"]) == r["teacher"]["valid_mask_sha256"] and s["utterance_id"] == u)
        ad = r["audio"]
        fine &= (s["audio_path"] == ad["path"] and s["audio_fingerprint_64k_sizeprefixed"] == ad["fingerprint_64k_sizeprefixed"] == t0run.audio_fingerprint_64k(ad["path"])
                 and t0run.audio_full_sha256(ad["path"]) == ad["full_sha256"] == s["audio_full_sha256"])
        bs, as_ = r["B0_sealed"], r["A2_sealed"]
        fine &= (row_fp(B0[u]["tokens"]) == bs["token_sha256"] and B0[u]["terminated"] == bs["termination"] and sha_text(B0[u]["text"]) == bs["text_sha256"]
                 and row_fp(A2[u]["tokens"]) == as_["token_sha256"] and A2[u]["terminated"] == as_["termination"] and sha_text(A2[u]["text"]) == as_["text_sha256"]
                 and p5["A2_state_hash"] == as_["effective_state_sha256"])
        arch = r["historical_final_master_archive"]
        if arch is not None:
            fine &= sha_file(ROOT / arch["path"]) == arch["byte_sha256"]
        if not fine:
            bad.append(u)
    parts, div = strata(ids, dlg, B0, A2, AUTO, fixed)
    pr = {r["utterance_id"]: r for r in P["rows"]}
    checks["strata_exact"] = (all(parts[k] == P["partitions"][k]["ids"] for k in STRATA) and {k: len(parts[k]) for k in STRATA} == cfg["counts"]
                              and all(membership_hash(k, parts[k], dlg) == cfg["partition_sha256"][k] == P["partitions"][k]["membership_order_sha256"] for k in STRATA)
                              and all(pr[u]["stratum"] == ("A2_SAME" if div[u] is None else "A2_DELTA") for u in ids)
                              and all(pr[u]["teacher_relation"] == ("AUTO_SAME" if u in set(parts["AUTO_SAME"]) else "AUTO_DELTA") for u in ids)
                              and all(pr[u]["divergence"] == div[u] for u in ids)
                              and len(parts["OTHER_INVALID"]) == 0 and len({dlg[u] for u in parts["CONTROL40"]}) == 20)
    names = [p["name"] for p in cfg["A2_inherited"]["trainables"]["parameters"]]
    fam = family_partition(names, cfg["families"])
    checks["family_partition"] = ({f: sum(1 for n in names if fam[n] == f) for f in FAMILIES} == cfg["family_tensor_counts"]
                                  and {f: sum(p["numel"] for p in cfg["A2_inherited"]["trainables"]["parameters"] if fam[p["name"]] == f) for f in FAMILIES}
                                  == cfg["family_scalar_counts"] and sum(cfg["family_scalar_counts"].values()) == cfg["A2_inherited"]["trainables"]["scalar_count"] == 248320)
    gen = GenerationConfig.from_pretrained(prep.MODEL, local_files_only=True)
    checks["suppression"] = {"suppress": list(gen.suppress_tokens or []), "begin": list(gen.begin_suppress_tokens or [])} == cfg["suppression"] == plan5["suppression"]
    tok = WhisperProcessor.from_pretrained(prep.MODEL, local_files_only=True).tokenizer
    mp = set(parts["MECH_PANEL"])
    for i, (u, x5) in enumerate(zip(ids, plan5["rows"])):
        r = pr[u]
        rows.append({"runtime": dict(x5["runtime"]),
                     "audit": {"canonical_index": i, "dialogue_id": dlg[u], "PATH5_partition": r["PATH5_partition"], "stratum": r["stratum"],
                               "kind": div[u]["kind"] if div[u] else "A2_SAME", "teacher_relation": r["teacher_relation"], "mech_panel": u in mp,
                               "control": u in set(parts["CONTROL40"]),
                               "B0": {k: B0[u][k] for k in ("tokens", "terminated", "text")}, "A2": {k: A2[u][k] for k in ("tokens", "terminated", "text")},
                               "A2_state_hash": r["A2_sealed"]["effective_state_sha256"],
                               "archive": r["historical_final_master_archive"]["path"] if r["historical_final_master_archive"] else None,
                               "divergence": div[u], "AUTO_stream": stream(AUTO[u]["tokens"], AUTO[u]["terminated"])}})
    checks["rows"] = not bad
    plan = {"schema": SCHEMA + "_plan", "panel_sha256": sha_file(ROOT / PANEL), "config_sha256": sha_file(ROOT / CONFIG), "checks": checks,
            "failures": bad, "ids": ids, "partitions": parts, "families": {f: cfg["families"][f] for f in FAMILIES}, "suppression": cfg["suppression"],
            "eos": EOS, "partition_hash": tokenizer_partition(tok)["hash"], "rows": rows, "outcomes_computed": False, "references_used": False,
            "created_unix": time.time()}
    plan["plan_hash"] = digest(plan)
    out = ROOT / PLAN
    if out.exists():
        raise FileExistsError("plan exists; never overwrite")
    if not all(checks.values()):
        raise SystemExit("BLOCK: " + json.dumps([k for k, v in checks.items() if not v]) + " rows " + json.dumps(bad[:10]))
    atomic_json(out, plan)
    print(json.dumps({"plan_hash": plan["plan_hash"], "checks": checks, "counts": {k: len(v) for k, v in parts.items()}}))


SOURCES = (SPEC, DESIGN, PANEL, CONFIG, PLAN, P5_PLAN, P5_PANEL, FULL, FIXED, "src/csasr/inference_cf/episodic_tta.py", "src/csasr/inference_cf/path_decode.py",
           "src/csasr/inference_cf/branch_adjudication.py", "experiments/inference_cf_p2tta0.py", "experiments/inference_cf_p2tta0_audit.py",
           "experiments/inference_cf_cached.py", "src/csasr/inference_cf/core_p1.py", "src/csasr/inference_cf/core_r2.py", "src/csasr/lss/sites.py",
           "src/csasr/models/whisper.py", "experiments/inference_cf_p2a2_mech0.py", "experiments/inference_cf_p2a2_mech0_analyze.py",
           "experiments/inference_cf_p2a2_mech0_audit.py", "slurm/inference_cf_p2a2_mech0.sbatch", "tests/test_inference_cf_p2a2_mech0.py")


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    pre = f"{BASE}/prerun_audit.json"
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, pre)
    if dirty:
        raise ValueError("commit MECH0 sources before preparing a manifest:\n" + dirty)
    if json.loads((ROOT / pre).read_text())["verdict"] != "PASS_TO_P2_A2_MECH0":
        raise ValueError("pre-run audit did not pass")
    plan = json.loads((ROOT / PLAN).read_text())
    cfg = json.loads((ROOT / CONFIG).read_text())
    man = {"schema": SCHEMA, "stage": "run1", "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "config_sha256": sha_file(ROOT / CONFIG), "panel_sha256": sha_file(ROOT / PANEL), "plan_hash": plan["plan_hash"], "ids": plan["ids"],
           "strata_sha256": cfg["partition_sha256"], "MECH_PANEL_sha256": cfg["partition_sha256"]["MECH_PANEL"],
           "trainables": [p["name"] for p in cfg["A2_inherited"]["trainables"]["parameters"]], "budget": cfg["compute"],
           "A2_provenance": {"teacher": P5_PLAN, "B0_A2_comparator": "PATH5 run1 rows theta0/A2_free", "AUTO": "P2-A r1 systems.B0_AUTO",
                             "instrumentation": "global torch optimizer step post-hook (no source change to episodic_tta.py / inference_cf_p2tta0.py)",
                             "instrumented_files_runtime_sha256": {p: sha_file(ROOT / p) for p in cfg["instrumentation_provenance"]["allowed_files"]}},
           "reference_barrier": cfg["reference_barrier"], "output_root": f"{args.out}",
           "environment": prep.environment(), "model": {"dir": str(prep.MODEL), "files": prep.model_hashes()},
           "sources": {p: file_hash(ROOT / p) for p in SOURCES + (pre,)}, "references_used": False, "created_unix": time.time()}
    man["manifest_hash"] = digest(man)
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("manifest exists; never overwrite")
    atomic_json(out / "manifest.json", man)
    print(json.dumps({"manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}))


def geometry_path(bundle, guard, enc, prefix: list, k: int, cands: dict, cont: list | None, expected_hash: str, sup, beg) -> dict:
    """Teacher-force the fixed historical prompt+common prefix under the resident state (fresh owned cache); current-query
    processed float32 log probs at k; optional teacher-forced H3 continuation score. STEP1 prefix need not be greedy."""
    import torch
    import experiments.inference_cf_cached as cached
    from csasr.inference_cf.branch_adjudication import processed_log_probs
    from csasr.inference_cf.core_p1 import processed_argmax
    from csasr.lss.sites import assert_no_site_hooks
    if guard.current_hash() != expected_hash:
        raise ValueError("resident state != declared owner before geometry")
    assert_no_site_hooks(bundle)
    br = cached.Branch(bundle, enc, list(CB), "MECH0")
    new, mism = list(CB), []
    with torch.inference_mode():
        for t in range(k + 1):
            logits, _, _ = br.step(new, capture_layer=16, attention=True)
            if not bool(torch.isfinite(logits.float()).all()):
                raise ValueError("nonfinite raw logits")
            if t < k:
                g = processed_argmax(logits, t, sup, beg)
                if g != int(prefix[t]):
                    mism.append(t)
                new = [int(prefix[t])]
        lp = processed_log_probs(logits, k, sup, beg)
        fin = torch.isfinite(lp)
        p = torch.exp(lp[fin].double())
        ent = float(-(p * lp[fin].double()).sum())
        top = torch.topk(lp, 2).values
        res = {"k": k, "argmax": int(torch.argmax(lp)), "entropy": ent, "top1_top2_gap": float(top[0] - top[1]), "prefix_greedy_mismatch_positions": mism,
               "cands": {}}
        for name, c in cands.items():
            if c is None or not (0 <= int(c) < lp.numel()) or not bool(torch.isfinite(lp[int(c)])):
                res["cands"][name] = {"token": c, "logp": None, "allowed": False}
            else:
                res["cands"][name] = {"token": int(c), "logp": float(lp[int(c)]), "allowed": True}
        if cont is not None:
            scores = []
            for j, tkn in enumerate(cont):
                lpj = lp if j == 0 else processed_log_probs(logits, k + j, sup, beg)
                v = float(lpj[int(tkn)])
                if not math.isfinite(v):
                    raise ValueError("continuation token not generation-valid")
                scores.append(v)
                if j + 1 < len(cont):
                    logits, _, _ = br.step([int(tkn)], capture_layer=16, attention=True)
            res["continuation"] = {"tokens": list(cont), "H_eff": len(cont), "logprobs": scores, "C_H": (sum(scores) / len(scores)) if scores else None}
    fed_expected = list(CB) + [int(t) for t in prefix[:k]] + ([int(t) for t in cont[:-1]] if cont else [])
    res["fed_ok"] = br.fed == fed_expected and br.positions == list(range(len(br.fed)))
    del br
    res["state_locked"] = guard.current_hash() == expected_hash
    return res


def cmd_run(args) -> None:
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.inference_cf.episodic_tta import LNGuard, TTAInvalid, decoder_ln_names, forced_decode, special_ids, tensor_bytes_hash, valid_mask
    from csasr.lss.sites import assert_no_site_hooks
    from csasr.inference_cf.path_decode import edit_distance
    from csasr.models.whisper import batch_model_inputs, load_whisper
    from csasr.utils.config import load_config
    import experiments.inference_cf_p2tta0 as t0run
    import experiments.inference_cf_p2tta0_audit as live_audit
    out = ROOT / args.out
    m = json.loads((out / "manifest.json").read_text())
    if m["schema"] != SCHEMA or digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("invalid manifest")
    for p, h in m["sources"].items():
        if file_hash(ROOT / p) != h:
            raise ValueError(f"source changed: {p}")
    if git("rev-parse", "HEAD") != m["git_commit"]:
        raise ValueError("git commit changed")
    plan = json.loads((ROOT / PLAN).read_text())
    if plan["plan_hash"] != m["plan_hash"]:
        raise ValueError("plan")
    t_setup = time.time()
    torch.manual_seed(240924)
    b = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if b.device != "cuda" or b.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    b.model.eval()
    b.model.requires_grad_(False)
    names = decoder_ln_names(b.model)
    if names != m["trainables"] or len(names) != 194:
        raise TTAInvalid("trainable enumeration")
    fam = family_partition(names, plan["families"])
    g = LNGuard(b.model, names)
    names_set = set(names)
    nonln0 = tensor_bytes_hash([p for n, p in b.model.named_parameters() if n not in names_set])
    tok = b.processor.tokenizer
    eos = tok.eos_token_id
    gen = b.model.generation_config
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    if {"suppress": sup, "begin": beg} != plan["suppression"] or eos != plan["eos"]:
        raise TTAInvalid("suppression/eos")
    partition = tokenizer_partition(tok)
    if partition["hash"] != plan["partition_hash"]:
        raise TTAInvalid("partition")
    specials = special_ids(tok)
    th0_bf16 = {n: g.theta0[n].detach().cpu().clone() for n in names}
    th0_f32 = {n: th0_bf16[n].float() for n in names}
    (out / "rows").mkdir(parents=True, exist_ok=True)
    (out / "states").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "states" / "theta0_ln_fp32.npz", **{f"p{j:03d}": th0_f32[n].numpy() for j, n in enumerate(names)})
    counters = {"A2": {}, "encoder_passes": 0, "theta0_decodes": 0, "A2_final_decodes": 0, "step1_decodes": 0, "drop_decodes": 0, "geometry_paths": 0,
                "drop_query_paths": 0, "observer_snapshots": 0, "audit": {"forwards": 0, "backwards": 0}}
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "start_unix": time.time(),
               "setup_sec": time.time() - t_setup, "status": "running", "theta0_ln_hash": g.theta0_hash, "nonln_hash_start": nonln0,
               "phase1_sec": 0.0, "phase2_sec": 0.0, "barrier": None}
    atomic_json(out / "runtime.json", runtime)
    invalid, rows_out, keep = [], [], {}
    to_bf16 = lambda st: {n: st[n].to(torch.bfloat16) for n in names}

    def state_hash(st_bf16):
        try:
            g.materialize({n: v.to(g.theta0[n].device) for n, v in st_bf16.items()})
            return g.current_hash()
        finally:
            g.restore()

    def decode_state(st_bf16, enc):
        try:
            g.materialize({n: v.to(g.theta0[n].device) for n, v in st_bf16.items()})
            h = g.current_hash()
            d = forced_decode(b, enc, CB, max_new_tokens=MAX_NEW)
            if g.current_hash() != h:
                raise TTAInvalid("state changed during decode")
        finally:
            g.restore()
        if not (g.verify() and g.current_hash() == g.theta0_hash):
            raise TTAInvalid("reset after diagnostic decode")
        return {"tokens": d["tokens"], "terminated": d["terminated"], "text": d["text"], "state_hash": h}

    try:
        torch.cuda.reset_peak_memory_stats()
        # ================= Phase 1: FULL300 exact A2 reconstruction barrier + dynamics =================
        t1 = time.time()
        for i, x in enumerate(plan["rows"]):
            s, au = x["runtime"], x["audit"]
            uid = s["utterance_id"]
            row = {"identity": uid, "canonical_index": i, "dialogue_id": au["dialogue_id"], "manifest_hash": m["manifest_hash"]}
            if t0run.audio_fingerprint_64k(s["audio_path"]) != s["audio_fingerprint_64k_sizeprefixed"] or \
                    t0run.audio_full_sha256(s["audio_path"]) != s["audio_full_sha256"]:
                raise TTAInvalid("audio bytes changed")
            inputs = batch_model_inputs(b, [s["audio_path"]])
            with torch.inference_mode():
                h_inf = b.model.model.encoder(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state
            counters["encoder_passes"] += 1
            enc_inf = BaseModelOutput(last_hidden_state=h_inf)
            h_train = h_inf.clone()
            if h_train.is_inference() or h_train.requires_grad or not torch.equal(h_train, h_inf):
                raise TTAInvalid("encoder clone")
            enc_train = BaseModelOutput(last_hidden_state=h_train)
            if not g.verify():
                raise TTAInvalid("theta0 not resident")
            d0 = forced_decode(b, enc_inf, CB, max_new_tokens=MAX_NEW)
            counters["theta0_decodes"] += 1
            row["theta0"] = {"tokens": d0["tokens"], "terminated": d0["terminated"], "text": d0["text"]}
            row["theta0_equals_B0"] = row["theta0"] == au["B0"]
            la = None
            if i == 0:
                la = live_audit.live_objective_check(b.model, g.names, enc_train, CB, s["y_A"], "A2", suppress=sup, begin=beg, tokenizer=tok)
                counters["audit"]["forwards"] += 1
                counters["audit"]["backwards"] += 1
                if not g.verify():
                    raise TTAInvalid("live audit changed resident parameters")
            assert_no_site_hooks(b)
            with step_observer() as snaps:
                r = t0run.run_objective(b, g, "A2", enc_train, enc_inf, s["y_A"], s["y_A_valid_mask"], d0["tokens"],
                                        valid_mask(d0["tokens"], sup, beg, specials, eos), suppress=sup, begin=beg, eos=eos, partition=partition,
                                        counters=counters["A2"], keep_grad0=(i == 0))
            assert_no_site_hooks(b)
            counters["A2_final_decodes"] += 1
            counters["observer_snapshots"] += len(snaps)
            if len(snaps) != 2 or any(len(sn) != 194 for sn in snaps):
                raise TTAInvalid("observer did not capture exactly two 194-tensor snapshots")
            S1 = {n: snaps[0][j] for j, n in enumerate(names)}
            S2 = {n: snaps[1][j] for j, n in enumerate(names)}
            fm = r["final_masters"]
            row["step2_snapshot_equals_final_masters"] = all(torch.equal(S2[n], fm[n]) for n in names)
            lg = r["log"]
            row["A2_log"] = {k: lg.get(k) for k in ("losses", "grad_l2", "finite", "valid", "content", "no_valid_content", "steps", "loss_evaluations",
                                                    "master_delta_l2", "master_delta_rel", "effective_delta_l2", "effective_changed_scalars",
                                                    "theta0_ln_l2", "start_hash", "end_hash", "reset_ok", "other_versions_unchanged")}
            dec = r["decode"]
            row["A2_final"] = {"tokens": dec["tokens"], "terminated": dec["terminated"], "text": dec["text"]}
            row["A2_equals_sealed"] = row["A2_final"] == au["A2"]
            S1b, S2b = to_bf16(S1), to_bf16(S2)
            row["step1_state_hash"] = state_hash(S1b)
            row["step2_state_hash"] = state_hash(S2b)
            row["A2_state_hash_equals_sealed"] = row["step2_state_hash"] == au["A2_state_hash"]
            row["step1_differs_from_step2"] = any(not torch.equal(S1[n], S2[n]) for n in names)
            if au["archive"] is not None:
                with np.load(ROOT / au["archive"]) as z:
                    row["fp32_equals_archive"] = all(z[f"A2_p{j:03d}"].dtype == np.float32 and np.array_equal(S2[n].numpy(), z[f"A2_p{j:03d}"])
                                                     for j, n in enumerate(names))
            else:
                row["fp32_equals_archive"] = None
            row["norms"] = {f"step{st}_{kind}": delta_norms(th0_f32, S, names, fam, effective=(kind == "effective"), theta0_bf16=th0_bf16)
                            for st, S in ((1, S1), (2, S2)) for kind in ("master", "effective")}
            row["norms"]["step1_to_step2_master_l2"] = math.sqrt(sum(float(((S2[n] - S1[n]).double() ** 2).sum()) for n in names))
            np.savez_compressed(out / "states" / f"{i:03d}_step1_masters_fp32.npz", **{f"A2_p{j:03d}": S1[n].numpy() for j, n in enumerate(names)})
            row["step1_archive_sha256"] = sha_file(out / "states" / f"{i:03d}_step1_masters_fp32.npz")
            if au["archive"] is None:
                np.savez_compressed(out / "states" / f"{i:03d}_step2_masters_fp32.npz", **{f"A2_p{j:03d}": S2[n].numpy() for j, n in enumerate(names)})
                row["step2_archive_sha256"] = sha_file(out / "states" / f"{i:03d}_step2_masters_fp32.npz")
            row["reset_ok"] = g.verify() and g.current_hash() == g.theta0_hash and bool(lg.get("reset_ok"))
            if la is not None:
                g_aud, g_pri = la.pop("_grad").cpu(), r["grad0"]
                la["primary_loss"] = lg["losses"][0]
                la["loss_abs_diff"] = abs(la["primary_loss"] - la["auditor_loss"])
                la["grad_diff_l2"] = float(torch.linalg.vector_norm(g_pri.double() - g_aud.double()))
                la["auditor_grad_l2"] = float(torch.linalg.vector_norm(g_aud.double()))
                la["pass"] = la["loss_abs_diff"] <= 1e-5 and la["grad_diff_l2"] <= max(1e-8, REL_GRAD_TOL * la["auditor_grad_l2"])
                row["live_audit"] = la
                if not la["pass"]:
                    invalid.append(f"{uid}: live audit")
            for k in ("theta0_equals_B0", "A2_equals_sealed", "A2_state_hash_equals_sealed", "step2_snapshot_equals_final_masters", "reset_ok"):
                if not row[k]:
                    invalid.append(f"{uid}: {k}")
            if row["fp32_equals_archive"] is False:
                invalid.append(f"{uid}: fp32 archive")
            if au["mech_panel"]:
                keep[uid] = (h_inf.detach().cpu().clone(), S1b, S2b)
            rows_out.append(row)
            del r, fm, S1, S2, snaps
            print(f"MECH0 phase1 {i + 1}/300 {uid} B0 {row['theta0_equals_B0']} A2 {row['A2_equals_sealed']} hash {row['A2_state_hash_equals_sealed']} "
                  f"arch {row['fp32_equals_archive']}", flush=True)
        runtime["phase1_sec"] = time.time() - t1
        runtime["barrier"] = {"rows": 300, "pass": not invalid, "failures": list(invalid)}
        atomic_json(out / "runtime.json", runtime)
        if invalid:
            raise TTAInvalid("FULL300 A2 reconstruction barrier failed (no STEP1 decode or ablation run): " + "; ".join(invalid))
        # ================= Phase 2: MECH_PANEL84 step1 / DROP decodes; changed44 geometry =================
        t2 = time.time()
        for i, x in enumerate(plan["rows"]):
            au = x["audit"]
            if not au["mech_panel"]:
                continue
            uid = x["runtime"]["utterance_id"]
            row = rows_out[i]
            h_cpu, S1b, S2b = keep.pop(uid)
            enc = BaseModelOutput(last_hidden_state=h_cpu.to(g.theta0[names[0]].device))
            states = {"A2_STEP1": S1b}
            for f, dname in zip(FAMILIES, DROPS):
                st = drop_family(S2b, th0_bf16, plan["families"][f])
                if any(not torch.equal(st[n], th0_bf16[n]) for n in plan["families"][f]) or \
                        any(not torch.equal(st[n], S2b[n]) for n in names if n not in set(plan["families"][f])):
                    raise TTAInvalid("drop-family construction")
                states[dname] = st
            row["diag_decodes"] = {}
            A2s = stream(row["A2_final"]["tokens"], row["A2_final"]["terminated"])
            B0s = stream(row["theta0"]["tokens"], row["theta0"]["terminated"])
            for cname, st in states.items():
                d = decode_state(st, enc)
                counters["step1_decodes" if cname == "A2_STEP1" else "drop_decodes"] += 1
                ds = stream(d["tokens"], d["terminated"])
                d.update(ED_to_A2=edit_distance(ds, A2s), ED_to_B0=edit_distance(ds, B0s), equals_A2=ds == A2s, equals_B0=ds == B0s)
                row["diag_decodes"][cname] = d
            row["drop_state_hashes"] = {c: row["diag_decodes"][c]["state_hash"] for c in DROPS}
            dv = au["divergence"]
            if dv is not None:
                k = dv["k"]
                prefix = au["B0"]["tokens"][:k]
                auto_s = au["AUTO_stream"]
                auto_a = auto_s[k] if k < len(auto_s) else None
                cands = {"EOS": eos, "B0": dv["B0_action"], "A2": dv["A2_action"], "AUTO": auto_a}
                cont = list(dv["H3_A2_content"]) if dv["A2_action"] != eos else []
                geo = {}
                for sname, st in (("STEP0", None), ("STEP1", S1b), ("STEP2", S2b)):
                    try:
                        if st is not None:
                            g.materialize({n: v.to(g.theta0[n].device) for n, v in st.items()})
                        hh = g.current_hash()
                        res = geometry_path(b, g, enc, prefix, k, cands, cont if cont else None, hh, sup, beg)
                    finally:
                        g.restore()
                    counters["geometry_paths"] += 1
                    if not cont:
                        res["continuation"] = {"tokens": [], "H_eff": 0, "logprobs": [], "C_H": None}
                    res["state_hash"] = hh
                    res["action"] = action_class(res["argmax"], dv["B0_action"], dv["A2_action"])
                    if res["cands"]["B0"]["logp"] is None or res["cands"]["A2"]["logp"] is None:
                        invalid.append(f"{uid}: mandatory B0/A2 candidate not finite under {sname}")
                    if not (res["fed_ok"] and res["state_locked"]):
                        invalid.append(f"{uid}: geometry ownership {sname}")
                    geo[sname] = res
                drop_q = {}
                for dname in DROPS:
                    try:
                        g.materialize({n: v.to(g.theta0[n].device) for n, v in states[dname].items()})
                        hh = g.current_hash()
                        res = geometry_path(b, g, enc, prefix, k, {"B0": dv["B0_action"], "A2": dv["A2_action"]}, None, hh, sup, beg)
                    finally:
                        g.restore()
                    counters["drop_query_paths"] += 1
                    if not (res["fed_ok"] and res["state_locked"]) or hh != row["drop_state_hashes"][dname]:
                        invalid.append(f"{uid}: drop query ownership {dname}")
                    drop_q[dname] = {"argmax": res["argmax"], "action": action_class(res["argmax"], dv["B0_action"], dv["A2_action"]), "state_hash": hh,
                                     "B0_logp": res["cands"]["B0"]["logp"], "A2_logp": res["cands"]["A2"]["logp"]}
                if not (g.verify() and g.current_hash() == g.theta0_hash):
                    raise TTAInvalid("reset after geometry")
                row["geometry"] = geo
                row["drop_query"] = drop_q
                row["auto_alignment"] = {"k": k, "AUTO_action": auto_a, "AUTO_prefix_equals_common": auto_s[:k] == B0s[:k],
                                         "A2_action_eq_AUTO": (dv["A2_action"] == auto_a) if auto_a is not None else None,
                                         "B0_action_eq_AUTO": (dv["B0_action"] == auto_a) if auto_a is not None else None,
                                         "A2_H_eq_AUTO": (auto_s[k:k + len(cont)] == cont) if (cont and k + len(cont) <= len(auto_s)) else None}
            del enc
            row["phase2"] = True
            print(f"MECH0 phase2 {uid} {au['kind']} step1==A2 {row['diag_decodes']['A2_STEP1']['equals_A2']}"
                  + (f" acts {[row['drop_query'][d]['action'] for d in DROPS]}" if dv else ""), flush=True)
        runtime["phase2_sec"] = time.time() - t2
        for row in rows_out:
            row["status"] = "ok"
            atomic_json(out / "rows" / f"{row['canonical_index']:03d}.json", row)
        status = "completed" if not invalid else "failed"
        if invalid:
            runtime["failure"] = {"reason": "phase-2 integrity: " + "; ".join(invalid[:20])}
    except Exception as exc:
        import traceback
        status = "failed"
        runtime["failure"] = {"reason": repr(exc), "traceback": traceback.format_exc()}
        for row in rows_out:
            p = out / "rows" / f"{row['canonical_index']:03d}.json"
            if not p.exists():
                atomic_json(p, {**row, "status": "partial"})
        try:
            g.restore()
        except Exception:
            pass
    runtime["rows_written"] = sum(1 for _ in (out / "rows").glob("*.json"))
    runtime["peak_alloc"] = int(torch.cuda.max_memory_allocated())
    runtime["peak_reserved"] = int(torch.cuda.max_memory_reserved())
    runtime["reset_final_ok"] = bool(g.verify() and g.current_hash() == g.theta0_hash)
    runtime["nonln_unchanged"] = tensor_bytes_hash([p for n, p in b.model.named_parameters() if n not in names_set]) == nonln0
    runtime["model_grads_none"] = all(p.grad is None and not p.requires_grad for p in b.model.parameters())
    runtime.update(end_unix=time.time(), status=status, invalid=invalid, counters=counters)
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)
    if status != "completed":
        raise SystemExit("MECH0 run failed: " + runtime["failure"]["reason"])


def cmd_seal(args) -> None:
    run = ROOT / args.run
    files = sorted(p for p in run.rglob("*") if p.is_file()) + [ROOT / BASE / "primary_analysis.json"]
    man = json.loads((run / "manifest.json").read_text())
    prim = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    doc = {"schema": SCHEMA + "_output_seal", "run": args.run, "files": {str(p.relative_to(ROOT)): sha_file(p) for p in files},
           "manifest_hash": man["manifest_hash"], "source_commit": man["git_commit"], "config_sha256": sha_file(ROOT / CONFIG),
           "panel_sha256": sha_file(ROOT / PANEL), "plan_hash": json.loads((ROOT / PLAN).read_text())["plan_hash"], "primary_valid": prim["valid"],
           "rows_sealed": sum(1 for p in files if p.parent.name == "rows"), "state_archives_sealed": sum(1 for p in files if p.parent.name == "states"),
           "references_used": False, "created_unix": time.time()}
    doc["seal_hash"] = digest(doc)
    out = ROOT / BASE / "output_seal.json"
    if out.exists():
        raise FileExistsError("output seal exists")
    atomic_json(out, doc)
    print(json.dumps({"seal_hash": doc["seal_hash"], "files": len(doc["files"]), "primary_valid": doc["primary_valid"]}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("manifest").add_argument("--out", required=True)
    sub.add_parser("run").add_argument("--out", required=True)
    sub.add_parser("seal").add_argument("--run", required=True)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "manifest": cmd_manifest, "run": cmd_run, "seal": cmd_seal}[args.cmd](args)


if __name__ == "__main__":
    main()
