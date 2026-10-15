#!/usr/bin/env python
"""P2-1c on-policy continuation and switch-back failure audit (no training; docs/reft/P2_1C_PROTOCOL.md).

Population: the ten P2-1b prefix-audit cases, unchanged (p2_1b/prefix_audit/cases_private.json; d0..d4 damaged, s0..s4
safe). Systems: B0 and the selected R2-CE / R2-KL checkpoints of seeds 42/43/44 (CE42 = P2-1 epoch 2).

Commands
  freeze   (CPU) write-once p2_1c/freeze.json: sources, inputs, adapter hashes, historical P2-1b/P2-0 chains.
  anchors  (CPU, reference-aware, T0) generated-prefix anchors on actual beam-5 outputs -> anchors_private.json and the
           reference-free anchors_inference.json that the GPU steps read.
  smoke    (GPU, firewalled) T1 replay-correctness gate on two cases + compute estimate -> preflight_<job>.json.
  replay   (GPU, firewalled) T1 checks on every case, T2 B0-prefix continuation, T3 own-prefix replay, T4 forced anchors,
           T5 candidate scoring -> replay/continuation/candidate jsonl + integrity.json.
  analyze  (CPU, reference-aware) span scoring, T6, case-level flags, verdict, figures -> analysis.json.

Replay engine: transformers GenerationMixin.generate (the call Whisper's own generate makes internally) with an explicit
decoder prefix [SOT, LID, TRANSCRIBE, NOTIMESTAMPS, b_1..b_k], the canonical suppress list at every step and the
begin-suppress list only at absolute position 4 (never re-armed after a forced prefix), beam 5, absolute length budget
4 + 200, length penalty 1.0. Teacher-forced scores: full uncached forward, float32 log-softmax, logit at position i
scores token i + 1. Not acoustic causality; reference-informed anchors are exploratory and non-deployable.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import art, firewall, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from reft.provenance import git_state, now, sha, slurm, weights_hash  # noqa: E402

CONF = ROOT / "configs/reft/p2_1c_frozen.yaml"
FREEZE_SOURCES = ["docs/reft/P2_1C_PROTOCOL.md", "docs/reft/P2_1C_DECISION_RULES.md", "configs/reft/p2_1c_frozen.yaml",
                  "reft/p2_1c_continuation_audit.py", "tests/reft/test_p2_1c.py", "slurm/reft_p2_1c.sbatch",
                  "slurm/reft_p2_1c_cpu.sbatch"]
GPU_FORBIDDEN_EXTRA = ("cases_private", "anchors_private", "p2_1d/", "targets.json", "prefix_audit/", "posthoc/",
                       "p2_1b/evaluation", "p2_1b/report", "case_analysis", "p2_1c/analysis")
SYSTEMS = ("B0", "CE42", "CE43", "CE44", "KL42", "KL43", "KL44")
ADAPTED = SYSTEMS[1:]
B0_ANCHORS = ("start", "pre_en", "post_en", "early_zh")
INFERENCE_CASE_KEYS = {"key", "uid", "prompt", "b0_anchors", "own_anchors"}
STATUS = ("recovered", "partial", "dropped")
FLAGS = ("H1_PREFIX_HISTORY", "H2_LOCAL_SWITCHBACK", "H3_SEQUENCE_PATH")


def cc() -> dict:
    return yaml.safe_load(CONF.read_text())


def out(*p) -> Path:
    return art("p2_1c", *p)


# ====================================================================== pure helpers (unit-tested on CPU)

def boundaries(content_bytes: list[bytes]):
    """Clean content-token boundaries. m[k] = number of canonical units in the first k tokens if boundary k is clean,
    else None. Clean: prefix and suffix bytes are valid UTF-8 and their canonical units concatenate to exactly the
    units of the whole output (no unit is split; one Han character is not assumed to be one token)."""
    from reft.text import tokenize
    full = tokenize(b"".join(content_bytes).decode("utf-8", "replace"))
    ms = []
    for k in range(len(content_bytes) + 1):
        try:
            pu = tokenize(b"".join(content_bytes[:k]).decode("utf-8"))
            su = tokenize(b"".join(content_bytes[k:]).decode("utf-8"))
        except UnicodeDecodeError:
            ms.append(None)
            continue
        ms.append(len(pu) if pu + su == full else None)
    return ms, full


def anchor_at(ms: list, h: int) -> dict:
    """The largest clean boundary holding exactly h units; otherwise the nearest earlier clean boundary (approximate)."""
    ks = [k for k, m in enumerate(ms) if m == h]
    if ks:
        return {"k": max(ks), "units": h, "exact": True, "shortfall": 0}
    ks = [k for k, m in enumerate(ms) if m is not None and m < h]
    k = max(ks) if ks else 0
    return {"k": k, "units": ms[k] if ms[k] is not None else 0, "exact": False, "shortfall": h - (ms[k] or 0)}


def anchor_inside(ms: list, lo: int, hi: int) -> dict | None:
    """The first clean boundary holding between lo and hi units (inclusive); among equal counts the largest boundary.
    None if no clean boundary falls in the range."""
    ok = [(m, k) for k, m in enumerate(ms) if m is not None and lo <= m <= hi]
    if not ok:
        return None
    m = min(ok)[0]
    return {"k": max(k for mm, k in ok if mm == m), "units": m, "exact": True, "shortfall": 0}


def hyp_before(ops: list, r: int) -> int:
    """Hypothesis units emitted (matches, substitutions, insertions) before reference unit r's alignment op."""
    n = 0
    for o, i, j in ops:
        if i == r:
            return n
        n += j is not None
    raise KeyError(r)


def hyp_after(ops: list, r: int) -> int:
    """Hypothesis units up to and including reference unit r's own hypothesis unit (if aligned)."""
    n = hyp_before(ops, r)
    return n + next(j is not None for o, i, j in ops if i == r)


def en_region(tags: list, ref: list, sb: int, fill: set):
    """The English run before the switch-back: ends at the last substantive EN unit before sb, extends over EN units."""
    e = next((r for r in range(sb - 1, -1, -1) if tags[r] == "EN" and ref[r] not in fill), None)
    if e is None:
        return None
    r0 = e
    while r0 > 0 and tags[r0 - 1] == "EN":
        r0 -= 1
    return [r0, e]


def first_divergences(b0_tokens: list, s_tokens: list, b0_units: list, s_units: list, ms_b0: list) -> dict:
    """First token divergence (any id difference, incl. format/tokenization) vs first substantive lexical divergence
    (first differing canonical unit). Token divergence is also expressed in B0 unit coordinates."""
    n = min(len(b0_tokens), len(s_tokens))
    kt = next((i for i in range(n) if b0_tokens[i] != s_tokens[i]), None if len(b0_tokens) == len(s_tokens) else n)
    nu = min(len(b0_units), len(s_units))
    jl = next((i for i in range(nu) if b0_units[i] != s_units[i]), None if len(b0_units) == len(s_units) else nu)
    mt = None
    if kt is not None:
        mt = max(m for k, m in enumerate(ms_b0[:kt + 1]) if m is not None)
    return {"token_k": kt, "token_div_b0_units": mt, "lexical_unit": jl,
            "format_only_first": kt is not None and (jl is None or mt < jl)}


def cand_logps(lp, plen: int, c: list) -> list:
    """Per-token log-probabilities of continuation c after a prefix of length plen: logit row plen - 1 + t scores c[t].
    Prompt/prefix tokens are never scored."""
    return [float(lp[plen - 1 + t, y]) for t, y in enumerate(c)]


def beam_equivalent(logps: list, c: list, suppressed: set, length_penalty: float = 1.0):
    """HF beam-search finished-hypothesis score of a forced-prefix continuation: sum log p / len(generated) ** lp.
    None if a token the canonical decoder suppresses occurs (its constrained score is -inf)."""
    if not c or any(t in suppressed for t in c):
        return None
    return float(sum(logps) / (len(c) ** length_penalty))


def status_of(retention, R: dict):
    if retention is None:
        return None
    return "recovered" if retention >= R["recovered_min_retention"] else "dropped" if retention <= R["dropped_max_retention"] \
        else "partial"


def local_disfavor(rank_s: int, logp_s: float, logp_b0: float, R: dict) -> bool:
    """L: under the same B0 prefix the adapted model no longer ranks B0's switch-back token first, or lowers it by
    >= local_drop_nats relative to B0."""
    return rank_s > 1 or (logp_s - logp_b0) <= -R["local_drop_nats"]


def seq_preference(cands: list, model: str):
    """Q under `model`: best beam-equivalent score of omission candidates > best of span-retaining candidates.
    None when either class is empty (no valid same-prefix competitor)."""
    best = {}
    for c in cands:
        s = c["scores"][model]["beam_eq"]
        if s is None or c["status"] not in ("recovered", "dropped"):
            continue
        best[c["status"]] = max(best.get(c["status"], -1e30), s)
    if "recovered" not in best or "dropped" not in best:
        return None, None
    return best["dropped"] > best["recovered"], best["recovered"] - best["dropped"]


def flags_for(orig_drop: bool, B, O, own_valid: bool, identical_prefix: bool, L, Q) -> list:
    """Case x system mechanism flags (docs/reft/P2_1C_DECISION_RULES.md). Several may hold; none -> unresolved."""
    if not orig_drop:
        return []
    f = []
    if B == "recovered" and own_valid and O == "dropped" and not identical_prefix:
        f.append("H1_PREFIX_HISTORY")
    if B == "dropped" and L:
        f.append("H2_LOCAL_SWITCHBACK")
    if (B == "dropped" and L is False and Q is True) or (own_valid and O == "recovered"):
        f.append("H3_SEQUENCE_PATH")
    return f


def aggregate_cells(rows: list, R: dict) -> dict:
    """(damaged case, arm) cells. A cell is applicable when >= seed_consistent_min seeds of the arm originally drop the
    span; a flag is seed-consistent in the cell when >= seed_consistent_min of those seeds carry it."""
    cells = {}
    for case in sorted({r["case"] for r in rows}):
        for arm in ("CE", "KL"):
            rr = [r for r in rows if r["case"] == case and r["system"].startswith(arm)]
            app = [r for r in rr if r["orig_drop"]]
            cell = {"n_seeds_orig_drop": len(app), "applicable": len(app) >= R["seed_consistent_min"], "supported": [],
                    "counterexamples": {}}
            if cell["applicable"]:
                for fl in FLAGS:
                    n = sum(fl in r["flags"] for r in app)
                    if n >= R["seed_consistent_min"]:
                        cell["supported"].append(fl)
                        miss = [r["system"] for r in app if fl not in r["flags"]]
                        if miss:
                            cell["counterexamples"][fl] = miss
            cells[f"{case}/{arm}"] = cell
    return cells


def decide(integrity_ok: bool, n_valid_damaged: int, cells: dict, R: dict) -> dict:
    """Terminal verdict and the single research recommendation (fixed before execution)."""
    app = {k: v for k, v in cells.items() if v["applicable"]}
    share = {}
    for fl in FLAGS:
        sup = [k for k, v in app.items() if fl in v["supported"]]
        share[fl] = {"cells": len(sup), "of": len(app), "fraction": len(sup) / len(app) if app else 0.0,
                     "cases": len({k.split("/")[0] for k in sup})}
    dom = [fl for fl in FLAGS if app and share[fl]["fraction"] >= R["dominance_fraction"]
           and share[fl]["cases"] >= R["dominance_min_cases"]]
    go = {"H1_PREFIX_HISTORY": "GO_P2_2_ON_POLICY_PRESERVATION_DESIGN",
          "H2_LOCAL_SWITCHBACK": "GO_P2_2_SWITCHBACK_AWARE_ADAPTATION",
          "H3_SEQUENCE_PATH": "GO_P2_2_SEQUENCE_LEVEL_STABILITY_DESIGN"}
    if not integrity_ok:
        return {"verdict": "P2_1C_ENGINEERING_BLOCKED", "recommendation": "HOLD_FOR_ENGINEERING_OR_ALIGNMENT_REPAIR",
                "dominant": [], "share": share}
    if n_valid_damaged < R["min_valid_damaged_anchors"]:
        return {"verdict": "P2_1C_INCONCLUSIVE", "recommendation": "HOLD_FOR_ENGINEERING_OR_ALIGNMENT_REPAIR",
                "dominant": [], "share": share}
    if len(dom) == 1:
        return {"verdict": "P2_1C_DIAGNOSTIC_COMPLETE", "recommendation": go[dom[0]], "dominant": dom, "share": share}
    lead = max(FLAGS, key=lambda f: (share[f]["cells"], -FLAGS.index(f)))
    tie = sum(share[f]["cells"] == share[lead]["cells"] for f in FLAGS) > 1
    if dom or any(share[f]["cells"] >= 2 for f in FLAGS):
        rec = go[lead] if (not tie and share[lead]["fraction"] >= 0.5 and share[lead]["cases"] >= R["dominance_min_cases"]) \
            else "GO_P2_2_ACOUSTIC_ARCHITECTURE_SCREENING"
        return {"verdict": "P2_1C_PARTIALLY_IDENTIFIABLE", "recommendation": rec, "dominant": dom, "share": share}
    return {"verdict": "P2_1C_INCONCLUSIVE", "recommendation": "GO_P2_2_ACOUSTIC_ARCHITECTURE_SCREENING", "dominant": [],
            "share": share}


def inference_manifest_clean(m: dict) -> bool:
    """The GPU-side manifest holds only ids, prompts and token counts: no reference strings or reference-unit fields."""
    return all(set(c) == INFERENCE_CASE_KEYS and all(v is None or isinstance(v, int) for v in c["b0_anchors"].values())
               and all(v is None or isinstance(v, int) for v in c["own_anchors"].values()) for c in m["cases"])


# ====================================================================== systems, freeze

def system_paths() -> dict:
    res = {"B0": {"decode": art("p2_0/frozen_baseline_dev150_beam5.jsonl")}}
    for name, s in cc()["systems"].items():
        if s is None:
            continue
        rd = art(s["run"])
        ep = read_json(rd / "selection.json")["selected"]["epoch"]
        if "expected_epoch" in s and ep != s["expected_epoch"]:
            raise SystemExit(f"{name}: selected epoch {ep} != {s['expected_epoch']}")
        h = next(h for h in read_json(rd / "manifest.json")["history"] if h["epoch"] == ep)
        res[name] = {"run": s["run"], "seed": s["seed"], "epoch": ep, "adapter": rd / f"epoch{ep}" / "adapter.pt",
                     "decode": rd / f"epoch{ep}" / "dev_beam5.jsonl", "expected_sha": h["adapter_sha256"]}
    return res


def frozen_inputs() -> list[str]:
    L = ["p2_0/dev_manifest.json", "p2_0/frozen_baseline_dev150_beam5.jsonl", "p2_1b/prefix_audit/cases_private.json",
         "p2_1b/prefix_audit/results_private.jsonl", "p2_1b/prefix_audit/summary.json",
         "p2_1b/posthoc/span_recovery_59293_private.json", "p2_1b/report.json", "p2_1b/evaluation.json",
         "p2_1b/verify.json", "p2_1/verify.json"]
    for n, s in system_paths().items():
        if n != "B0":
            L += [f"{s['run']}/selection.json", f"{s['run']}/manifest.json", f"{s['run']}/epoch{s['epoch']}/adapter.pt",
                  f"{s['run']}/epoch{s['epoch']}/dev_beam5.jsonl"]
    return L


def adapters_ok() -> dict:
    return {n: sha(s["adapter"]) == s["expected_sha"] for n, s in system_paths().items() if n != "B0"}


def run_freeze():
    p = out("freeze.json")
    if p.exists():
        raise SystemExit(f"{p} exists; write-once")
    from reft.p2_1b_train import check_freeze as p21b
    g = git_state()
    if g["branch"] != "cs-asr-reft":
        raise SystemExit(f"wrong branch {g['branch']}")
    ad = adapters_ok()
    if not all(ad.values()):
        raise SystemExit(f"adapter hash mismatch: {ad}")
    rec = {"schema": "reft_p2_1c_freeze_v1", "utc": now(), "git": g, "sources": {s: sha(ROOT / s) for s in FREEZE_SOURCES},
           "inputs": {i: sha(art(i)) for i in frozen_inputs()}, "adapters_match_recorded_sha": ad, "p2_1b_chain": p21b(),
           "authorization": "P2-1c bounded no-training diagnostic authorized by the project owner (2026-10-15); "
                            "no P2-2/P3, no training, no new data"}
    print(write_json(p, rec))


def gpu_firewall():
    from reft.train_supervised import GPU_FORBIDDEN
    firewall(GPU_FORBIDDEN + GPU_FORBIDDEN_EXTRA)


def check_freeze(gpu: bool = False) -> dict:
    """Sources and inputs unchanged; on CPU also the historical P2-1b -> P2-0 chain. In GPU (firewalled) mode the
    reference-bearing inputs and the historical chain are not opened; the CPU steps verify them."""
    p = out("freeze.json")
    rec = read_json(p)
    skip = (lambda i: any(x in i for x in GPU_FORBIDDEN_EXTRA)) if gpu else (lambda i: False)
    bad = [s for s, h in rec["sources"].items() if sha(ROOT / s) != h] + \
          [i for i, h in rec["inputs"].items() if not skip(i) and sha(art(i)) != h]
    if bad:
        raise SystemExit(f"P2-1c frozen sources/inputs changed: {bad}")
    if not gpu:
        from reft.p2_1b_train import check_freeze as p21b
        if p21b() != rec["p2_1b_chain"]:
            raise SystemExit("P2-1b / P2-0 freeze chain changed")
    return {"p2_1c_freeze_sha256": sha(p), **rec["p2_1b_chain"], "historical_chain_checked_here": not gpu}


# ====================================================================== T0 anchors (CPU, reference-aware)

def load_outputs() -> dict:
    return {n: {x["id"]: x for x in read_jsonl(s["decode"])} for n, s in system_paths().items()}


def run_anchors():
    from transformers import WhisperTokenizer
    from reft.common import cfg, fillers
    from reft.evaluate_supervised import dev_refs
    from reft.text import one, tok_bytes_fn
    fz = check_freeze()
    if out("anchors_private.json").exists():
        raise SystemExit("anchors already frozen")
    C = cc()
    cases = read_json(art(C["cases"]))
    tok = WhisperTokenizer.from_pretrained(cfg()["model"]["path"], local_files_only=True)
    tb, eos = tok_bytes_fn(tok), tok.eos_token_id
    FILL = fillers()
    O = load_outputs()
    dm = {r["utterance_id"]: r for r in read_json(art("p2_0/dev_manifest.json"))["rows"]}
    allc = [("d", i, c) for i, c in enumerate(cases["damaged"])] + [("s", i, c) for i, c in enumerate(cases["safe"])]
    refs = dev_refs(sorted({c["id"] for _, _, c in allc}))
    group = {k: "recovered" for k in C["recovered_spans"]} | {k: "persistent" for k in C["persistent_spans"]}
    priv, inf, recorded = [], [], {}
    for kind, i, c in allc:
        key, uid = f"{kind}{i}", c["id"]
        ref = refs[uid]
        b0 = O["B0"][uid]
        assert b0["prompt"] == dm[uid]["prompt"] == c["prompt"], key
        content = [t for t in b0["tokens"] if t != eos]
        ms, full = boundaries([tb(t) for t in content])
        o = one(ref, b0["text"])
        assert full == o["hyp"], f"{key}: token-byte units != scorer units"
        sb, span = c["sb_unit"], sorted(c["span_units"])
        sw = span[0]   # switch-back location = first (B0-correct, damaged) span unit; sb_unit itself may be B0-wrong
        reg = en_region(o["tags"], o["ref"], sb, FILL)
        op = lambda r: o["by"][r][0]  # noqa: E731
        A = {"start": {"k": 0, "units": 0, "exact": True, "shortfall": 0, "ref_unit": None, "ref_op": None, "valid": True}}
        if reg:
            a = anchor_at(ms, hyp_before(o["ops"], reg[0]))
            A["pre_en"] = {**a, "ref_unit": reg[0], "ref_op": op(reg[0]), "valid": a["exact"] and op(reg[0]) in ("match", "sub")}
        else:
            A["pre_en"] = {"k": None, "valid": False, "reason": "no substantive English before the switch-back"}
        a = anchor_at(ms, hyp_before(o["ops"], sw))
        A["post_en"] = {**a, "ref_unit": sw, "ref_op": op(sw), "sb_unit_op": op(sb), "valid": a["exact"] and op(sw) == "match"}
        n_e = C["anchors"]["early_zh_after_span_units"]
        a = anchor_inside(ms, hyp_after(o["ops"], span[n_e - 1]), hyp_before(o["ops"], span[-1])) if len(span) > n_e else None
        if a is not None:
            scope = [r for r in span if o["by"][r][1] is not None and o["by"][r][1] >= a["units"]]
            A["early_zh"] = {**a, "ref_unit": span[n_e - 1], "ref_op": op(span[n_e - 1]), "scope_units": scope,
                             "valid": all(op(r) == "match" for r in span[:n_e]) and len(scope) > 0}
        else:
            A["early_zh"] = {"k": None, "valid": False, "reason": "no clean boundary inside the span after the first units"}
        for a in A.values():
            if a.get("k") is not None:
                a["has_next_token"] = a["k"] < len(b0["tokens"])
                a["valid"] = a["valid"] and a["has_next_token"]
        own, divs = {}, {}
        for s in ADAPTED:
            r_ = O[s][uid]
            assert r_["prompt"] == b0["prompt"], (key, s)
            sc = [t for t in r_["tokens"] if t != eos]
            msS, fullS = boundaries([tb(t) for t in sc])
            oS = one(ref, r_["text"])
            assert fullS == oS["hyp"], (key, s)
            divs[s] = first_divergences(b0["tokens"], r_["tokens"], full, fullS, ms)
            last = next((r for r in range(sw - 1, -1, -1) if oS["by"][r][0] in ("match", "sub")), None)
            if last is None:
                own[s] = {"k": None, "valid": False, "reason": "no aligned unit before the switch-back"}
                continue
            a = anchor_at(msS, hyp_after(oS["ops"], last))
            common = reg is not None and last >= reg[0]
            own[s] = {**a, "ref_unit": last, "ref_op": oS["by"][last][0], "common_location": common,
                      "valid": a["exact"] and common and a["k"] < len(r_["tokens"]),
                      "context_lexically_matched": fullS[:a["units"]] == full[:A["post_en"]["units"]],
                      "identical_prefix_tokens": r_["tokens"][:a["k"]] == b0["tokens"][:A["post_en"]["k"]]}
        priv.append({"key": key, "kind": c["kind"], "group": group.get(key, "safe"), "uid": uid, "sb_unit": sb, "switch_unit": sw,
                     "span_units": span, "en_region": reg, "b0_span_retained": float(np.mean([op(r) == "match" for r in span])),
                     "b0_n_units": len(full), "b0_n_tokens": len(b0["tokens"]), "b0_boundaries": ms, "b0_anchors": A,
                     "own_anchors": own, "divergence_vs_b0": divs})
        inf.append({"key": key, "uid": uid, "prompt": b0["prompt"],
                    "b0_anchors": {n: A[n].get("k") for n in B0_ANCHORS}, "own_anchors": {s: own[s].get("k") for s in ADAPTED}})
    # recorded P2-1b teacher-forced rescoring (reference-free numbers) for the T1 reload/score reproduction check
    names = {"CE42_e2_P2-1": "CE42"}
    for n, s in system_paths().items():
        if n != "B0" and n != "CE42":
            names[f"{n}_e{s['epoch']}"] = n
    uid2key = {c["uid"]: c["key"] for c in inf}
    for r in read_jsonl(art("p2_1b/prefix_audit/results_private.jsonl")):
        if r["system"] in names:
            recorded.setdefault(uid2key[r["case"]], {})[names[r["system"]]] = {
                "sys_out_under_sys": r["rescoring"]["sys_out_under_sys"]["sum"],
                "b0_out_under_b0": r["rescoring"]["b0_out_under_b0"]["sum"]}
    m = {"schema": "reft_p2_1c_anchors_inference_v1", "utc": now(), "cases": inf, "recorded_scores": recorded}
    assert inference_manifest_clean(m)
    write_json(out("anchors_private.json"), {"schema": "reft_p2_1c_anchors_private_v1", "utc": now(), **fz, "cases": priv,
                                             "note": "reference-informed anchors: exploratory, non-deployable"})
    write_json(out("anchors_inference.json"), m)
    summ = {c["key"]: {n: (c["b0_anchors"][n].get("k"), c["b0_anchors"][n]["valid"]) for n in B0_ANCHORS} for c in priv}
    g2 = {"all_cases_accounted": len(priv) == len(cases["damaged"]) + len(cases["safe"]) == 10,
          "valid_post_en_damaged": sum(c["b0_anchors"]["post_en"]["valid"] for c in priv if c["kind"] == "damaged"),
          "inference_manifest_clean": inference_manifest_clean(m), "anchors": summ,
          "own_valid": {c["key"]: {s: c["own_anchors"][s]["valid"] for s in ADAPTED} for c in priv}}
    write_json(out("anchors_check.json"), {"utc": now(), **fz, "slurm": slurm(), **g2,
                                           "anchors_private_sha256": sha(out("anchors_private.json")),
                                           "anchors_inference_sha256": sha(out("anchors_inference.json"))})
    print(json.dumps(g2, indent=1))


# ====================================================================== GPU replay engine

class Engine:
    """Frozen backbone + the six selected adapters; reference-free (firewall on before construction)."""

    def __init__(self):
        import torch
        from reft.adapters import make_adapter
        from reft.p2_1b_train import canonical_env_or_exit, seed_config
        from reft.text import tok_bytes_fn
        from reft.train_supervised import c21, load_backbone
        self.torch, self.C = torch, cc()
        self.b, self.env = load_backbone()
        self.libs = canonical_env_or_exit()
        self.w0 = weights_hash(self.b.model)
        gj = json.loads((Path(__import__("reft.common", fromlist=["cfg"]).cfg()["model"]["path"]) / "generation_config.json").read_text())
        gen = self.b.model.generation_config
        assert list(gen.suppress_tokens) == gj["suppress_tokens"] and list(gen.begin_suppress_tokens) == gj["begin_suppress_tokens"]
        self.supp, self.begin_supp = list(gj["suppress_tokens"]), list(gj["begin_suppress_tokens"])
        self.sot, self.eos = gen.decoder_start_token_id, self.b.processor.tokenizer.eos_token_id
        self.pad, self.bos = gen.pad_token_id, gen.bos_token_id
        self.tb = tok_bytes_fn(self.b.processor.tokenizer)
        self.paths = {r["utterance_id"]: r["audio_path"] for r in read_json(art("p2_0/dev_manifest.json"))["rows"]}
        self.sp = system_paths()
        self.adapter_sha_ok = adapters_ok()
        C1 = c21()
        self.ads = {"B0": None}
        for n in ADAPTED:
            a = make_adapter("R2", self.b.model, seed_config(C1, self.sp[n]["seed"]), self.b.device)
            a.load_state_dict(torch.load(self.sp[n]["adapter"]))
            a.module.eval()
            for p in a.module.parameters():
                p.requires_grad_(False)
            self.ads[n] = a
        self.outputs = load_outputs()
        self.n_tf = self.n_gen = self.n_gen_steps = 0
        self.inp = self.enc = None

    def set_case(self, uid):
        from reft.train_supervised import encode
        self.inp, self.enc = encode(self.b, [self.paths[uid]])

    def ctx(self, name):
        a = self.ads[name]
        return a.active(self.b.model) if a is not None else contextlib.nullcontext()

    def lp_full(self, name, X):
        torch = self.torch
        with torch.inference_mode(), self.ctx(name):
            z = self.b.model(encoder_outputs=(self.enc,), decoder_input_ids=torch.tensor([X], device=self.b.device),
                             use_cache=False).logits[0].float()
        self.n_tf += 1
        return torch.log_softmax(z, -1)

    def next_token(self, name, P, targets: dict) -> dict:
        lp = self.lp_full(name, P)[-1]
        v, ix = lp.topk(5)
        res = {"eos_logp": float(lp[self.eos]), "top5": [[int(i), float(x)] for i, x in zip(ix, v)],
               "finite": bool(self.torch.isfinite(lp).all())}
        for key, t in targets.items():
            if t is None:
                continue
            o = lp.clone()
            o[t] = -float("inf")
            res[key] = {"id": int(t), "logp": float(lp[t]), "rank": int((lp > lp[t]).sum()) + 1,
                        "margin_best_other": float(lp[t] - o.max()), "margin_vs_eos": float(lp[t] - lp[self.eos])}
        return res

    def generate(self, name, P) -> dict:
        torch = self.torch
        from transformers import GenerationConfig
        from transformers.generation.logits_process import (LogitsProcessorList, SuppressTokensAtBeginLogitsProcessor,
                                                            SuppressTokensLogitsProcessor)
        from transformers.generation.utils import GenerationMixin
        R = self.C["replay"]
        dev = self.b.device
        gc = GenerationConfig(num_beams=R["num_beams"], num_return_sequences=R["num_return_sequences"], do_sample=False,
                              max_length=R["total_max_length"], length_penalty=R["length_penalty"],
                              early_stopping=R["early_stopping"], eos_token_id=self.eos, pad_token_id=self.pad,
                              bos_token_id=self.bos, decoder_start_token_id=self.sot, return_dict_in_generate=True,
                              output_scores=True, use_cache=True)
        procs = LogitsProcessorList([SuppressTokensAtBeginLogitsProcessor(self.begin_supp, begin_index=R["begin_suppress_index"], device=dev),
                                     SuppressTokensLogitsProcessor(self.supp, device=dev)])
        t0 = time.time()
        with torch.inference_mode(), self.ctx(name):
            o = GenerationMixin.generate(self.b.model, **self.inp, decoder_input_ids=torch.tensor([P], device=dev),
                                         generation_config=gc, logits_processor=procs)
        torch.cuda.synchronize()
        beams = []
        for s, sc in zip(o.sequences.tolist(), o.sequences_scores.tolist()):
            assert s[:len(P)] == P, "forced prefix not preserved"
            suf = s[len(P):]
            if self.eos in suf:
                suf = suf[:suf.index(self.eos) + 1]
            beams.append({"suffix": suf, "score": float(sc), "terminated": self.eos in suf})
        self.n_gen += 1
        self.n_gen_steps += max(len(s) for s in o.sequences.tolist()) - len(P)
        return {"beams": beams, "seconds": time.time() - t0}

    def score(self, name, P, c) -> dict:
        lp = self.lp_full(name, P + c[:-1]) if len(c) > 1 else self.lp_full(name, P)
        lps = cand_logps(lp, len(P), c)
        sup = set(self.supp) | (set(self.begin_supp) if len(P) == self.C["replay"]["begin_suppress_index"] else set())
        bad_first = len(P) == self.C["replay"]["begin_suppress_index"] and c and c[0] in self.begin_supp
        be = None if bad_first else beam_equivalent(lps, c, set(self.supp), self.C["replay"]["length_penalty"])
        return {"sum": float(sum(lps)), "mean": float(np.mean(lps)) if lps else None, "len": len(c),
                "beam_eq": be, "eos_included": bool(c) and c[-1] == self.eos, "constrained_valid": be is not None,
                "logps": lps, "suppressed_tokens": sorted(set(c) & sup)}

    # ---------------------------------------------------------------- T1 checks
    def cached_vs_full(self, name, X) -> dict:
        """Incremental KV-cached decoding vs one full forward on the same prefix; adapter positions spied."""
        torch = self.torch
        seen = []
        L = self.b.model.model.decoder.layers[self.C.get("site_layer", 31)]
        h = L.register_forward_pre_hook(lambda m, a, k: seen.append(k["cache_position"].tolist()), with_kwargs=True)
        try:
            full = self.lp_full(name, X)
            full_pos = seen[-1]
            steps = []
            with torch.inference_mode(), self.ctx(name):
                dev = self.b.device
                o = self.b.model(encoder_outputs=(self.enc,), decoder_input_ids=torch.tensor([X[:4]], device=dev), use_cache=True)
                rows = [torch.log_softmax(o.logits[0].float(), -1)]
                steps.append(seen[-1])
                pkv = o.past_key_values
                for t in range(4, len(X)):
                    o = self.b.model(encoder_outputs=(self.enc,), decoder_input_ids=torch.tensor([[X[t]]], device=dev),
                                     past_key_values=pkv, use_cache=True, cache_position=torch.tensor([t], device=dev))
                    pkv = o.past_key_values
                    rows.append(torch.log_softmax(o.logits[0].float(), -1))
                    steps.append(seen[-1])
                inc = torch.cat(rows, 0)
        finally:
            h.remove()
        obs = torch.tensor(X[1:], device=full.device)
        d_obs = (full[:-1].gather(1, obs[:, None]) - inc[:-1].gather(1, obs[:, None])).abs().max().item()
        top2 = full.topk(2, -1).values
        sure = (top2[:, 0] - top2[:, 1]) >= self.C["tolerance"]["incremental_vs_full_argmax_margin"]
        agree = bool((full.argmax(-1) == inc.argmax(-1))[sure].all())
        pos_ok = full_pos == list(range(len(X))) and steps[0] == [0, 1, 2, 3] and all(s == [t] for s, t in zip(steps[1:], range(4, len(X))))
        return {"max_abs_logp_observed": d_obs, "argmax_agree_where_sure": agree, "n_positions": len(X),
                "positions_ok": pos_ok, "finite": bool(torch.isfinite(full).all() and torch.isfinite(inc).all())}

    def hooks_clear(self) -> bool:
        from reft.adapters import site_hook_count
        return site_hook_count(self.b.model, 31) == 0

    def validate_case(self, case: dict, recorded: dict) -> dict:
        """T1 gate on one case: parity of the empty-prefix replay with every recorded original decode, beam score vs
        teacher-forced score, recorded P2-1b scores, cached vs full, early positions untouched, determinism, hooks."""
        T = self.C["tolerance"]
        uid, P0 = case["uid"], case["prompt"]
        self.set_case(uid)
        res = {"parity": {}, "beam_vs_tf": {}, "recorded": {}, "cached_vs_full": {}}
        for n in SYSTEMS:
            orig = self.outputs[n][uid]["tokens"]
            g = self.generate(n, P0)
            top = g["beams"][0]
            res["parity"][n] = top["suffix"] == orig
            s = self.score(n, P0, orig)
            res["beam_vs_tf"][n] = None if s["beam_eq"] is None else abs(s["beam_eq"] - top["score"])
            rec = recorded.get(n)
            if rec is not None:
                res["recorded"][n] = abs(s["sum"] - rec["sys_out_under_sys"])
        rec0 = next(iter(recorded.values()), None)
        if rec0 is not None:
            res["recorded"]["B0"] = abs(self.score("B0", P0, self.outputs["B0"][uid]["tokens"])["sum"] - rec0["b0_out_under_b0"])
        X = P0 + self.outputs["B0"][uid]["tokens"][:-1]
        for n in ("B0", "CE42", "KL42"):
            res["cached_vs_full"][n] = self.cached_vs_full(n, X)
        z0 = self.lp_full("B0", X)[:3]
        res["early_positions_untouched"] = all(bool(self.torch.equal(self.lp_full(n, X)[:3], z0)) for n in ADAPTED)
        k = case["b0_anchors"]["post_en"] or 0
        P = P0 + self.outputs["B0"][uid]["tokens"][:k]
        g1, g2 = self.generate("CE42", P), self.generate("CE42", P)
        res["deterministic"] = [b["suffix"] for b in g1["beams"]] == [b["suffix"] for b in g2["beams"]] and \
            [b["score"] for b in g1["beams"]] == [b["score"] for b in g2["beams"]]
        res["hooks_clear"] = self.hooks_clear()
        res["ok"] = (all(res["parity"].values())
                     and all(v is not None and v <= T["beam_score_vs_teacher_forced"] for v in res["beam_vs_tf"].values())
                     and all(v <= T["recorded_score_abs"] for v in res["recorded"].values())
                     and all(v["max_abs_logp_observed"] <= T["incremental_vs_full_max_abs_logp"] and v["argmax_agree_where_sure"]
                             and v["positions_ok"] and v["finite"] for v in res["cached_vs_full"].values())
                     and res["early_positions_untouched"] and res["deterministic"] and res["hooks_clear"])
        return res

    # ---------------------------------------------------------------- T2-T5 on one case
    def run_case(self, case: dict):
        uid, P0 = case["uid"], case["prompt"]
        self.set_case(uid)
        b0t = self.outputs["B0"][uid]["tokens"]
        kpost = case["b0_anchors"]["post_en"]
        rep, cont = [], []
        conds = []
        for a in B0_ANCHORS:
            k = case["b0_anchors"][a]
            if k is not None:
                conds += [(a, "B0", k, m) for m in SYSTEMS]
        cp = self.C["counterpart"]
        for s in ADAPTED:
            k = case["own_anchors"][s]
            if k is not None:
                conds += [("own", s, k, m) for m in (s, "B0", cp[s])]
        for a, src, k, m in conds:
            toks = self.outputs[src][uid]["tokens"]
            P = P0 + toks[:k]
            tg = {"src_next": toks[k] if k < len(toks) else None}
            if a == "own":
                tg["b0_switch"] = b0t[kpost] if kpost is not None and kpost < len(b0t) else None
            rep.append({"case": case["key"], "anchor": a, "prefix_source": src, "k": k, "model": m, **self.next_token(m, P, tg)})
            g = self.generate(m, P)
            cont.append({"case": case["key"], "anchor": a, "prefix_source": src, "k": k, "model": m, **g})
        cands_out = []
        for a in self.C["anchors"]["candidate_pool_anchors"]:
            k = case["b0_anchors"][a]
            if k is None:
                continue
            P = P0 + b0t[:k]
            pool = collections.OrderedDict()
            pool[tuple(b0t[k:])] = ["B0_original"]
            for c in cont:
                if c["anchor"] == a and c["prefix_source"] == "B0":
                    for i, bm in enumerate(c["beams"]):
                        pool.setdefault(tuple(bm["suffix"]), []).append(f"{c['model']}_beam{i}")
            for ci, (cand, srcs) in enumerate(pool.items()):
                cands_out.append({"case": case["key"], "anchor": a, "k": k, "cand": ci, "sources": srcs, "tokens": list(cand),
                                  "scores": {m: self.score(m, P, list(cand)) for m in SYSTEMS}})
            # beam score of every generated beam vs its teacher-forced beam-equivalent score under the generating model
            for c in cont:
                if c["anchor"] == a and c["prefix_source"] == "B0":
                    for i, bm in enumerate(c["beams"]):
                        sc = next(x for x in cands_out if x["anchor"] == a and x["tokens"] == bm["suffix"])["scores"][c["model"]]
                        bm["tf_beam_eq_abs_diff"] = None if sc["beam_eq"] is None else abs(sc["beam_eq"] - bm["score"])
        return rep, cont, cands_out


# ====================================================================== GPU commands

def newest_preflight():
    ps = sorted(out().glob("preflight_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    return ps[-1] if ps else None


def run_smoke():
    gpu_firewall()
    fz = check_freeze(gpu=True)
    m = read_json(out("anchors_inference.json"))
    chk = read_json(out("anchors_check.json"))
    t_start = time.time()
    E = Engine()
    t_load = time.time() - t_start
    sub = [c for c in m["cases"] if c["key"] in ("d0", "s0")]
    t_v = time.time()
    val = {c["key"]: E.validate_case(c, m["recorded_scores"].get(c["key"], {})) for c in sub}
    t1 = time.time()
    n0g, n0f = E.n_gen, E.n_tf
    E.run_case(sub[0])
    dt, dg, df = time.time() - t1, E.n_gen - n0g, E.n_tf - n0f
    per_case = dt
    n_cases = len(m["cases"])
    est_min = (t_load + (t1 - t_v) / len(sub) * n_cases + per_case * (n_cases + 1)) / 60
    B = cc()["budget"]
    w1 = weights_hash(E.b.model)
    from reft.common import cfg
    gates = {
        "G1_provenance": {"freeze_ok": True, **{k: v for k, v in fz.items() if k != "historical_chain_checked_here"},
                          "historical_chain_verified_by_cpu_steps": True, "branch": git_state()["branch"] == "cs-asr-reft",
                          "adapters_sha_ok": all(E.adapter_sha_ok.values()), "weights_expected": E.w0 == cfg()["model"]["expected_weights_hash"]},
        "G2_anchors": {"all_cases_accounted": chk["all_cases_accounted"], "inference_manifest_clean": inference_manifest_clean(m),
                       "valid_post_en_damaged": chk["valid_post_en_damaged"]},
        "G3_replay": {k: v["ok"] for k, v in val.items()},
        "G4_validity": {"format_vs_lexical_divergence_separate": True, "beam_vs_conditional_scores_separate": True,
                        "cpu_tests": "passed earlier in this job (set -e)"},
        "G5_compute": {"estimated_full_replay_minutes": est_min, "target": B["target_gpu_minutes"], "hard": B["hard_gpu_minutes"],
                       "case_seconds": per_case, "case_generate_calls": dg, "case_tf_forwards": df},
        "G6_reproducibility": {"deterministic": all(v["deterministic"] for v in val.values()), "weights_unchanged": E.w0 == w1},
        "G7_rules_frozen": {"decision_rules_sha256": read_json(out("freeze.json"))["sources"]["docs/reft/P2_1C_DECISION_RULES.md"]},
    }
    ok = {"G1": all(v for k, v in gates["G1_provenance"].items() if isinstance(v, bool)),
          "G2": gates["G2_anchors"]["all_cases_accounted"] and gates["G2_anchors"]["inference_manifest_clean"]
          and gates["G2_anchors"]["valid_post_en_damaged"] >= cc()["rules"]["min_valid_damaged_anchors"],
          "G3": all(gates["G3_replay"].values()), "G4": True, "G5": est_min <= B["hard_gpu_minutes"],
          "G6": all(gates["G6_reproducibility"].values()), "G7": True}
    bad = [k for k, v in ok.items() if not v]
    verdict = "READY_FOR_P2_1C_EXECUTION" if not bad else f"P2_1C_BLOCKED_PREFLIGHT_{bad[0]}"
    rec = {"schema": "reft_p2_1c_preflight_v1", "utc": now(), "slurm": slurm(), "cuda_libs": E.libs, "gates": gates,
           "gate_ok": ok, "validation": val, "verdict": verdict, "seconds": time.time() - t_start,
           "generate_calls": E.n_gen, "tf_forwards": E.n_tf}
    write_json(out(f"preflight_{slurm()['SLURM_JOB_ID']}.json"), rec)
    print(json.dumps({"verdict": verdict, "gate_ok": ok, "G5": gates["G5_compute"], "validation": val}, indent=1, default=str)[:8000])


def run_replay():
    gpu_firewall()
    fz = check_freeze(gpu=True)
    pf = newest_preflight()
    if pf is None or read_json(pf)["verdict"] != "READY_FOR_P2_1C_EXECUTION":
        raise SystemExit(f"preflight not READY: {pf}")
    for f in ("replay_private.jsonl", "continuation_private.jsonl", "candidate_scores_private.jsonl", "integrity.json"):
        if out(f).exists():
            raise SystemExit(f"{f} exists; write-once")
    m = read_json(out("anchors_inference.json"))
    t0 = time.time()
    E = Engine()
    val = {c["key"]: E.validate_case(c, m["recorded_scores"].get(c["key"], {})) for c in m["cases"]}
    t_val = time.time() - t0
    rep, cont, cand = [], [], []
    first, partial = None, False
    for c in m["cases"]:
        if (time.time() - t0) / 60 > cc()["budget"]["hard_gpu_minutes"]:
            partial = True  # stop, keep what exists (protocol: preserve partial artifacts, no silent expansion)
            break
        r, co, ca = E.run_case(c)
        rep += r
        cont += co
        cand += ca
        if first is None:
            first = (r, co)
    leak_ok = False
    if not partial:  # cross-case leakage check: the first case again at the end
        r2, co2, _ = E.run_case(m["cases"][0])
        strip = lambda rows: [{k: v for k, v in x.items() if k != "seconds"} for x in rows]  # noqa: E731
        leak_ok = strip(r2) == strip(first[0]) and strip(co2) == strip(first[1])
    w1 = weights_hash(E.b.model)
    from reft.common import cfg
    T = cc()["tolerance"]
    bvt = [b["tf_beam_eq_abs_diff"] for c in cont for b in c["beams"] if "tf_beam_eq_abs_diff" in b]
    bvt_valid = [x for x in bvt if x is not None]
    finite = all(r["finite"] for r in rep)
    write_jsonl(out("replay_private.jsonl"), rep)
    write_jsonl(out("continuation_private.jsonl"), cont)
    write_jsonl(out("candidate_scores_private.jsonl"), cand)
    integ = {"schema": "reft_p2_1c_integrity_v1", "utc": now(), **fz, "slurm": slurm(), "cuda_libs": E.libs,
             "preflight": pf.name, "partial": partial, "validation": val, "validation_all_ok": all(v["ok"] for v in val.values()),
             "parity": f"{sum(sum(v['parity'].values()) for v in val.values())}/{sum(len(v['parity']) for v in val.values())}",
             "no_cross_case_leakage": leak_ok, "weights_unchanged": E.w0 == w1 == cfg()["model"]["expected_weights_hash"],
             "adapters_sha_ok": E.adapter_sha_ok, "all_next_token_finite": finite,
             "generated_beams_vs_tf_beam_eq": {"n": len(bvt), "n_scored": len(bvt_valid),
                                               "max_abs": max(bvt_valid) if bvt_valid else None,
                                               "n_over_tolerance": sum(x > T["beam_score_vs_teacher_forced"] for x in bvt_valid)},
             "counts": {"generate_calls": E.n_gen, "generated_steps": E.n_gen_steps, "tf_forwards": E.n_tf,
                        "replay_rows": len(rep), "continuations": len(cont), "candidates": len(cand)},
             "seconds": {"validation": t_val, "total": time.time() - t0},
             "outputs_sha256": {f: sha(out(f)) for f in ("replay_private.jsonl", "continuation_private.jsonl",
                                                          "candidate_scores_private.jsonl")}}
    integ["all_ok"] = bool(not partial and integ["validation_all_ok"] and leak_ok and integ["weights_unchanged"] and
                           all(E.adapter_sha_ok.values()) and finite)
    write_json(out("integrity.json"), integ)
    print(json.dumps({k: v for k, v in integ.items() if k != "validation"}, indent=1, default=str))


# ====================================================================== analysis (CPU, reference-aware)

def run_analyze():
    from transformers import WhisperTokenizer
    from reft.common import cfg, fillers
    from reft.evaluate_supervised import dev_refs, p21
    from reft.p2_1b_prefix_audit import script
    from reft.text import loops, one, tok_bytes_fn
    fz = check_freeze()
    if out("analysis.json").exists():
        raise SystemExit("analysis.json exists; write-once")
    C, R = cc(), cc()["rules"]
    tok = WhisperTokenizer.from_pretrained(cfg()["model"]["path"], local_files_only=True)
    tb, eos = tok_bytes_fn(tok), tok.eos_token_id
    loop_units = p21()["evaluation"]["loop_units"]
    A = {c["key"]: c for c in read_json(out("anchors_private.json"))["cases"]}
    MI = {c["key"]: c for c in read_json(out("anchors_inference.json"))["cases"]}
    integ = read_json(out("integrity.json"))
    REP, CONT, CAND = (read_jsonl(out(f)) for f in ("replay_private.jsonl", "continuation_private.jsonl",
                                                     "candidate_scores_private.jsonl"))
    present = {c["case"] for c in CONT}
    missing = sorted(set(A) - present)          # only after a partial (budget-stopped) run; integrity is then not ok
    A = {k: v for k, v in A.items() if k in present}
    refs = dev_refs(sorted(c["uid"] for c in A.values()))
    O = load_outputs()
    text = lambda ids: tok.decode([t for t in ids if t != eos], skip_special_tokens=True).strip()  # noqa: E731
    from reft.text import tokenize
    decode_ok = all(tokenize(text(O[n][c["uid"]]["tokens"])) == tokenize(O[n][c["uid"]]["text"]) for c in A.values() for n in SYSTEMS)

    def ev(key, tokens, scope, region_start):
        c = A[key]
        o = one(refs[c["uid"]], text(tokens))
        ob = one(refs[c["uid"]], O["B0"][c["uid"]]["text"])
        ret = float(np.mean([o["by"][r][0] == "match" for r in scope])) if scope else None
        new_err = sum(ob["by"][r][0] == "match" and o["by"][r][0] != "match" for r in range(region_start, len(o["ref"])))
        reg = c["en_region"]
        en_ok = sum(o["by"][r][0] == "match" for r in range(reg[0], reg[1] + 1)) if reg else None
        return {"retention": ret, "status": status_of(ret, R), "new_errors_generated_region": new_err,
                "identical_to_b0": o["hyp"] == ob["hyp"], "loop": loops(o["hyp"]) >= loop_units, "n_units": len(o["hyp"]),
                "en_region_correct": en_ok}

    def scope_of(key, anchor):
        c = A[key]
        if anchor == "early_zh":
            return c["b0_anchors"]["early_zh"]["scope_units"], c["b0_anchors"]["early_zh"]["scope_units"][0]
        start = {"start": 0, "pre_en": (c["en_region"] or [c["switch_unit"]])[0]}.get(anchor, c["switch_unit"])
        return c["span_units"], start

    # ---- originals and continuations
    orig = {(k, n): ev(k, O[n][A[k]["uid"]]["tokens"], *scope_of(k, "post_en")) for k in A for n in SYSTEMS}
    for c in CONT:
        key, src = c["case"], c["prefix_source"]
        toks = O[src][A[key]["uid"]]["tokens"][:c["k"]] + c["beams"][0]["suffix"]
        scope, start = scope_of(key, c["anchor"])
        c["eval"] = ev(key, toks, scope, start)
        b0_after = A[key]["b0_n_units"] - (A[key]["b0_anchors"][c["anchor"]]["units"] if c["anchor"] != "own" else
                                           A[key]["b0_anchors"]["post_en"]["units"])
        gen_units = c["eval"]["n_units"] - (A[key]["b0_anchors"][c["anchor"]]["units"] if c["anchor"] != "own" else
                                            A[key]["own_anchors"][src]["units"])
        c["eval"]["terminated"] = c["beams"][0]["terminated"]
        c["eval"]["premature_eos"] = c["beams"][0]["terminated"] and gen_units < R["premature_eos_fraction"] * b0_after
        c["eval"]["n_new_tokens"] = len(c["beams"][0]["suffix"])
    cont = {(c["case"], c["anchor"], c["prefix_source"], c["model"]): c for c in CONT}
    rep = {(r["case"], r["anchor"], r["prefix_source"], r["model"]): r for r in REP}
    # ---- candidates (T5)
    for x in CAND:
        key = x["case"]
        toks = O["B0"][A[key]["uid"]]["tokens"][:x["k"]] + x["tokens"]
        scope, start = scope_of(key, x["anchor"])
        e = ev(key, toks, scope, start)
        x["retention"], x["status"] = e["retention"], e["status"]
    pools = collections.defaultdict(list)
    for x in CAND:
        pools[(x["case"], x["anchor"])].append(x)

    def t5(key, anchor, m):
        cs = pools.get((key, anchor), [])
        q, margin = seq_preference(cs, m)
        b0c = next((x for x in cs if "B0_original" in x["sources"]), None)
        sc = sorted((x["scores"][m]["beam_eq"] for x in cs if x["scores"][m]["beam_eq"] is not None), reverse=True)
        best_tf = {}
        for x in cs:
            if x["status"] in ("recovered", "dropped"):
                best_tf[x["status"]] = max(best_tf.get(x["status"], -1e30), x["scores"][m]["sum"])
        return {"Q_omission_preferred": q, "beam_eq_margin_retain_minus_omit": margin,
                "n_cands": len(cs), "n_retain": sum(x["status"] == "recovered" for x in cs),
                "n_omit": sum(x["status"] == "dropped" for x in cs),
                "b0_cont_rank_beam_eq": (sc.index(b0c["scores"][m]["beam_eq"]) + 1) if b0c and b0c["scores"][m]["beam_eq"] is not None else None,
                "tf_sum_omit_preferred": (best_tf["dropped"] > best_tf["recovered"]) if len(best_tf) == 2 else None}

    # ---- case x system rows (T2/T3/T4/T5 + flags)
    rows = []
    for key, c in A.items():
        for s in ADAPTED:
            post_valid = c["b0_anchors"]["post_en"]["valid"]
            ow = c["own_anchors"][s]
            ident = bool(ow.get("identical_prefix_tokens"))
            B = cont[(key, "post_en", "B0", s)]["eval"]["status"] if post_valid else None
            O_ = cont[(key, "own", s, s)]["eval"]["status"] if ow.get("k") is not None else None
            b0_own = cont[(key, "own", s, "B0")]["eval"]["status"] if ow.get("k") is not None else None
            cp_own = cont[(key, "own", s, C["counterpart"][s])]["eval"]["status"] if ow.get("k") is not None else None
            rS, r0 = rep[(key, "post_en", "B0", s)]["src_next"], rep[(key, "post_en", "B0", "B0")]["src_next"]
            L = local_disfavor(rS["rank"], rS["logp"], r0["logp"], R) if post_valid else None
            T5 = t5(key, "post_en", s)
            od = orig[(key, s)]["status"] == "dropped"
            rescue = {a: cont[(key, a, "B0", s)]["eval"]["status"] for a in B0_ANCHORS if (key, a, "B0", s) in cont}
            valid_rescue = [a for a in ("pre_en", "post_en", "early_zh") if c["b0_anchors"][a]["valid"]]
            boundary = next((a for a in ["start"] + valid_rescue if rescue.get(a) == "recovered"), None)
            fl = flags_for(od, B, O_, bool(ow.get("valid")), ident, L, T5["Q_omission_preferred"]) if post_valid else []
            dv = c["divergence_vs_b0"][s]
            rows.append({
                "case": key, "kind": c["kind"], "group": c["group"], "system": s, "arm": s[:2], "seed": int(s[2:]),
                "orig_status": orig[(key, s)]["status"], "orig_retention": orig[(key, s)]["retention"], "orig_drop": od,
                "B_b0_prefix_status": B, "B_retention": cont[(key, "post_en", "B0", s)]["eval"]["retention"],
                "B_premature_eos": cont[(key, "post_en", "B0", s)]["eval"]["premature_eos"],
                "B_identical_to_b0": cont[(key, "post_en", "B0", s)]["eval"]["identical_to_b0"],
                "B_new_errors": cont[(key, "post_en", "B0", s)]["eval"]["new_errors_generated_region"],
                "O_own_prefix_status": O_, "own_valid": bool(ow.get("valid")), "own_common_location": ow.get("common_location"),
                "own_context_lexically_matched": ow.get("context_lexically_matched"), "own_identical_prefix_tokens": ident,
                "B0_at_own_prefix_status": b0_own, "counterpart_at_own_prefix_status": cp_own,
                "local": {"logp_S": rS["logp"], "logp_B0": r0["logp"], "delta": rS["logp"] - r0["logp"], "rank_S": rS["rank"],
                          "margin_S": rS["margin_best_other"], "margin_B0": r0["margin_best_other"],
                          "eos_logp_S": rep[(key, "post_en", "B0", s)]["eos_logp"], "eos_logp_B0": rep[(key, "post_en", "B0", "B0")]["eos_logp"],
                          "top1_script_S": script(rep[(key, "post_en", "B0", s)]["top5"][0][0], tb, eos)},
                "L_local_disfavor": L, "T5": T5, "T5_pre_en": t5(key, "pre_en", s),
                "rescue_by_anchor": rescue, "rescue_boundary": boundary,
                "own_b0_switch_token": rep[(key, "own", s, s)].get("b0_switch") if (key, "own", s, s) in rep else None,
                "divergence": {**dv, "switch_b0_units": c["b0_anchors"]["post_en"]["units"],
                               "lexical_before_switch": dv["lexical_unit"] is not None and dv["lexical_unit"] < c["b0_anchors"]["post_en"]["units"],
                               "token_before_switch": dv["token_div_b0_units"] is not None and dv["token_div_b0_units"] < c["b0_anchors"]["post_en"]["units"]},
                "en_region_correct": {"B0": orig[(key, "B0")]["en_region_correct"], "S": orig[(key, s)]["en_region_correct"],
                                      "n": (c["en_region"][1] - c["en_region"][0] + 1) if c["en_region"] else None},
                "flags": fl, "label": "+".join(fl) if fl else ("UNRESOLVED" if od else "NO_ORIGINAL_OMISSION")})
    dmg = [r for r in rows if r["kind"] == "damaged"]
    cells = aggregate_cells(dmg, R)
    n_valid = sum(A[k]["b0_anchors"]["post_en"]["valid"] for k in A if A[k]["kind"] == "damaged")
    dec = decide(bool(integ["all_ok"]) and decode_ok, n_valid, cells, R)
    b0rows = {key: {"local_b0": rep[(key, "post_en", "B0", "B0")]["src_next"], "b0_cont": cont[(key, "post_en", "B0", "B0")]["eval"],
                    "T5_b0": t5(key, "post_en", "B0")} for key in A}
    # ---- T6 groups
    def grp(g, arm=None):
        rr = [r for r in rows if r["group"] == g and (arm is None or r["arm"] == arm)]
        f = lambda pred: f"{sum(pred(r) for r in rr)}/{len(rr)}"  # noqa: E731
        return {"n": len(rr), "orig_dropped": f(lambda r: r["orig_status"] == "dropped"),
                "b0_prefix_recovered": f(lambda r: r["B_b0_prefix_status"] == "recovered"),
                "own_prefix_recovered": f(lambda r: r["O_own_prefix_status"] == "recovered"),
                "b0_model_at_own_prefix_recovered": f(lambda r: r["B0_at_own_prefix_status"] == "recovered"),
                "local_disfavor": f(lambda r: bool(r["L_local_disfavor"])),
                "mean_local_delta": float(np.mean([r["local"]["delta"] for r in rr])) if rr else None,
                "omission_preferred_T5": f(lambda r: r["T5"]["Q_omission_preferred"] is True),
                "T5_defined": f(lambda r: r["T5"]["Q_omission_preferred"] is not None),
                "lexical_div_before_switch": f(lambda r: r["divergence"]["lexical_before_switch"]),
                "token_div_before_switch": f(lambda r: r["divergence"]["token_before_switch"]),
                "own_context_lexically_matched": f(lambda r: bool(r["own_context_lexically_matched"])),
                "rescue_boundary": dict(collections.Counter(str(r["rescue_boundary"]) for r in rr)),
                "labels": dict(collections.Counter(r["label"] for r in rr))}
    T6 = {g: {"all": grp(g), "CE": grp(g, "CE"), "KL": grp(g, "KL")} for g in ("recovered", "persistent", "safe")}
    pub = {"schema": "reft_p2_1c_analysis_v1", "utc": now(), **fz, "slurm": slurm(), "decision": dec,
           "integrity_ok": integ["all_ok"], "decode_text_reproduced": decode_ok, "n_valid_damaged_post_en": n_valid,
           "cases_missing": missing,
           "inventory": {k: {"kind": A[k]["kind"], "group": A[k]["group"], "span_len": len(A[k]["span_units"]),
                             "b0_span_retained": A[k]["b0_span_retained"],
                             "anchors_valid": {a: A[k]["b0_anchors"][a]["valid"] for a in B0_ANCHORS},
                             "anchors_exact": {a: A[k]["b0_anchors"][a].get("exact") for a in B0_ANCHORS},
                             "own_valid": {s: A[k]["own_anchors"][s]["valid"] for s in ADAPTED}} for k in sorted(A)},
           "rows": [{k: v for k, v in r.items() if k not in ("own_b0_switch_token",)} for r in rows],
           "b0_reference_rows": b0rows, "cells": cells, "T6": T6,
           "continuation_summary": {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": {x: c["eval"][x] for x in ("status", "retention", "premature_eos", "loop", "terminated", "identical_to_b0", "new_errors_generated_region", "en_region_correct")}
                                    for k, c in cont.items()},
           "orig": {f"{k}|{n}": orig[(k, n)] for k in A for n in SYSTEMS},
           "inputs_sha256": {f: sha(out(f)) for f in ("anchors_private.json", "anchors_inference.json", "integrity.json",
                                                       "replay_private.jsonl", "continuation_private.jsonl",
                                                       "candidate_scores_private.jsonl")}}
    write_json(out("case_analysis_private.json"), {"utc": now(), "rows": rows,
                                                   "candidates": [{k: v for k, v in x.items() if k != "tokens"} | {"n_tokens": len(x["tokens"])} for x in CAND]})
    write_json(out("analysis.json"), pub)
    figures(pub)
    write_json(out("manifest.json"), {"utc": now(), **fz, "files": {p.name: sha(p) for p in sorted(out().glob("*.json*"))
                                                                     if p.name != "manifest.json"},
                                      "figures": {p.name: sha(p) for p in sorted(out("figures").glob("*.png"))}})
    print(json.dumps({"decision": dec, "cells": cells, "n_valid": n_valid, "integrity_ok": integ["all_ok"],
                      "decode_ok": decode_ok}, indent=1))


def figures(pub):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = out("figures")
    fd.mkdir(parents=True, exist_ok=True)
    rows = [r for r in pub["rows"] if r["kind"] == "damaged"]
    inv = pub["inventory"]
    keys = sorted({r["case"] for r in rows})
    sysn = list(ADAPTED)
    # F1 divergence timeline (B0 unit coordinates, relative to the switch-back anchor)
    fig, ax = plt.subplots(figsize=(9, 5))
    y = 0
    yt, yl = [], []
    for k in keys:
        for s in sysn:
            r = next(x for x in rows if x["case"] == k and x["system"] == s)
            d = r["divergence"]
            sw = d["switch_b0_units"]
            ax.plot([0, inv[k]["span_len"]], [y, y], color="#d62728" if r["orig_drop"] else "#2ca02c", lw=6, alpha=0.35)
            if d["token_div_b0_units"] is not None:
                ax.plot(d["token_div_b0_units"] - sw, y, "o", mfc="none", mec="k")
            if d["lexical_unit"] is not None:
                ax.plot(d["lexical_unit"] - sw, y, "x", color="#1f77b4")
            yt.append(y)
            yl.append(f"{k} {s}")
            y += 1
        y += 1
    ax.axvline(0, color="grey", ls="--")
    ax.set_yticks(yt, yl, fontsize=6)
    ax.set_xlabel("B0 canonical units relative to switch-back anchor (bar = span; red = originally dropped)")
    ax.set_title("F1 first token (o) and first lexical (x) divergence from B0")
    fig.tight_layout()
    fig.savefig(fd / "F1_divergence_timeline.png", dpi=130)
    plt.close(fig)
    # F2 forced-prefix recovery grid
    conds = [("orig", "original"), ("pre_en", "B0 pre-EN"), ("post_en", "B0 post-EN"), ("early_zh", "B0 early-ZH"), ("own", "own prefix")]
    val = {"recovered": 2, "partial": 1, "dropped": 0, None: np.nan}
    M = []
    lab = []
    for k in keys:
        for s in sysn:
            r = next(x for x in rows if x["case"] == k and x["system"] == s)
            M.append([val[r["orig_status"]], val[r["rescue_by_anchor"].get("pre_en")], val[r["B_b0_prefix_status"]],
                      val[r["rescue_by_anchor"].get("early_zh")], val[r["O_own_prefix_status"]]])
            lab.append(f"{k} {s}")
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(np.array(M, float), cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(conds)), [c[1] for c in conds], rotation=30, fontsize=7)
    ax.set_yticks(range(len(lab)), lab, fontsize=6)
    ax.set_title("F2 span status (green recovered, yellow partial, red dropped)")
    fig.tight_layout()
    fig.savefig(fd / "F2_forced_prefix_recovery.png", dpi=130)
    plt.close(fig)
    # F3 local delta vs sequence-level margin at the B0 post-EN prefix
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    allr = pub["rows"]
    for arm, col in (("CE", "#d62728"), ("KL", "#1f77b4")):
        rr = [r for r in allr if r["arm"] == arm]
        axs[0].scatter([r["local"]["delta"] for r in rr], [{"recovered": 0, "persistent": 1, "safe": 2}[r["group"]] + (0.15 if arm == "KL" else -0.15) for r in rr],
                       c=col, label=arm, s=18)
        rr = [r for r in rr if r["T5"]["beam_eq_margin_retain_minus_omit"] is not None]
        axs[1].scatter([r["T5"]["beam_eq_margin_retain_minus_omit"] for r in rr], [r["local"]["delta"] for r in rr], c=col, label=arm, s=18)
    axs[0].axvline(0, color="grey", ls="--")
    axs[0].set_yticks([0, 1, 2], ["KL-recovered", "persistent", "safe"])
    axs[0].set_xlabel("log p_S - log p_B0 of B0's switch-back token (nats)")
    axs[1].axvline(0, color="grey", ls="--")
    axs[1].axhline(0, color="grey", ls="--")
    axs[1].set_xlabel("beam-eq score: best retaining - best omitting candidate")
    axs[1].set_ylabel("local delta (nats)")
    axs[0].legend(fontsize=7)
    fig.suptitle("F3 local vs sequence-level preference at the B0 post-English prefix")
    fig.tight_layout()
    fig.savefig(fd / "F3_local_vs_sequence.png", dpi=130)
    plt.close(fig)
    # F4 recovered vs persistent compact per-case panel
    fig, ax = plt.subplots(figsize=(9, 3.5))
    metrics = [("orig_drop", "orig dropped"), ("B_b0_prefix_status", "B0-prefix recovered"), ("O_own_prefix_status", "own-prefix recovered"),
               ("B0_at_own_prefix_status", "B0 at own prefix recovered"), ("L_local_disfavor", "local disfavor"),
               ("T5", "T5 omission preferred")]
    M = []
    cols = []
    for k in keys:
        for arm in ("CE", "KL"):
            rr = [r for r in rows if r["case"] == k and r["arm"] == arm]
            colv = []
            for m_, _ in metrics:
                if m_ == "orig_drop":
                    colv.append(np.mean([r["orig_drop"] for r in rr]))
                elif m_ == "L_local_disfavor":
                    colv.append(np.mean([bool(r["L_local_disfavor"]) for r in rr]))
                elif m_ == "T5":
                    colv.append(np.mean([r["T5"]["Q_omission_preferred"] is True for r in rr]))
                else:
                    colv.append(np.mean([r[m_] == "recovered" for r in rr]))
            M.append(colv)
            cols.append(f"{k}/{arm}\n{inv[k]['group'][:4]}")
    ax.imshow(np.array(M).T, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, fontsize=6)
    ax.set_yticks(range(len(metrics)), [m[1] for m in metrics], fontsize=7)
    ax.set_title("F4 fraction of seeds per (case, arm)")
    fig.tight_layout()
    fig.savefig(fd / "F4_recovered_vs_persistent.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "anchors", "smoke", "replay", "analyze"])
    a = ap.parse_args(argv)
    {"freeze": run_freeze, "anchors": run_anchors, "smoke": run_smoke, "replay": run_replay, "analyze": run_analyze}[a.cmd]()


if __name__ == "__main__":
    main()
