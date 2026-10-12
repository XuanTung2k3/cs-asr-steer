#!/usr/bin/env python
"""ABTS-0 K1 evaluator v2 (reference side; offline diagnostic only). Never imported by the runner.

Additive amendments A1-A3 approved in docs/research_review/ABTS0_K1_REPORT.md section 10 (2026-10-12). The v1 evaluator
(experiments/abts0_k1_evaluate.py) is unchanged and is also run, so both results sit side by side.

groups    -- pre-capture: per-utterance group (MM/MIX/END) + dialogue from the references only -> groups_prerun.json.
             Read later by the reference-free pre-score gate (experiments/abts0_k1_gate.py). Reads NO K1 output.
evaluate  -- requires a PASS pre-score gate, a valid amendment seal and a valid output seal.

A1  timing-free lexical label. A chosen pair's B0 Han block H is mapped to canonical B0 units; the frozen Levenshtein
    alignment (csasr.evaluation.mer.align_tokens, the one that defines "reachable") and every co-optimal alignment decide
    which reference unit the block spells. Repeated words, co-optimal reassignment and cross-run blocks are AMBIGUOUS,
    and AMBIGUOUS counts as incorrect in primary precision (optimistic bound reported separately). No timing is used.
A2  onset recall: hits on the FIRST unit of a reachable run / reachable (39 in MM); also / genuine heard runs (186).
A3  precedence BLOCKED > INSUFFICIENT(STRUCTURAL|REACHABLE|COVERAGE|CLASSES) > FAIL > PASS, numeric GO unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, digest
import experiments.abts0_k1_evaluate as V1

BASE = V1.BASE
GROUPS = f"{BASE}/groups_prerun.json"
SCHEMA = "abts0_k1_evaluation_v2"
LITERAL = "LITERAL"
AMBIGUOUS = ("AMBIGUOUS_EMPTY", "AMBIGUOUS_UNMAPPED", "AMBIGUOUS_NO_ALIGNMENT", "AMBIGUOUS_REPEAT", "AMBIGUOUS_ALIGNMENT",
             "AMBIGUOUS_RUN", "AMBIGUOUS_BOUNDARY")
NONLITERAL = ("WRONG_OCCURRENCE", "WRONG_ENGLISH", "EN_ON_MANDARIN_SPEECH")
SENSITIVITY_TOL = 6400          # 400 ms: descriptive association sensitivity only; runtime stays at 3200
PRE_ASSOCIATION = ("EN_UNTIMED", "BEYOND_HEARD", "ZERO_DURATION", "PRESENT_IN_B0", "NO_HAN_OVERLAP", "NONCONTIGUOUS_BLOCK",
                   "LOW_HAN_COVER", "ONSET_MISMATCH", "MID_CHARACTER_ONSET", "CONTENT_T0")      # = v1's U1 exclusions


# ---- A1: alignment geometry ---------------------------------------------------------------------------------------

def _dp(ref, hyp) -> list[list[int]]:
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    return d


def cooptimal(ref, hyp) -> tuple[set, set]:
    """Exact union over ALL minimum-cost alignments (unit-cost Levenshtein, as align_tokens):
    diagonal pairs (i, j) that some optimal alignment matches/substitutes, and ref indices some optimal alignment deletes."""
    n, m = len(ref), len(hyp)
    f = _dp(ref, hyp)
    rb = _dp(ref[::-1], hyp[::-1])                       # rb[n-i][m-j] = distance(ref[i:], hyp[j:])
    D = f[n][m]
    diag = {(i, j) for i in range(n) for j in range(m)
            if f[i][j] + (ref[i] != hyp[j]) + rb[n - i - 1][m - j - 1] == D}
    dele = {i for i in range(n) for j in range(m + 1) if f[i][j] + 1 + rb[n - i - 1][m - j] == D}
    return diag, dele


def map_units(k1_surfaces: list[str], canon: list[str]) -> list[int | None]:
    """K1 content units (raw B0 text) -> canonical B0 unit index, by exact monotone surface match after the frozen
    normalization of each unit; a unit that normalizes to != 1 canonical unit or is not MATCHed maps to None."""
    from csasr.evaluation.mer import MATCH, align_tokens
    from csasr.evaluation.normalization import normalize_text, segment_units
    flat, origin = [], []
    for k, s in enumerate(k1_surfaces):
        cu = [u.surface for u in segment_units(normalize_text(s))]
        flat += cu
        origin += [k if len(cu) == 1 else None] * len(cu)
    out: list[int | None] = [None] * len(k1_surfaces)
    for o in align_tokens(flat, canon):
        if o.op == MATCH and origin[o.ref_idx] is not None:
            out[origin[o.ref_idx]] = o.hyp_idx
    return out


def run_index(is_en: list[bool]) -> list[int | None]:
    """Maximal EN run id per reference unit (None for non-EN), the same runs as V1.runs_registry."""
    out, r, prev = [], -1, False
    for e in is_en:
        if e and not prev:
            r += 1
        out.append(r if e else None)
        prev = e
    return out


def lexical_label(rt: list[str], is_en: list[bool], hyp: list[str], H: list[int | None], cu: list[str]) -> dict:
    """A1 label of one candidate (canonical units cu) at B0 block H (canonical hypothesis indices)."""
    from csasr.evaluation.mer import MATCH, SUB, align_tokens
    rec = {"label": None, "ref_index": None, "onset": False, "block_b0_correct_mandarin": 0, "r_frozen": [], "r_any": []}
    ops = align_tokens(rt, hyp)
    frozen = {(o.ref_idx, o.hyp_idx): o.op for o in ops if o.op in (MATCH, SUB)}
    Hs = set(h for h in H if h is not None)
    rec["block_b0_correct_mandarin"] = sum(1 for (i, j), op in frozen.items() if j in Hs and op == MATCH and not is_en[i])
    if not cu:
        rec["label"] = "AMBIGUOUS_EMPTY"
        return rec
    if not H or None in H:
        rec["label"] = "AMBIGUOUS_UNMAPPED"
        return rec
    diag, dele = cooptimal(rt, hyp)
    r_frozen = sorted(i for (i, j) in frozen if j in Hs)
    r_any = sorted({i for (i, j) in diag if j in Hs})
    rec.update(r_frozen=r_frozen, r_any=r_any)
    if not r_any:
        rec["label"] = "AMBIGUOUS_NO_ALIGNMENT"
        return rec
    L = len(cu)
    occ = [k for k in range(len(rt) - L + 1) if rt[k:k + L] == cu and all(is_en[k:k + L])]
    M = [k for k in occ if k in r_any]
    if M:
        i = M[0]
        rec["ref_index"] = i
        runs = run_index(is_en)
        first_en = next(k for k in r_any if is_en[k])
        if len(M) > 1:
            rec["label"] = "AMBIGUOUS_REPEAT"                    # the same word twice inside the alignable window
        elif i not in r_frozen or not {j for (a, j) in diag if a == i} <= Hs or i in dele:
            rec["label"] = "AMBIGUOUS_ALIGNMENT"                 # some optimal alignment assigns it elsewhere / deletes it
        elif runs[first_en] != runs[i]:
            rec["label"] = "AMBIGUOUS_RUN"                       # block starts on a different English run
        else:
            rec["label"] = LITERAL
            rec["onset"] = i == 0 or runs[i - 1] != runs[i]
        return rec
    if any(r_any[0] - 1 <= k <= r_any[-1] + 1 for k in occ):
        rec["label"] = "AMBIGUOUS_BOUNDARY"
    elif occ:
        rec["label"] = "WRONG_OCCURRENCE"
    elif any(is_en[k] for k in r_any):
        rec["label"] = "WRONG_ENGLISH"
    else:
        rec["label"] = "EN_ON_MANDARIN_SPEECH"
    return rec


def canonical_units(text: str) -> list[str]:
    from csasr.evaluation.normalization import normalize_text, segment_units
    return [u.surface for u in segment_units(normalize_text(text))]


def label_pair(r: dict, st: list, hyp: list[str], b0_units: list[dict], unit_map: list, p: dict) -> dict:
    """A1 label + genuine/artifact/safety fields for one proposal carrying a B0 block."""
    k0, k1 = p["block"]
    H = [unit_map[k] for k in range(k0, k1 + 1)]
    if any(b0_units[k]["kind"] != "HAN" for k in range(k0, k1 + 1)):
        H = [None]
    lab = lexical_label(r["tokens"], r["is_en"], hyp, H, canonical_units(p["surface"]))
    i = lab["ref_index"]
    u = st[i] if (i is not None and lab["label"] == LITERAL) else None
    lab.update(genuine=bool(u and u["genuine"]), artifact=u["artifact"] if u else None,
               b0_category=u["category"] if u else None,
               over_correct_mandarin=lab["label"] != LITERAL and lab["block_b0_correct_mandarin"] > 0)
    return lab


# ---- A3 -----------------------------------------------------------------------------------------------------------

def decide_v2(m: dict, go: dict, *, blocked: list[str], gate_status: str) -> tuple[str, list[str]]:
    if blocked:
        return "K1_BLOCKED", blocked
    why = []
    if gate_status != "PASS":
        why.append("STRUCTURAL")
    if m["reachable_genuine_runs"] < 12 or m["reachable_dialogues"] < 6:
        why.append("REACHABLE")
    if m["scored"] < go["accepted_min"] or m["scored_dialogues"] < go["accepted_dialogues_min"]:
        why.append("COVERAGE")
    if m["auroc_dialogues"] < go["auroc_dialogues_min"]:
        why.append("CLASSES")
    if why:
        return "K1_INSUFFICIENT", why
    checks = {"accepted": m["accepted"] >= go["accepted_min"],
              "accepted_dialogues": m["accepted_dialogues"] >= go["accepted_dialogues_min"],
              "precision": m["precision"] is not None and m["precision"] >= go["precision_min"],
              "genuine_words": m["genuine_recoverable_words"] >= go["genuine_recoverable_words_min"],
              "genuine_dialogues": m["genuine_recoverable_dialogues"] >= go["genuine_recoverable_dialogues_min"],
              "onset_recall": m["onset_recall"] is not None and m["onset_recall"] >= go["reachable_recall_min"],
              "auroc": m["auroc_macro"] is not None and m["auroc_macro"] >= go["auroc_min"]}
    failed = [k for k, ok in checks.items() if not ok]
    return ("K1_FAIL", failed) if failed else ("K1_PASS", [])


# ---- groups (pre-capture) -----------------------------------------------------------------------------------------

def group_of(reference: str) -> tuple[str, float]:
    """V1.reference_context's frozen group rule (canonical EN fraction)."""
    from csasr.evaluation.normalization import EN, normalize_text, segment_units, tag_unit
    ru = segment_units(normalize_text(reference))
    frac = sum(tag_unit(u) == EN for u in ru) / max(1, len(ru))
    return ("MM" if frac < 0.2 else "MIX" if frac < 0.5 else "END"), frac


def cmd_groups(args) -> None:
    from experiments.inference_cf_p2_evaluate import load_references
    plan = json.loads((ROOT / BASE / "plan_sealed.json").read_text())
    if (ROOT / BASE / "run1/capture").exists() and any((ROOT / BASE / "run1/capture").iterdir()):
        raise RuntimeError("groups must be frozen before any K1 capture output exists")
    refs = load_references()
    rows = {}
    for r in plan["rows"]:
        g, _ = group_of(refs[r["utterance_id"]]["reference"])
        if refs[r["utterance_id"]]["dialogue_id"] != r["dialogue_id"]:
            raise ValueError("dialogue id mismatch")
        rows[r["utterance_id"]] = {"group": g, "dialogue_id": r["dialogue_id"]}
    out = {"schema": "abts0_k1_groups_v1", "plan_hash": plan["plan_hash"], "k1_outputs_read": False,
           "rule": "canonical reference EN fraction: MM < 0.2 <= MIX < 0.5 <= END (evaluator.groups)",
           "counts": dict(Counter(v["group"] for v in rows.values())), "rows": rows}
    out["groups_hash"] = digest(out)
    atomic_json(ROOT / GROUPS, out)
    print(json.dumps(out["counts"]))


# ---- evaluate -----------------------------------------------------------------------------------------------------

def cmd_evaluate(args) -> None:
    from transformers import WhisperTokenizer
    from experiments.abts0_k1_gate import verify_amendment_seal, verify_gate
    amend = verify_amendment_seal()
    gate = verify_gate()
    seal = V1.verify_seal()
    V1.cmd_evaluate(args)                                   # v1 result, unchanged, for comparison (evaluation.json)
    v1 = json.loads((ROOT / BASE / "evaluation.json").read_text())
    c = V1.cfg()
    go = c["evaluator"]["go_MM"]
    plan = json.loads((ROOT / BASE / "plan_sealed.json").read_text())
    sel = json.loads((ROOT / BASE / "run1/selection.json").read_text())
    groups = json.loads((ROOT / GROUPS).read_text())["rows"]
    reg = json.loads((ROOT / BASE / "registry_prerun.json").read_text())["registry"]
    tok = WhisperTokenizer.from_pretrained(plan["model_dir"], local_files_only=True)
    rc = V1.reference_context(plan["rows"])
    blocked = []
    G = {g: Counter() for g in ("MM", "MIX", "END")}
    per_d = {g: defaultdict(Counter) for g in G}
    pos, neg = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    onset_hits, run_hits = {g: set() for g in G}, {g: set() for g in G}
    words, word_d = {g: set() for g in G}, {g: set() for g in G}
    examples, b0_equal = [], 0
    for row, pr in zip(sel["rows"], plan["rows"]):
        uid = row["utterance_id"]
        r = rc[uid]
        g, dlg = r["group"], r["dialogue_id"]
        if groups[uid]["group"] != g:
            blocked.append(f"group mismatch {uid}")
        cap = json.loads((ROOT / BASE / f"run1/capture/{pr['index']:03d}.json").read_text())
        b0_equal += int(cap["B0"]["equals_B0M"])
        hyp = canonical_units(tok.decode(cap["B0"]["tokens"], skip_special_tokens=True))
        st = V1.unit_status(r, tok.decode(cap["B0"]["tokens"], skip_special_tokens=True))
        runs = V1.runs_registry(r, st)
        run_of = {k: n for n, run in enumerate(runs) for k in run["units"]}
        for run in runs:
            if run["reachable"]:
                G[g]["reachable_genuine_runs"] += 1
                per_d[g][dlg]["reachable"] += 1
            if run["genuine"] and run["heard"]:
                G[g]["genuine_heard_runs"] += 1
        umap = map_units([u["surface"] for u in row["b0_units"]], hyp)
        lab_of = lambda p: label_pair(r, st, hyp, row["b0_units"], umap, p)       # noqa: E731
        for p in row["proposals"]:
            G[g][f"status_{p['status']}"] += 1
            if p["status"] not in PRE_ASSOCIATION:
                G[g]["U1_n"] += 1
                G[g][f"U1_{lab_of(p)['label']}"] += 1
                if p["status"] == "ELIGIBLE":
                    G[g]["U2_n"] += 1
                    G[g][f"U2_{lab_of(p)['label']}"] += 1
                if p["status"] == "ONSET_MISMATCH" and abs(p["onset_diff"]) <= SENSITIVITY_TOL:
                    G[g]["sens400_associated"] += 1                                  # descriptive only
                    G[g][f"sens400_{lab_of(p)['label']}"] += 1
        if row["shuffled_chosen"] is not None:
            G[g][f"shuffled_{lab_of(row['shuffled_chosen'])['label']}"] += 1
            G[g]["shuffled_n"] += 1
        ch = row["chosen"]
        if ch is None:
            continue
        lab = lab_of(ch)
        G[g]["chosen"] += 1
        G[g][f"chosen_{lab['label']}"] += 1
        G[g]["chosen_encoding_valid"] += int(ch["encoding_valid"])
        G[g]["chosen_over_correct_mandarin"] += int(lab["over_correct_mandarin"])
        sp = ROOT / BASE / f"run1/scores/{pr['index']:03d}.json"
        sc = json.loads(sp.read_text()) if sp.exists() else {"status": "MISSING"}
        if sc["status"] in ("INVALID_ENGINE_IDENTITY", "MISSING") and ch["encoding_valid"]:
            blocked.append(f"{sc['status']} {uid}")
            continue
        if sc["status"] != "SCORED" or sc["rule"]["status"] == "NONFINITE":
            G[g][f"score_{sc['status'] if sc['status'] != 'SCORED' else 'NONFINITE'}"] += 1
            continue
        rule = sc["rule"]
        G[g]["scored"] += 1
        per_d[g][dlg]["scored"] += 1
        if lab["label"] == LITERAL:
            pos[g][dlg].append(rule["C"])
        elif lab["label"] in NONLITERAL:
            neg[g][dlg].append(rule["C"])
        if rule["off_only_accept"]:
            G[g]["offonly_accepted"] += 1
            G[g]["offonly_literal"] += int(lab["label"] == LITERAL)
        if not rule["accept"]:
            continue
        G[g]["accepted"] += 1
        G[g][f"accepted_{lab['label']}"] += 1
        G[g]["accepted_literal"] += int(lab["label"] == LITERAL)
        G[g]["accepted_ambiguous"] += int(lab["label"] in AMBIGUOUS)
        G[g]["accepted_over_correct_mandarin"] += int(lab["over_correct_mandarin"])
        G[g]["accepted_on_mandarin_speech"] += int(lab["label"] == "EN_ON_MANDARIN_SPEECH")
        per_d[g][dlg]["accepted"] += 1
        per_d[g][dlg]["accepted_literal"] += int(lab["label"] == LITERAL)
        if lab["label"] == LITERAL and lab["genuine"]:
            i = lab["ref_index"]
            words[g].add((uid, i))
            word_d[g].add(dlg)
            n = run_of.get(i)
            if n is not None and runs[n]["reachable"]:
                run_hits[g].add((uid, n))
            if n is not None and runs[n]["genuine"] and runs[n]["heard"] and runs[n]["units"][0] == i:
                onset_hits[g].add((uid, n, runs[n]["reachable"]))
        examples.append({"utterance_id": uid, "group": g, "c": ch["surface"], "b": ch["b_surface"], "t": ch["t"],
                         "label": lab["label"], "onset": lab["onset"], "genuine": lab["genuine"],
                         "over_correct_mandarin": lab["over_correct_mandarin"], "C": rule["C"],
                         "A_target_c": rule["A"]["target"]["c"], "accept": rule["accept"]})
    if b0_equal != len(plan["rows"]):
        blocked.append(f"B0 equals B0M only {b0_equal}/{len(plan['rows'])}")
    if G["MM"]["reachable_genuine_runs"] != reg["MM|reachable"]:
        blocked.append(f"MM reachable {G['MM']['reachable_genuine_runs']} != registry {reg['MM|reachable']}")
    metrics = {}
    for g in G:
        m = dict(G[g])
        acc, reach, gh = m.get("accepted", 0), m.get("reachable_genuine_runs", 0), m.get("genuine_heard_runs", 0)
        aucs = {d: V1.auroc(pos[g][d], neg[g][d]) for d in set(pos[g]) | set(neg[g])}
        aucs = {d: v for d, v in aucs.items() if v is not None}
        reach_onset = sum(1 for h in onset_hits[g] if h[2])
        m.update(accepted=acc, scored=m.get("scored", 0),
                 scored_dialogues=sum(1 for d in per_d[g].values() if d["scored"]),
                 accepted_dialogues=sum(1 for d in per_d[g].values() if d["accepted"]),
                 precision=(m.get("accepted_literal", 0) / acc) if acc else None,
                 precision_optimistic=((m.get("accepted_literal", 0) + m.get("accepted_ambiguous", 0)) / acc) if acc else None,
                 precision_chosen=(m.get(f"chosen_{LITERAL}", 0) / m["chosen"]) if m.get("chosen") else None,
                 precision_U1=(m.get(f"U1_{LITERAL}", 0) / m["U1_n"]) if m.get("U1_n") else None,
                 precision_U2=(m.get(f"U2_{LITERAL}", 0) / m["U2_n"]) if m.get("U2_n") else None,
                 precision_offonly=(m.get("offonly_literal", 0) / m["offonly_accepted"]) if m.get("offonly_accepted") else None,
                 precision_shuffled=(m.get(f"shuffled_{LITERAL}", 0) / m["shuffled_n"]) if m.get("shuffled_n") else None,
                 genuine_recoverable_words=len(words[g]), genuine_recoverable_dialogues=len(word_d[g]),
                 reachable_genuine_runs=reach, reachable_dialogues=sum(1 for d in per_d[g].values() if d["reachable"]),
                 onset_hits_reachable=reach_onset, onset_hits_genuine=len(onset_hits[g]),
                 onset_recall=(reach_onset / reach) if reach else None,
                 genuine_run_onset_coverage=(len(onset_hits[g]) / gh) if gh else None,
                 reachable_run_coverage_a1=(len(run_hits[g]) / reach) if reach else None,
                 reachable_run_coverage_v1=v1["metrics"][g].get("reachable_recall"),
                 auroc_by_dialogue=aucs, auroc_dialogues=len(aucs),
                 auroc_macro=float(np.mean(list(aucs.values()))) if aucs else None,
                 precision_ci_dialogue_boot=V1.boot_precision(
                     {d: (v["accepted_literal"], v["accepted"]) for d, v in per_d[g].items()}, sorted(per_d[g]),
                     c["evaluator"]["bootstrap"]["seed"], c["evaluator"]["bootstrap"]["reps"]),
                 by_dialogue={d: dict(v) for d, v in sorted(per_d[g].items())})
        metrics[g] = m
    label, reasons = decide_v2(metrics["MM"], go, blocked=blocked, gate_status=gate["status"])
    out = {"schema": SCHEMA, "seal_hash": seal["seal_hash"], "amendment_seal_hash": amend["seal_hash"],
           "gate_hash": gate["gate_hash"], "label": label, "reasons": reasons, "v1_label": v1["label"],
           "metrics": metrics, "blocked": blocked, "examples": examples,
           "notes": {"precision": "primary = LITERAL / accepted (AMBIGUOUS counted incorrect)",
                     "sens400": "ONSET_MISMATCH proposals with |onset_diff| <= 400 ms: descriptive, never scored or decisive"}}
    out["evaluation_hash"] = digest(out)
    atomic_json(ROOT / BASE / "evaluation_v2.json", out)
    print(label, reasons)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("groups", "evaluate"))
    args = ap.parse_args()
    cmd_groups(args) if args.stage == "groups" else cmd_evaluate(args)


if __name__ == "__main__":
    main()
