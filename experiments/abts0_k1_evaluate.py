#!/usr/bin/env python
"""ABTS-0 K1 evaluator (reference side; offline diagnostic only). Never imported by the runner.

registry  -- genuine-target registry and pre-run attainability ceiling. Uses only already-exposed historical artifacts
             (B0M_L16, historical forced-EN B1, R0 PRIMARY tracks) and references; reads NO K1 output.
evaluate  -- requires results/inference_cf/abts0_k1/output_seal.json to verify; labels every sealed proposal, computes
             the frozen K1 metrics and the K1 label (configs/inference_cf/abts0_k1.json "evaluator").

Reference timing is the R0 oracle MMS-FA CTC proxy (not ground truth); +-200 ms sensitivity is reported.
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

from csasr.inference_cf.core import atomic_json, digest, file_hash

CONFIG = "configs/inference_cf/abts0_k1.json"
R0_CONFIG = "configs/inference_cf/r0_region_vector.json"
BASE = "results/inference_cf/abts0_k1"
HAN_CATS = ("wrong_language_substitution", "phonetic_transliteration_or_script")
TOL = 3200


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


# ---- reference side -----------------------------------------------------------------------------------------------

def reference_context(plan_rows: list[dict]) -> dict:
    """Canonical reference units, CTC timing per canonical unit (exact-surface map to nat5h units), group, dialogue."""
    import pyarrow.parquet as pq
    from csasr.evaluation.mer import MATCH, align_tokens
    from csasr.evaluation.normalization import EN, normalize_text, segment_units, tag_unit
    from csasr.nat5h.units import build_reference_units
    from experiments.inference_cf_p2_evaluate import load_references
    from experiments.inference_cf_r0_analyze import oracle_tracks
    ids = [r["utterance_id"] for r in plan_rows]
    refs = load_references()
    r0c = json.loads((ROOT / R0_CONFIG).read_text())
    _, info = oracle_tracks([{"utterance_id": r["utterance_id"], "n_samples": r["n_samples"]} for r in plan_rows], r0c)
    panel = json.loads((ROOT / "docs/inference_cf/R0_PANEL.json").read_text())
    role = pq.read_table(panel["role_source"]["path"], filters=[("utterance_id", "in", ids)],
                         columns=["utterance_id", "transcript_raw"]).to_pylist()
    nat = {x["utterance_id"]: build_reference_units(x["utterance_id"], x["transcript_raw"]) for x in role}
    g = cfg()["evaluator"]
    out = {}
    for r in plan_rows:
        uid = r["utterance_id"]
        ru = segment_units(normalize_text(refs[uid]["reference"]))
        rt = [u.surface for u in ru]
        is_en = [tag_unit(u) == EN for u in ru]
        frac = sum(is_en) / max(1, len(rt))
        grp = "MM" if frac < 0.2 else ("MIX" if frac < 0.5 else "END")
        mp = {o.ref_idx: o.hyp_idx for o in align_tokens(rt, [u.surface for u in nat[uid]]) if o.op == MATCH}
        tm = {ui: (int(s), int(e)) for s, e, _, ui in info["units"].get(uid, [])}
        timing = [tm.get(mp.get(i)) if mp.get(i) is not None else None for i in range(len(rt))]
        out[uid] = {"tokens": rt, "is_en": is_en, "group": grp, "en_fraction": frac, "timing": timing,
                    "dialogue_id": refs[uid]["dialogue_id"], "heard": r["heard_samples"]}
    out["_meta"] = {"timing_stats": info["stats"], "fillers": g["artifact_taxonomy"]["filler"]}
    return out


def unit_status(rc: dict, hyp_text: str) -> list[dict]:
    """Per canonical EN reference unit: B0 category, artifact class, genuine flag, aligned hypothesis index."""
    from csasr.evaluation.mer import align_tokens
    from csasr.evaluation.normalization import normalize_text, segment_units
    from csasr.evaluation.pier import _categorize
    rt = rc["tokens"]
    hyp = [u.surface for u in segment_units(normalize_text(hyp_text))]
    ops = align_tokens(rt, hyp)
    hidx = {o.ref_idx: o.hyp_idx for o in ops if o.ref_idx is not None}
    fill = set(cfg()["evaluator"]["artifact_taxonomy"]["filler"])
    res = []
    for i, w in enumerate(rt):
        if not rc["is_en"][i]:
            res.append(None)
            continue
        ok, cat, hs = _categorize(rt, hyp, ops, i)
        single = len(w) == 1 and w.isalpha()
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(rt) and rc["is_en"][j] and len(rt[j]) == 1]
        art = None
        if not ok:
            if w in fill:
                art = "filler"
            elif single and nb:
                art = "acronym"
            elif cat == "boundary_error" or (hs and hs != w and w in hs and hs.isascii()):
                art = "boundary_join"
        res.append({"correct": ok, "category": cat, "hyp": hs, "artifact": art, "genuine": (not ok) and art is None,
                    "hyp_idx": hidx.get(i)})
    return res


def runs_registry(rc: dict, st: list) -> list[dict]:
    """Maximal reference EN runs with heard / genuine / rendering / reachable flags."""
    runs, i, n = [], 0, len(rc["tokens"])
    while i < n:
        if not rc["is_en"][i]:
            i += 1
            continue
        j = i
        while j + 1 < n and rc["is_en"][j + 1]:
            j += 1
        idx = list(range(i, j + 1))
        tm = rc["timing"][i]
        heard = tm is not None and tm[0] < rc["heard"]
        wrong = [k for k in idx if not st[k]["correct"]]
        genuine = any(st[k]["genuine"] for k in idx)
        on = st[i]
        rendering = ("HAN" if on["category"] in HAN_CATS else "OMITTED" if on["category"] in ("deletion", "insertion_near_poi")
                     else "WRONG_EN" if on["category"] == "same_language_substitution" else
                     "CORRECT_ONSET" if on["correct"] else on["category"].upper())
        reachable = genuine and heard and on["category"] in HAN_CATS and (on["hyp_idx"] or 0) > 0
        runs.append({"units": idx, "heard": heard, "wrong": bool(wrong), "genuine": genuine, "onset_rendering": rendering,
                     "reachable": bool(reachable), "timed": tm is not None})
        i = j + 1
    return runs


def label_candidate(rc: dict, st: list, surface: str, iv) -> dict:
    """Literal / nonliteral label of one forced-EN proposal at its EN-word interval (primary tol 0, sensitivity 200 ms)."""
    from csasr.evaluation.normalization import normalize_text, segment_units
    cu = [u.surface for u in segment_units(normalize_text(surface))]
    rt, tm = rc["tokens"], rc["timing"]
    if iv is None or not cu:
        return {"label": "UNLABELLED", "literal": False, "literal_pm200": False}
    s, e = int(iv[0]), int(iv[1])

    def hit(tol):
        for i in range(len(rt) - len(cu) + 1):
            if rt[i:i + len(cu)] == cu and all(rc["is_en"][i:i + len(cu)]) and tm[i] is not None \
                    and min(e + tol, tm[i + len(cu) - 1][1]) > max(s - tol, tm[i][0]):
                return i
        return None
    i0, i2 = hit(0), hit(TOL)
    rec = {"literal": i0 is not None, "literal_pm200": i2 is not None, "ref_index": i0}
    if i0 is not None:
        u = st[i0]
        rec.update(label="LITERAL", b0_category=u["category"], genuine=u["genuine"], artifact=u["artifact"],
                   b0_correct=u["correct"], b0_rendering="HAN" if u["category"] in HAN_CATS else u["category"])
        return rec
    over = [k for k in range(len(rt)) if tm[k] is not None and min(e, tm[k][1]) > max(s, tm[k][0])]
    if any(rc["is_en"][k] for k in over):
        lab = "WRONG_SPAN" if any(rt[k:k + len(cu)] == cu for k in range(len(rt))) else "WRONG_ENGLISH"
    elif over:
        lab = "EN_FROM_ZH_SPEECH"
    else:
        lab = "UNTIMED_SPAN"
    rec["label"] = lab
    return rec


def auroc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    w = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return w / (len(pos) * len(neg))


def boot_precision(by_d: dict, keys: list[str], seed: int, reps: int) -> list[float] | None:
    if not keys:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        ks = rng.choice(keys, size=len(keys), replace=True)
        n = sum(by_d[k][1] for k in ks)
        if n:
            vals.append(sum(by_d[k][0] for k in ks) / n)
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] if vals else None


# ---- registry (pre-run; no K1 output) -----------------------------------------------------------------------------

def cmd_registry(args) -> None:
    from csasr.evaluation.normalization import normalize_text, segment_units
    plan = json.loads((ROOT / BASE / "plan_sealed.json").read_text())
    rc = reference_context(plan["rows"])
    hist = {}
    for r in plan["rows"]:
        d = json.loads((ROOT / f"results/inference_cf/p2_A_r1_L16/rows/{r['index']:03d}.json").read_text())
        if d["identity"] != r["utterance_id"]:
            raise ValueError("P2-A order")
        hist[r["utterance_id"]] = d["systems"]
    reg = Counter()
    dl = defaultdict(set)
    ceil = Counter()
    ceil_u = defaultdict(set)
    ceil_d = defaultdict(set)
    for r in plan["rows"]:
        uid = r["utterance_id"]
        c = rc[uid]
        st = unit_status(c, hist[uid]["B0M_L16"]["text"])
        b1 = {u.surface for u in segment_units(normalize_text(hist[uid]["B1"]["text"]))}
        with np.load(ROOT / r["r0_arrays"]) as z:
            track = z["track_heard_intervals"]
        from csasr.inference_cf.abts0_k1 import en_fraction
        g = c["group"]
        for run in runs_registry(c, st):
            reg[(g, "runs")] += 1
            for k in ("heard", "wrong", "genuine", "reachable"):
                reg[(g, k)] += int(run[k])
            if run["genuine"] and run["heard"]:
                reg[(g, "genuine_heard")] += 1
                reg[(g, f"genuine_heard_{run['onset_rendering']}")] += 1
            if run["reachable"]:
                dl[g].add(c["dialogue_id"])
        for i, u in enumerate(st):
            if u is None or u["correct"] or not u["genuine"] or c["timing"][i] is None or c["timing"][i][0] >= c["heard"]:
                continue
            s, e = c["timing"][i]
            inr0 = max(en_fraction(track, (s + d, e + d), c["heard"]) for d in (-TOL, 0, TOL)) >= 0.5
            inb1 = c["tokens"][i] in b1
            key = "han" if u["category"] in HAN_CATS else "other"
            ceil[(g, key, "genuine_heard")] += 1
            ceil[(g, key, "in_B1")] += int(inb1)
            ceil[(g, key, "in_R0EN")] += int(inr0)
            if inb1 and inr0:
                ceil[(g, key, "both")] += 1
                ceil_u[(g, key)].add(uid)
                ceil_d[(g, key)].add(c["dialogue_id"])
    out = {"schema": "abts0_k1_registry_v1", "plan_hash": plan["plan_hash"], "k1_outputs_read": False,
           "baseline_for_registry": "historical B0M_L16 (token-identical to the matched engine on 300/300 via TTLS-R1/A2P run-time checks)",
           "registry": {f"{g}|{k}": v for (g, k), v in sorted(reg.items())},
           "reachable_dialogues": {g: len(v) for g, v in dl.items()},
           "ceiling_words": {f"{g}|{k}|{m}": v for (g, k, m), v in sorted(ceil.items())},
           "ceiling_utterances": {f"{g}|{k}": len(v) for (g, k), v in sorted(ceil_u.items())},
           "ceiling_dialogues": {f"{g}|{k}": len(v) for (g, k), v in sorted(ceil_d.items())},
           "ceiling_note": "upper-bound proxy for literal accepted selections: genuine heard B0-wrong word, present in historical "
                           "forced-EN B1 (cached_greedy path; ~80% token-identical to matched), >=0.5 inside R0 EN at CTC "
                           "timing +-200 ms; one selection per utterance, so utterances bound accepted literals",
           "timing_stats": rc["_meta"]["timing_stats"]}
    out["registry_hash"] = digest(out)
    atomic_json(ROOT / BASE / "registry_prerun.json", out)
    print(json.dumps({k: out[k] for k in ("registry", "reachable_dialogues", "ceiling_utterances", "ceiling_dialogues")}, indent=1))


# ---- K1 evaluation ------------------------------------------------------------------------------------------------

def verify_seal() -> dict:
    seal = json.loads((ROOT / BASE / "output_seal.json").read_text())
    if digest({k: v for k, v in seal.items() if k not in ("created_unix", "seal_hash")}) != seal["seal_hash"]:
        raise ValueError("output seal hash")
    bad = [p for p, h in seal["files"].items() if file_hash(ROOT / p) != h]
    if bad:
        raise ValueError(f"sealed files changed: {bad[:3]}")
    return seal


def decide(m: dict, go: dict) -> str:
    if m["invalid"]:
        return "K1_BLOCKED"
    if m["reachable_genuine_runs"] < 12 or m["reachable_dialogues"] < 6 or m["auroc_dialogues"] < go["auroc_dialogues_min"]:
        return "K1_INSUFFICIENT"
    ok = (m["accepted"] >= go["accepted_min"] and m["accepted_dialogues"] >= go["accepted_dialogues_min"]
          and m["precision"] is not None and m["precision"] >= go["precision_min"]
          and m["genuine_recoverable_words"] >= go["genuine_recoverable_words_min"]
          and m["genuine_recoverable_dialogues"] >= go["genuine_recoverable_dialogues_min"]
          and m["reachable_recall"] is not None and m["reachable_recall"] >= go["reachable_recall_min"]
          and m["auroc_macro"] is not None and m["auroc_macro"] >= go["auroc_min"])
    return "K1_PASS" if ok else "K1_FAIL"


def cmd_evaluate(args) -> None:
    from transformers import WhisperTokenizer
    seal = verify_seal()
    c = cfg()
    plan = json.loads((ROOT / BASE / "plan_sealed.json").read_text())
    sel = json.loads((ROOT / BASE / "run1/selection.json").read_text())
    tok = WhisperTokenizer.from_pretrained(plan["model_dir"], local_files_only=True)
    rc = reference_context(plan["rows"])
    go = c["evaluator"]["go_MM"]
    G = {g: Counter() for g in ("MM", "MIX", "END")}
    per_d = {g: defaultdict(Counter) for g in G}
    pos, neg = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    rec_runs = {g: set() for g in G}
    words = {g: set() for g in G}
    word_d = {g: set() for g in G}
    examples = []
    invalid = []
    b0_identity = Counter()
    for row, pr in zip(sel["rows"], plan["rows"]):
        uid = row["utterance_id"]
        r = rc[uid]
        g, dlg = r["group"], r["dialogue_id"]
        cap = json.loads((ROOT / BASE / f"run1/capture/{pr['index']:03d}.json").read_text())
        b0_identity[cap["B0"]["equals_B0M"]] += 1
        st = unit_status(r, tok.decode(cap["B0"]["tokens"], skip_special_tokens=True))
        runs = runs_registry(r, st)
        for run in runs:
            if run["reachable"]:
                G[g]["reachable_genuine_runs"] += 1
                per_d[g][dlg]["reachable"] += 1
            if run["genuine"] and run["heard"]:
                G[g]["genuine_heard_runs"] += 1
        run_of = {k: n for n, run in enumerate(runs) for k in run["units"]}
        for pool, items in (("U0", row["U0"]), ("U1", [p for p in row["proposals"] if p["status"] not in
                                                        ("EN_UNTIMED", "BEYOND_HEARD", "ZERO_DURATION", "PRESENT_IN_B0", "NO_HAN_OVERLAP",
                                                         "NONCONTIGUOUS_BLOCK", "LOW_HAN_COVER", "ONSET_MISMATCH",
                                                         "MID_CHARACTER_ONSET", "CONTENT_T0")]),
                            ("U2", [p for p in row["proposals"] if p["status"] == "ELIGIBLE"]),
                            ("SHUFFLED", [row["shuffled_chosen"]] if row["shuffled_chosen"] else [])):
            for p in items:
                lab = label_candidate(r, st, p["surface"], p["en_interval"])
                G[g][f"{pool}_n"] += 1
                G[g][f"{pool}_literal"] += int(lab["literal"])
                G[g][f"{pool}_{lab['label']}"] += 1
        for p in row["proposals"]:
            G[g][f"status_{p['status']}"] += 1
        ch = row["chosen"]
        if ch is None:
            G[g]["abstain_no_eligible"] += 1
            continue
        lab = label_candidate(r, st, ch["surface"], ch["en_interval"])
        G[g]["chosen"] += 1
        G[g][f"chosen_{lab['label']}"] += 1
        sp = ROOT / BASE / f"run1/scores/{pr['index']:03d}.json"
        sc = json.loads(sp.read_text()) if sp.exists() else {"status": "MISSING"}
        if sc["status"] in ("INVALID_ENGINE_IDENTITY", "MISSING"):
            invalid.append((uid, sc["status"]))
            continue
        if sc["status"] != "SCORED":
            G[g][f"score_{sc['status']}"] += 1
            continue
        rule = sc["rule"]
        if rule["status"] == "NONFINITE":
            G[g]["score_NONFINITE"] += 1
            continue
        G[g]["scored"] += 1
        per_d[g][dlg]["scored"] += 1
        if lab["label"] != "UNTIMED_SPAN":
            (pos if lab["literal"] else neg)[g][dlg].append(rule["C"])
        if rule["off_only_accept"]:
            G[g]["offonly_accepted"] += 1
            G[g]["offonly_literal"] += int(lab["literal"])
        if rule["accept"]:
            G[g]["accepted"] += 1
            G[g]["accepted_literal"] += int(lab["literal"])
            G[g]["accepted_literal_pm200"] += int(lab["literal_pm200"])
            G[g][f"accepted_{lab['label']}"] += 1
            per_d[g][dlg]["accepted"] += 1
            per_d[g][dlg]["accepted_literal"] += int(lab["literal"])
            if lab["literal"] and lab.get("genuine"):
                words[g].add((uid, lab["ref_index"]))
                word_d[g].add(dlg)
                n = run_of.get(lab["ref_index"])
                if n is not None and runs[n]["reachable"]:
                    rec_runs[g].add((uid, n))
        examples.append({"utterance_id": uid, "group": g, "c": ch["surface"], "b": ch["b_surface"], "t": ch["t"],
                         "label": lab["label"], "b0_rendering": lab.get("b0_rendering"), "genuine": lab.get("genuine"),
                         "C": rule["C"], "A_target_c": rule["A"]["target"]["c"], "accept": rule["accept"]})
    metrics = {}
    for g in G:
        m = dict(G[g])
        acc = m.get("accepted", 0)
        aucs = {d: auroc(pos[g][d], neg[g][d]) for d in set(pos[g]) | set(neg[g])}
        aucs = {d: v for d, v in aucs.items() if v is not None}
        reach = m.get("reachable_genuine_runs", 0)
        m.update(precision=(m.get("accepted_literal", 0) / acc) if acc else None,
                 precision_pm200=(m.get("accepted_literal_pm200", 0) / acc) if acc else None,
                 accepted=acc, accepted_dialogues=sum(1 for d in per_d[g].values() if d["accepted"]),
                 genuine_recoverable_words=len(words[g]), genuine_recoverable_dialogues=len(word_d[g]),
                 reachable_genuine_runs=reach, reachable_dialogues=sum(1 for d in per_d[g].values() if d["reachable"]),
                 reachable_recall=(len(rec_runs[g]) / reach) if reach else None,
                 recall_all_genuine_heard=(len(rec_runs[g]) / m["genuine_heard_runs"]) if m.get("genuine_heard_runs") else None,
                 auroc_by_dialogue=aucs, auroc_dialogues=len(aucs),
                 auroc_macro=float(np.mean(list(aucs.values()))) if aucs else None,
                 precision_U0=(m.get("U0_literal", 0) / m["U0_n"]) if m.get("U0_n") else None,
                 precision_U1=(m.get("U1_literal", 0) / m["U1_n"]) if m.get("U1_n") else None,
                 precision_U2=(m.get("U2_literal", 0) / m["U2_n"]) if m.get("U2_n") else None,
                 precision_chosen=(m.get("chosen_LITERAL", 0) / m["chosen"]) if m.get("chosen") else None,
                 precision_offonly=(m.get("offonly_literal", 0) / m["offonly_accepted"]) if m.get("offonly_accepted") else None,
                 precision_shuffled=(m.get("SHUFFLED_literal", 0) / m["SHUFFLED_n"]) if m.get("SHUFFLED_n") else None,
                 precision_ci_dialogue_boot=boot_precision({d: (v["accepted_literal"], v["accepted"]) for d, v in per_d[g].items()},
                                                           sorted(per_d[g]), 240924, 2000),
                 by_dialogue={d: dict(v) for d, v in sorted(per_d[g].items())}, invalid=bool(invalid))
        metrics[g] = m
    label = decide(metrics["MM"], go)
    out = {"schema": "abts0_k1_evaluation_v1", "seal_hash": seal["seal_hash"], "label": label, "metrics": metrics,
           "b0_matched_equals_b0m": {str(k): v for k, v in b0_identity.items()}, "invalid_rows": invalid,
           "examples": examples, "timing_stats": rc["_meta"]["timing_stats"]}
    atomic_json(ROOT / BASE / "evaluation.json", out)
    print(label)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("registry", "evaluate"))
    args = ap.parse_args()
    cmd_registry(args) if args.stage == "registry" else cmd_evaluate(args)


if __name__ == "__main__":
    main()
