#!/usr/bin/env python
"""ABTS-0 K1-F: CPU-only retrospective attrition diagnosis of the concluded K1 (verdict K1_INSUFFICIENT, unchanged).

Reference side only (offline diagnostic; never imported by the runner). Reads the sealed FULL300 capture, selection and
G1 gate read-only, verifies their seals, and writes only results/inference_cf/abts0_k1f/attrition.json.
No candidate, filter or threshold is selected or tuned here; nothing feeds back into K1.

For each reachable genuine MM run onset (frozen registry definition) it records
  presence   -- the onset word anywhere in the forced-EN decode (canonical units)
  position   -- the forced-EN decode MATCHes that very reference unit (frozen align_tokens; timing-free)
  filters    -- the frozen K1 status of the forced-EN proposal at that position (first failing filter)
  block      -- A1 lexical label of its associated B0 block, if association produced one
  selection  -- chosen / lost to an earlier eligible pair, and what was chosen instead
and classifies every MM proposal by what the forced-EN word corresponds to in the reference (safety view).
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, digest, file_hash
import experiments.abts0_k1_evaluate as V1
import experiments.abts0_k1_evaluate_v2 as E2

BASE = "results/inference_cf/abts0_k1"
OUT = "results/inference_cf/abts0_k1f/attrition.json"
ORDER = ("EN_UNTIMED", "BEYOND_HEARD", "ZERO_DURATION", "PRESENT_IN_B0", "NO_HAN_OVERLAP", "NONCONTIGUOUS_BLOCK",
         "LOW_HAN_COVER", "ONSET_MISMATCH", "MID_CHARACTER_ONSET", "CONTENT_T0", "NOT_IN_R0_EN", "TARGET", "CROP_MISSES_WORD",
         "OFF", "ELIGIBLE")                                   # frozen pipeline order (first failing filter wins)
STAGE = {"EN_UNTIMED": "EN_TIMING", "BEYOND_HEARD": "EN_TIMING", "ZERO_DURATION": "EN_TIMING",
         "PRESENT_IN_B0": "PRESENT_IN_B0", "NO_HAN_OVERLAP": "LOCALIZATION", "NONCONTIGUOUS_BLOCK": "LOCALIZATION",
         "LOW_HAN_COVER": "LOCALIZATION", "ONSET_MISMATCH": "LOCALIZATION", "MID_CHARACTER_ONSET": "LOCALIZATION",
         "CONTENT_T0": "T0", "NOT_IN_R0_EN": "R0_REGION", "TARGET": "CROP", "CROP_MISSES_WORD": "CROP", "OFF": "CROP",
         "ELIGIBLE": "ELIGIBLE"}


def key(status: str) -> str:
    return "TARGET" if status.startswith("TARGET_") else "OFF" if status.startswith("OFF_") else status


def verify_k1_frozen() -> dict:
    """The concluded K1 artifacts must be exactly as sealed (read-only use)."""
    cs = json.loads((ROOT / BASE / "capture_seal.json").read_text())
    bad = [p for p, h in cs["files"].items() if file_hash(ROOT / p) != h]
    cand = json.loads((ROOT / BASE / "run1/candidates_sealed.json").read_text())
    if cand["selection_sha256"] != file_hash(ROOT / BASE / "run1/selection.json"):
        bad.append("selection")
    gate = json.loads((ROOT / BASE / "gate_prescore.json").read_text())
    if digest({k: v for k, v in gate.items() if k != "gate_hash"}) != gate["gate_hash"] or gate["status"] != "K1_INSUFFICIENT":
        bad.append("gate")
    if bad:
        raise ValueError(f"K1 artifacts changed: {bad[:3]}")
    return {"capture_seal": cs["seal_hash"], "selection_sha256": cand["selection_sha256"], "gate": gate["gate_hash"],
            "k1_label": gate["status"]}


def en_correspondence(rt, is_en, st, en_canon) -> dict:
    """Canonical forced-EN index -> (ref index, class) under the frozen alignment."""
    from csasr.evaluation.mer import MATCH, SUB, align_tokens
    out = {}
    for o in align_tokens(rt, en_canon):
        if o.hyp_idx is None:
            continue
        if o.op == MATCH and is_en[o.ref_idx]:
            cls = "CORRECT_GENUINE" if st[o.ref_idx]["genuine"] else "CORRECT_B0_RIGHT" if st[o.ref_idx]["correct"] \
                else "CORRECT_ARTIFACT"
        elif o.op == SUB:
            cls = "WRONG_ENGLISH" if is_en[o.ref_idx] else "OVER_MANDARIN"
        elif o.op == MATCH:
            cls = "MATCH_NON_EN"
        else:
            cls = "INSERTED"
        out[o.hyp_idx] = (o.ref_idx, cls)
    return out


def main() -> None:
    from transformers import WhisperTokenizer
    from csasr.inference_cf.abts0_k1 import en_fraction
    frozen = verify_k1_frozen()
    plan = json.loads((ROOT / BASE / "plan_sealed.json").read_text())
    sel = json.loads((ROOT / BASE / "run1/selection.json").read_text())
    tok = WhisperTokenizer.from_pretrained(plan["model_dir"], local_files_only=True)
    rc = V1.reference_context(plan["rows"])
    onsets, pool, mm_chosen, mm_eligible = [], Counter(), [], []
    pool_examples = defaultdict(list)
    for row, pr in zip(sel["rows"], plan["rows"]):
        uid = row["utterance_id"]
        r = rc[uid]
        if r["group"] != "MM":
            continue
        cap = json.loads((ROOT / BASE / f"run1/capture/{pr['index']:03d}.json").read_text())
        b0_text = tok.decode(cap["B0"]["tokens"], skip_special_tokens=True)
        en_text = tok.decode(cap["EN"]["tokens"], skip_special_tokens=True)
        hyp, en_canon = E2.canonical_units(b0_text), E2.canonical_units(en_text)
        st = V1.unit_status(r, b0_text)
        st_full = [s if s is not None else {"genuine": False, "correct": True} for s in st]
        runs = V1.runs_registry(r, st)
        b0_map = E2.map_units([u["surface"] for u in row["b0_units"]], hyp)
        en_map = E2.map_units([u["surface"] for u in row["en_units"]], en_canon)
        corr = en_correspondence(r["tokens"], r["is_en"], st_full, en_canon)
        unit_of = {(u["tok_first"], u["surface"]): k for k, u in enumerate(row["en_units"]) if u["kind"] == "LATIN"}
        with np.load(ROOT / pr["r0_arrays"]) as z:
            track = z["track_heard_intervals"].astype(np.int64)
        ch = row["chosen"]
        ch_lab = E2.label_pair(r, st, hyp, row["b0_units"], b0_map, ch) if ch else None

        def describe(p):
            k = unit_of.get((p["en_first_token"], p["surface"]))
            ce = en_map[k] if k is not None else None
            ref_i, cls = corr.get(ce, (None, "UNMAPPED")) if ce is not None else (None, "UNMAPPED")
            d = {"surface": p["surface"], "status": p["status"], "stage": STAGE[key(p["status"])], "ref_index": ref_i,
                 "en_class": cls, "onset_diff_ms": None if p["onset_diff"] is None else p["onset_diff"] / 16,
                 "b_surface": p["b_surface"], "en_interval": p["en_interval"], "is_chosen": ch is not None and p["status"] == "ELIGIBLE"
                 and (p["en_first_token"], p["surface"]) == (ch["en_first_token"], ch["surface"])}
            if p["block"] is not None:
                lab = E2.label_pair(r, st, hyp, row["b0_units"], b0_map, p)
                d.update(a1=lab["label"], a1_ref_index=lab["ref_index"], over_correct_mandarin=lab["over_correct_mandarin"])
            if p["en_interval"] is not None:
                d["r0_en_fraction_desc"] = round(en_fraction(track, tuple(p["en_interval"]), row["heard"]), 3)
            return d

        props = [describe(p) for p in row["proposals"]]
        for d in props:
            pool[(d["stage"], d["en_class"])] += 1
            if len(pool_examples[(d["stage"], d["en_class"])]) < 3:
                pool_examples[(d["stage"], d["en_class"])].append(
                    {"uid": uid, "c": d["surface"], "b": d["b_surface"], "ref": r["tokens"][d["ref_index"]]
                     if d["ref_index"] is not None else None, "onset_diff_ms": d["onset_diff_ms"]})
            if d["status"] == "ELIGIBLE":
                mm_eligible.append(dict(d, uid=uid))
        if ch is not None:
            dch = next(d for d in props if d["is_chosen"])
            mm_chosen.append(dict(dch, uid=uid, dialogue=r["dialogue_id"], ref=r["tokens"][dch["a1_ref_index"]]
                                  if dch.get("a1_ref_index") is not None else None))
        for n, run in enumerate(runs):
            if not run["reachable"]:
                continue
            i = run["units"][0]
            w = r["tokens"][i]
            at = [d for d in props if d["ref_index"] == i and d["en_class"] == "CORRECT_GENUINE"]
            rec = {"uid": uid, "dialogue": r["dialogue_id"], "run_index": n, "ref_index": i, "word": w,
                   "run_words": [r["tokens"][k] for k in run["units"]], "b0_onset": st[i]["hyp"],
                   "present_anywhere": w in en_canon, "en_matches_position": any(corr.get(ce, (None,))[0] == i
                                                                                  for ce in corr if corr[ce][1] == "CORRECT_GENUINE"),
                   "same_word_elsewhere": [d["status"] for d in props if E2.canonical_units(d["surface"]) == [w]
                                           and d["ref_index"] != i],
                   "proposal": at[0] if at else None, "chosen_in_utterance": None}
            if ch is not None:
                rec["chosen_in_utterance"] = {"c": ch["surface"], "b": ch["b_surface"], "a1": ch_lab["label"],
                                              "is_this_onset": bool(at and at[0]["is_chosen"])}
            if not rec["present_anywhere"]:
                rec["loss"] = "GEN_ABSENT"
            elif not at:
                rec["loss"] = "GEN_MISPLACED" if not rec["en_matches_position"] else "UNMAPPED"
            else:
                p = at[0]
                if p["stage"] != "ELIGIBLE":
                    rec["loss"] = p["stage"]
                elif not p["is_chosen"]:
                    rec["loss"] = "SELECTION_EARLIER_PAIR"
                elif p.get("a1") == "LITERAL" and p.get("a1_ref_index") == i:
                    rec["loss"] = "CHOSEN_VALID"
                else:
                    rec["loss"] = "CHOSEN_WRONG_BLOCK"
            onsets.append(rec)
    # sequential funnel over the reachable onsets
    funnel = {"reachable": len(onsets), "present_anywhere": sum(o["present_anywhere"] for o in onsets),
              "aligned_proposal": sum(o["proposal"] is not None for o in onsets)}
    alive = [o for o in onsets if o["proposal"] is not None]
    for s in ORDER:
        alive = [o for o in alive if key(o["proposal"]["status"]) != s] if s != "ELIGIBLE" else alive
        funnel[f"after_{s}"] = len(alive)
    funnel["chosen"] = sum(1 for o in onsets if o["proposal"] and o["proposal"]["is_chosen"])
    funnel["chosen_A1_literal_at_onset"] = sum(o["loss"] == "CHOSEN_VALID" for o in onsets)
    loss = Counter(o["loss"] for o in onsets)
    lost_at = Counter(o["proposal"]["status"] for o in onsets if o["proposal"] and o["loss"] not in ("CHOSEN_VALID",))
    loc = [o["proposal"] for o in onsets if o["proposal"] and o["proposal"]["status"] == "ONSET_MISMATCH"]
    after_assoc = [o["proposal"] for o in onsets if o["proposal"] and o["proposal"].get("a1")]
    # independent (non-sequential) descriptive pass rates of correct onset proposals
    indep = {"with_interval": sum(1 for o in onsets if o["proposal"] and o["proposal"]["en_interval"]),
             "r0_en_fraction_ge_0.5": sum(1 for o in onsets if o["proposal"] and o["proposal"].get("r0_en_fraction_desc", 0) >= 0.5),
             "onset_mismatch_abs_ms": sorted(round(abs(p["onset_diff_ms"])) for p in loc),
             "associated_block_A1": dict(Counter(p["a1"] for p in after_assoc))}
    pool_tab = defaultdict(dict)
    for (stage, cls), n in sorted(pool.items()):
        pool_tab[stage][cls] = n
    out = {"schema": "abts0_k1f_attrition_v1", "k1_frozen": frozen, "k1_label_unchanged": frozen["k1_label"],
           "funnel_reachable_onsets": funnel, "loss_class": dict(loss), "lost_at_status": dict(lost_at),
           "independent_descriptive": indep, "mm_pool_stage_by_en_class": pool_tab,
           "mm_pool_examples": {f"{s}|{c}": v for (s, c), v in pool_examples.items()},
           "mm_chosen": mm_chosen, "mm_chosen_a1": dict(Counter(d.get("a1") for d in mm_chosen)),
           "mm_chosen_en_class": dict(Counter(d["en_class"] for d in mm_chosen)),
           "mm_eligible_en_class": dict(Counter(d["en_class"] for d in mm_eligible)),
           "mm_eligible_a1": dict(Counter(d.get("a1") for d in mm_eligible)),
           "onsets": onsets}
    out["attrition_hash"] = digest(out)
    atomic_json(ROOT / OUT, out)
    print(json.dumps({k: out[k] for k in ("funnel_reachable_onsets", "loss_class", "lost_at_status", "independent_descriptive",
                                          "mm_chosen_a1", "mm_chosen_en_class", "mm_eligible_en_class", "mm_eligible_a1")},
                     indent=1, ensure_ascii=False))
    print(json.dumps(pool_tab, indent=1))


if __name__ == "__main__":
    main()
