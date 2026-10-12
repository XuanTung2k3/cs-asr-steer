#!/usr/bin/env python
"""INTERNAL-0A SECONDARY (automatic A1) diagnostic — descriptive only, never the primary verdict.

Post-seal, post-PRIMARY-audit only (same verify_seal as the frozen evaluator). Uses the unchanged ABTS A1
co-optimal B0 mapping exactly as `internal0a_offline.evaluate` computes `evaluation`, plus surface-level
appearance of each onset word. The design makes blinded two-rater audio adjudication the ONLY primary
lexical truth; this file cannot produce an INTERNAL0A_* label and does not call `decision`.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from csasr.inference_cf.core import atomic_json, digest, file_hash

BASE = "results/research_review/internal0a"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--primary-audit", required=True)
    p.add_argument("--sealed-ref", required=True)
    a = p.parse_args()
    from experiments.internal0a_offline import verify_seal
    from experiments.abts0_k1_evaluate_v2 import lexical_label, map_units, canonical_units
    from experiments.abts0_k1_evaluate import unit_status
    from transformers import WhisperTokenizer
    m = verify_seal(a)
    ep = json.loads((ROOT / BASE / "evaluation_panel.json").read_text())
    if ep["runtime_manifest_hash"] != m["manifest_hash"] or digest({k: v for k, v in ep.items() if k != "panel_hash"}) != ep["panel_hash"]:
        raise ValueError("offline population hash")
    cap = json.loads((ROOT / BASE / "run1/candidates.json").read_text())
    tok = WhisperTokenizer.from_pretrained(m["model_dir"], local_files_only=True)
    meta = {r["utterance_id"]: r for r in m["rows"]}
    T = Counter()
    tax = defaultdict(Counter)
    dial = defaultdict(set)
    onsets, selections, examples = [], [], []
    for r, e in zip(cap["rows"], ep["rows"]):
        uid = r["utterance_id"]
        assert uid == e["utterance_id"]
        b0 = json.loads((ROOT / meta[uid]["global_capture"]).read_text())["B0"]
        text = tok.decode(b0["tokens"], skip_special_tokens=True)
        hyp = canonical_units(text)
        bmap = map_units([u["surface"] for u in meta[uid]["baseline_units"]], hyp)
        st = unit_status({"tokens": e["tokens"], "is_en": e["is_en"]}, text)

        def label(x):  # identical to internal0a_offline.evaluate.label
            b = x["block"]
            H = [] if b is None else [bmap[k] for k in range(b[0], b[1] + 1)]
            return lexical_label(e["tokens"], e["is_en"], hyp, H, canonical_units(x["surface"]))
        words = [dict(w, A1=label(w)) for w in r["words"]]
        cands = [dict(c, A1=label(c)) for c in r["candidates"]]
        shuf = [dict(c, A1=label(c)) for c in r["shuffled"]["candidates"]]
        for w in words:
            tax["emission"][w["A1"]["label"]] += 1
            tax["emission_branch_" + w["branch"]][w["A1"]["label"]] += 1
        for c in cands:
            tax["group_all"][c["A1"]["label"]] += 1
            if c["cross_context_supported"]:
                tax["group_supported"][c["A1"]["label"]] += 1
            if c["selection_eligible"]:
                tax["group_eligible"][c["A1"]["label"]] += 1
        sup_keys = {(w["view_id"], w["tok_first"]) for c in cands if c["cross_context_supported"] for w in c["members"]}
        for o in e["onsets"]:
            any_w = [w for w in words if canonical_units(w["surface"]) == [o["word"]]]
            occ = [w for w in any_w if w["A1"]["label"] == "LITERAL" and w["A1"]["ref_index"] == o["ref_index"]]
            near = [w for w in any_w if w["A1"]["label"].startswith("AMBIGUOUS")]
            rec = {"uid": uid, "dialogue_id": e["dialogue_id"], "word": o["word"], "ref_index": o["ref_index"],
                   "absent_globally": not o["present_anywhere"], "local_anywhere": len(any_w),
                   "local_anywhere_branches": sorted({w["branch"] for w in any_w}),
                   "A1_occurrence_hit": bool(occ), "A1_ambiguous_emissions": len(near),
                   "supported_anywhere": any((w["view_id"], w["tok_first"]) in sup_keys for w in any_w),
                   "selected_word": next((c["surface"] for c in cands if c["candidate_id"] == r["selected"]), None)}
            onsets.append(rec)
            T.update(onsets=1, anywhere=bool(any_w), A1_hit=bool(occ), supported_anywhere=rec["supported_anywhere"])
            if rec["absent_globally"]:
                T.update(absent=1, absent_anywhere=bool(any_w), absent_A1_hit=bool(occ),
                         absent_supported_anywhere=rec["supported_anywhere"])
                if any_w:
                    dial["absent_anywhere"].add(e["dialogue_id"])
        for fam, pool, chosen in (("selected", cands, r["selected"]), ("shuffled", shuf, r["shuffled"]["selected"])):
            c = next((x for x in pool if x["candidate_id"] == chosen), None)
            if c is None:
                continue
            if fam == "shuffled" and not r["shuffled"]["informative"]:
                continue
            lab = c["A1"]["label"]
            tax[fam][lab] += 1
            i = c["A1"]["ref_index"]
            genuine = lab == "LITERAL" and st[i] is not None and st[i]["genuine"]
            T.update(**{f"{fam}": 1, f"{fam}_literal": lab == "LITERAL", f"{fam}_genuine": genuine,
                        f"{fam}_control": e["is_control"], f"{fam}_control_nonliteral": e["is_control"] and lab != "LITERAL"})
            dial[fam].add(e["dialogue_id"])
            if genuine:
                dial[f"{fam}_genuine"].add(e["dialogue_id"])
            if fam == "selected":
                selections.append({"uid": uid, "dialogue_id": e["dialogue_id"], "is_control": e["is_control"],
                                   "surface": c["surface"], "branches": c["branches"], "recurrence": c["recurrence"],
                                   "baseline_block": "".join(meta[uid]["baseline_units"][k]["surface"]
                                                             for k in range(c["block"][0], c["block"][1] + 1)),
                                   "A1": lab, "genuine": genuine, "onset_word": [o["word"] for o in e["onsets"]],
                                   "informative_shuffle": r["shuffled"]["informative"]})
        if r["shuffled"]["informative"]:
            T.update(paired=1, paired_selected=r["selected"] is not None,
                     paired_shuffled_selected=r["shuffled"]["selected"] is not None)
    out = {"schema": "internal0a_secondary_A1_v1", "PRIMARY_VERDICT": None,
           "status": "SECONDARY_DESCRIPTIVE_ONLY; primary lexical truth requires two independent bilingual audio reviewers",
           "manifest_hash": m["manifest_hash"], "counts": dict(T),
           "dialogue_breadth": {k: len(v) for k, v in dial.items()},
           "taxonomy": {k: dict(v) for k, v in tax.items()}, "onsets": onsets, "selections": selections,
           "candidates_sha256": file_hash(ROOT / BASE / "run1/candidates.json")}
    atomic_json(ROOT / BASE / "secondary_A1.json", out)
    print(json.dumps({k: out[k] for k in ("counts", "dialogue_breadth", "taxonomy")}, indent=1))


if __name__ == "__main__":
    main()
