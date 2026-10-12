#!/usr/bin/env python
"""INTERNAL-0A OFFLINE ONLY: exposed population preparation and post-seal evaluation.

Never imported by the capture runner. Preparation selects utterances from old
labels, not word locations; runtime projection is explicitly allowlisted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
from csasr.inference_cf.core import atomic_json, digest, file_hash
from csasr.inference_cf import internal0a as I

CONFIG = "configs/research_review/internal0a.json"
BASE = "results/research_review/internal0a"


def prepare():
    import pyarrow.parquet as pq
    from transformers import WhisperTokenizer
    from csasr.evaluation.mer import MATCH, align_tokens
    from csasr.evaluation.normalization import ZH, normalize_text, segment_units, tag_unit
    from experiments.internal0a import SOURCES
    c = json.loads((ROOT / CONFIG).read_text())
    p = json.loads((ROOT / c["historical_plan"]).read_text())
    a = json.loads((ROOT / c["historical_attrition"]).read_text())
    if digest({k: v for k, v in a.items() if k != "attrition_hash"}) != a["attrition_hash"]:
        raise ValueError("historical attrition hash")
    if len(a["onsets"]) != 39 or sum(not o["present_anywhere"] for o in a["onsets"]) != 14:
        raise ValueError("historical diagnostic population changed")
    cs = json.loads((ROOT / c["historical_capture_seal"]).read_text())
    sel = json.loads((ROOT / c["historical_selection"]).read_text())
    cand = json.loads((ROOT / "results/inference_cf/abts0_k1/run1/candidates_sealed.json").read_text())
    if file_hash(ROOT / c["historical_selection"]) != cand["selection_sha256"]:
        raise ValueError("historical selection seal")
    role = json.loads((ROOT / "docs/inference_cf/R0_PANEL.json").read_text())["role_source"]
    refs = {r["utterance_id"]: r["transcript_raw"] for r in pq.read_table(role["path"],
        columns=["utterance_id", "transcript_raw"], filters=[("utterance_id", "in", p["ids"])]).to_pylist()}
    if set(refs) != set(p["ids"]):
        raise ValueError("FULL300 reference roster mismatch")
    tok = WhisperTokenizer.from_pretrained(p["model_dir"], local_files_only=True)
    focus = {o["uid"] for o in a["onsets"]}
    by_dialogue = defaultdict(list)
    captures, canonical_refs = {}, {}
    for r in p["rows"]:
        uid = r["utterance_id"]
        path = f"results/inference_cf/abts0_k1/run1/capture/{r['index']:03d}.json"
        if cs["files"].get(path) != file_hash(ROOT / path):
            raise ValueError("historical global capture seal")
        cap = json.loads((ROOT / path).read_text())
        if cap["utterance_id"] != uid or not cap["B0"]["equals_B0M"]:
            raise ValueError("matched historical B0 identity")
        captures[uid] = cap
        ru = segment_units(normalize_text(refs[uid]))
        rt = [u.surface for u in ru]
        ht = [u.surface for u in segment_units(normalize_text(tok.decode(cap["B0"]["tokens"], skip_special_tokens=True)))]
        zh = {i for i, u in enumerate(ru) if tag_unit(u) == ZH}
        matches = {o.ref_idx for o in align_tokens(rt, ht) if o.op == MATCH}
        canonical_refs[uid] = {"tokens": rt, "is_en": [tag_unit(u) == "EN" for u in ru]}
        if uid not in focus and zh and zh <= matches:
            by_dialogue[r["dialogue_id"]].append(uid)
    controls = {d: min(ids, key=lambda uid: (hashlib.sha256(f"{c['seed']}:{uid}".encode()).hexdigest(), uid))
                for d, ids in sorted(by_dialogue.items())}
    selected_ids = focus | set(controls.values())
    smap = {r["utterance_id"]: r for r in sel["rows"]}
    rows = []
    evaluation = []
    for r in p["rows"]:
        uid = r["utterance_id"]
        if uid not in selected_ids:
            continue
        path = f"results/inference_cf/abts0_k1/run1/capture/{r['index']:03d}.json"
        rows.append({k: r[k] for k in ("index", "utterance_id", "dialogue_id", "audio_path", "audio_file_sha256",
                                      "waveform_sha256", "heard_samples", "n_samples")})
        rows[-1].update(global_capture=path, global_capture_sha256=file_hash(ROOT / path),
                        baseline_units=smap[uid]["b0_units"], global_en_units=smap[uid]["en_units"])
        evaluation.append(dict(utterance_id=uid, dialogue_id=r["dialogue_id"], is_control=uid in controls.values(),
                               reference=refs[uid], onsets=[o for o in a["onsets"] if o["uid"] == uid],
                               **canonical_refs[uid]))
    # Recheck exact source waveform and small metadata hashes at run time; model
    # checkpoint bytes are hashed once in the authorized smoke / future job.
    m = {"schema": "internal0a_runtime_v1", "config_sha256": file_hash(ROOT / CONFIG),
         "git_head_at_prepare": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
         "parent_plan_hash": p["plan_hash"], "parent_panel_hash": p["panel_identity_hash"],
         "model_dir": p["model_dir"], "model_files": p["model_files"],
         "sources": {s: file_hash(ROOT / s) for s in SOURCES}, "rows": rows,
         "membership_hash": digest([r["utterance_id"] for r in rows]),
         "shuffle_map": I.shuffle_map(rows, c["seed"])}
    m["manifest_hash"] = digest(m)
    ep = {"schema": "internal0a_OFFLINE_population_v1", "runtime_manifest_hash": m["manifest_hash"],
          "attrition_sha256": file_hash(ROOT / c["historical_attrition"]), "role_source": role,
          "onset_utterances": sorted(focus), "control_roster": dict(sorted(by_dialogue.items())), "controls": controls,
          "dialogues_without_control": sorted(set(r["dialogue_id"] for r in p["rows"]) - set(controls)),
          "rows": evaluation, "references_used_only_for_diagnostic_population": True}
    ep["panel_hash"] = digest(ep)
    atomic_json(ROOT / BASE / "runtime_manifest.json", m)
    atomic_json(ROOT / BASE / "evaluation_panel.json", ep)
    print(json.dumps({"rows": len(rows), "focus": len(focus), "controls": len(controls),
                      "dialogues": len({r['dialogue_id'] for r in rows}), "membership_hash": m["membership_hash"],
                      "manifest_hash": m["manifest_hash"], "dialogues_without_control": ep["dialogues_without_control"]}))


def verify_seal(args):
    from experiments.internal0a import load_manifest
    m = load_manifest()
    seal_path = ROOT / BASE / "output_seal.json"
    s = json.loads(seal_path.read_text())
    if s["manifest_hash"] != m["manifest_hash"] or digest({k: v for k, v in s.items() if k != "seal_hash"}) != s["seal_hash"]:
        raise ValueError("output seal identity")
    for p, h in {**s["files"], **s["archives"]}.items():
        if file_hash(p if Path(p).is_absolute() else ROOT / p) != h:
            raise ValueError(f"sealed output changed: {p}")
    audit = json.loads(Path(args.primary_audit).read_text())
    if audit.get("status") != "PASS" or audit.get("seal_hash") != s["seal_hash"] or audit.get("independent") is not True:
        raise ValueError("independent PRIMARY PASS required")
    # A pushed reference must contain the exact seal and candidate bytes.
    for rel in (f"{BASE}/output_seal.json", f"{BASE}/run1/candidates.json"):
        blob = subprocess.check_output(["git", "show", f"{args.sealed_ref}:{rel}"], cwd=ROOT)
        if "sha256:" + hashlib.sha256(blob).hexdigest() != file_hash(ROOT / rel):
            raise ValueError("remote seal/candidates differ")
    return m


def review_key(uid, family, value):
    return digest([uid, family, value])


def review_inventory(cap, ep, meta):
    """Evaluator-only blinded review set. Never feeds a selection decision back."""
    from experiments.abts0_k1_evaluate_v2 import canonical_units
    out = {}
    for r, e in zip(cap["rows"], ep["rows"]):
        uid = r["utterance_id"]
        targets = {o["word"] for o in e["onsets"]}
        words = r["words"]
        sample = min(words, key=lambda w: digest([240924, uid, w["view_id"], w["tok_first"], w["tok_last"]])) if words else None
        def entry(surface, crop, interval):
            return {"utterance_id": uid, "audio_path": meta[uid]["audio_path"], "reference": e["reference"],
                    "surface": surface, "crop": crop, "predicted_interval": interval,
                    "A": {"label": None, "ref_index": None}, "B": {"label": None, "ref_index": None}}
        for w in words:
            if canonical_units(w["surface"]) in [[x] for x in targets] or w is sample:
                key = review_key(uid, "word", [w["view_id"], w["tok_first"], w["tok_last"]])
                out[key] = entry(w["surface"], w["crop"], w["interval"])
        for family, pool, chosen in (("candidate", r["candidates"], r["selected"]),
                                     ("shuffled", r["shuffled"]["candidates"], r["shuffled"]["selected"])):
            for p in pool:
                if p["candidate_id"] == chosen:
                    key = review_key(uid, family, p["candidate_id"])
                    ivs = [w["crop"] for w in p["members"]]
                    out[key] = entry(p["surface"], [min(i[0] for i in ivs), max(i[1] for i in ivs)], p["interval"])
        for o in e["onsets"]:
            if not o["present_anywhere"]:
                key = review_key(uid, "onset", o["ref_index"])
                out[key] = {"utterance_id": uid, "audio_path": meta[uid]["audio_path"], "reference": e["reference"],
                            "surface": o["word"], "reference_occurrence_index": o["ref_index"],
                            "A": {"evaluable": None}, "B": {"evaluable": None}}
    return dict(sorted(out.items()))


def review_template(args):
    m = verify_seal(args)
    ep = json.loads((ROOT / BASE / "evaluation_panel.json").read_text())
    cap = json.loads((ROOT / BASE / "run1/candidates.json").read_text())
    entries = review_inventory(cap, ep, {r["utterance_id"]: r for r in m["rows"]})
    atomic_json(ROOT / BASE / "adjudication_template.json", {"manifest_hash": m["manifest_hash"],
        "candidate_sha256": file_hash(ROOT / BASE / "run1/candidates.json"), "rater_A": None, "rater_B": None,
        "instructions": "independent bilingual audio review; no access to branch IDs, selection/support flags or earlier rater judgments; unsure/disagreement is AMBIGUOUS; TRANSLATION requires semantic equivalence to spoken Mandarin, HALLUCINATION requires no audible source, and WORD OCCURRENCE must identify exact canonical reference index",
        "entries": entries})
    print(json.dumps({"review_entries": len(entries), "path": f"{BASE}/adjudication_template.json"}))


def evaluate(args):
    from csasr.evaluation.mer import MATCH, align_tokens
    from csasr.evaluation.normalization import EN, normalize_text, segment_units, tag_unit
    from experiments.abts0_k1_evaluate_v2 import lexical_label, map_units, canonical_units
    from experiments.abts0_k1_evaluate import unit_status
    m = verify_seal(args)  # before opening evaluator-only population/labels
    c = json.loads((ROOT / CONFIG).read_text())
    ep = json.loads((ROOT / BASE / "evaluation_panel.json").read_text())
    if ep["runtime_manifest_hash"] != m["manifest_hash"] or digest({k: v for k, v in ep.items() if k != "panel_hash"}) != ep["panel_hash"]:
        raise ValueError("offline population hash")
    cap = json.loads((ROOT / BASE / "run1/candidates.json").read_text())
    if cap["manifest_hash"] != m["manifest_hash"]:
        raise ValueError("candidate manifest hash")
    from transformers import WhisperTokenizer
    tok = WhisperTokenizer.from_pretrained(m["model_dir"], local_files_only=True)
    meta = {r["utterance_id"]: r for r in m["rows"]}
    adjud = json.loads(Path(args.adjudications).read_text())
    if (adjud.get("manifest_hash") != m["manifest_hash"] or adjud.get("candidate_sha256") != file_hash(ROOT / BASE / "run1/candidates.json") or
        not adjud.get("rater_A") or not adjud.get("rater_B") or adjud["rater_A"] == adjud["rater_B"]):
        raise ValueError("complete independent A/B adjudication provenance required")
    inventory = review_inventory(cap, ep, meta)
    if set(adjud["entries"]) != set(inventory):
        raise ValueError("adjudication inventory coverage")
    for key, fixed in inventory.items():
        if any(adjud["entries"][key].get(k) != v for k, v in fixed.items() if k not in ("A", "B")):
            raise ValueError("blinded review input changed")
    def reviewed(key, e, surface):
        rec = adjud["entries"][key]
        aa, bb = rec["A"], rec["B"]
        allowed = set(c["evaluation"]["semantic_taxonomy"])
        if aa.get("label") not in allowed or bb.get("label") not in allowed:
            raise ValueError("unreviewed lexical record")
        for rr in (aa, bb):
            i = rr.get("ref_index")
            if rr["label"] == "LITERAL" and (not isinstance(i, int) or not 0 <= i < len(e["tokens"]) or
                not e["is_en"][i] or canonical_units(surface) != [e["tokens"][i]]):
                raise ValueError("literal review must match exact canonical EN reference unit")
        if aa != bb:
            return {"label": "AMBIGUOUS", "ref_index": None}
        return {"label": aa["label"], "ref_index": aa.get("ref_index")}
    per_row, total = [], Counter()
    new_dialogues, selected_dialogues, genuine_dialogues = set(), set(), set()
    by_dialogue = defaultdict(Counter)
    for r, e in zip(cap["rows"], ep["rows"]):
        uid = r["utterance_id"]
        if uid != e["utterance_id"]:
            raise ValueError("row order")
        baseline = json.loads((ROOT / meta[uid]["global_capture"]).read_text())["B0"]
        text = tok.decode(baseline["tokens"], skip_special_tokens=True)
        hyp = canonical_units(text)
        bmap = map_units([u["surface"] for u in meta[uid]["baseline_units"]], hyp)
        rc = {"tokens": e["tokens"], "is_en": e["is_en"]}
        statuses = unit_status(rc, text)
        def label(p):
            b = p["block"]
            H = [] if b is None else [bmap[k] for k in range(b[0], b[1] + 1)]
            return lexical_label(e["tokens"], e["is_en"], hyp, H, canonical_units(p["surface"]))
        lab = [dict(p, evaluation=label(p)) for p in r["candidates"]]
        # Discovery is a union of individual word occurrences, independent of
        # clustering, recurrence, selection, or conflicting cross-view blocks.
        wl = [dict(w, evaluation=label(w)) for w in r["words"]]
        for w in wl:
            key = review_key(uid, "word", [w["view_id"], w["tok_first"], w["tok_last"]])
            if key in inventory:
                w["primary_evaluation"] = reviewed(key, e, w["surface"])
        shuf = [dict(p, evaluation=label(p)) for p in r["shuffled"]["candidates"]]
        chosen = next((p for p in lab if p["candidate_id"] == r["selected"]), None)
        schosen = next((p for p in shuf if p["candidate_id"] == r["shuffled"]["selected"]), None)
        for family, chosen_p in (("candidate", chosen), ("shuffled", schosen)):
            if chosen_p is not None:
                chosen_p["primary_evaluation"] = reviewed(review_key(uid, family, chosen_p["candidate_id"]), e, chosen_p["surface"])
        selected = chosen is not None
        literal = bool(chosen and chosen["primary_evaluation"]["label"] == "LITERAL")
        ri = chosen["primary_evaluation"]["ref_index"] if literal else None
        genuine = bool(ri is not None and statuses[ri] and statuses[ri]["genuine"])
        d = Counter(selected=int(selected), literal=int(literal), genuine=int(genuine),
                    false_control=int(e["is_control"] and selected and not literal))
        if r["shuffled"]["informative"]:
            d.update(shuffled_selected=int(schosen is not None),
                     shuffled_literal=int(schosen is not None and schosen["primary_evaluation"]["label"] == "LITERAL"),
                     paired_selected=int(selected), paired_literal=int(literal))
        if selected:
            selected_dialogues.add(e["dialogue_id"])
        if genuine:
            genuine_dialogues.add(e["dialogue_id"])
        if wl:
            sample = min(wl, key=lambda w: digest([240924, uid, w["view_id"], w["tok_first"], w["tok_last"]]))
            d.update(unfiltered_sampled_utterances=1, unfiltered_weight=len(wl),
                     unfiltered_literal_weight=len(wl) * int(sample["primary_evaluation"]["label"] == "LITERAL"))
        d.update(word_emissions=len(wl), occurrence_groups=len(lab),
                 boundary_emissions=sum(w["boundary_truncation_possible"] for w in wl),
                 cap_emissions=sum(w["capped"] for w in wl), unlocated_emissions=sum(w["interval"] is None for w in wl))
        onset_results = []
        for o in e["onsets"]:
            hit = any(p.get("primary_evaluation", {}).get("label") == "LITERAL" and p["primary_evaluation"]["ref_index"] == o["ref_index"]
                      and canonical_units(p["surface"]) == [o["word"]] for p in wl)
            anywhere = any(canonical_units(p["surface"]) == [o["word"]] for p in wl)
            supported_keys = {(w["view_id"], w["tok_first"], w["tok_last"]) for p in lab if p["cross_context_supported"]
                              for w in p["members"]}
            supported_hit = any(p.get("primary_evaluation", {}).get("label") == "LITERAL" and
                                p["primary_evaluation"]["ref_index"] == o["ref_index"] and
                                (p["view_id"], p["tok_first"], p["tok_last"]) in supported_keys for p in wl)
            absent = not o["present_anywhere"]
            evaluable = True
            if absent:
                aud = adjud["entries"][review_key(uid, "onset", o["ref_index"])]
                if any(type(aud[rr].get("evaluable")) is not bool for rr in ("A", "B")):
                    raise ValueError("missing original-onset audibility review")
                evaluable = aud["A"]["evaluable"] and aud["B"]["evaluable"]
                if not evaluable:
                    hit = supported_hit = False
            d.update(all_onsets=1, occurrence_hits=int(hit), anywhere_hits=int(anywhere), supported_hits=int(supported_hit),
                     absent_onsets=int(absent), evaluable_absent=int(absent and evaluable), new_occurrence_hits=int(absent and hit))
            if absent and hit:
                new_dialogues.add(e["dialogue_id"])
            onset_results.append(dict(o, local_anywhere=anywhere, local_occurrence=hit,
                                      supported_occurrence=supported_hit, intrinsic_evaluable=evaluable))
        total.update(d)
        by_dialogue[e["dialogue_id"]].update(d)
        per_row.append({"utterance_id": uid, "dialogue_id": e["dialogue_id"], "counts": dict(d), "onsets": onset_results,
                        "candidate_labels": lab, "discovery_word_labels": wl, "shuffled_labels": shuf})
    if total["all_onsets"] != 39 or total["absent_onsets"] != 14:
        raise ValueError("39/14 denominator identity")
    precision = total["literal"] / total["selected"] if total["selected"] else 0.0
    paired = total["paired_literal"] / total["paired_selected"] if total["paired_selected"] else None
    sp = total["shuffled_literal"] / total["shuffled_selected"] if total["shuffled_selected"] else None
    metrics = {**dict(total), "new_hit_dialogues": len(new_dialogues), "selected_dialogues": len(selected_dialogues),
               "selected_genuine": total["genuine"], "selected_genuine_dialogues": len(genuine_dialogues),
               "false_control_utterances": total["false_control"], "precision": precision,
               "precision_minus_shuffled": paired - sp if paired is not None and sp is not None else None,
               "unfiltered_emission_precision_HT": total["unfiltered_literal_weight"] / total["unfiltered_weight"] if total["unfiltered_weight"] else None,
               "recall_39": total["occurrence_hits"] / 39, "new_recall_14": total["new_occurrence_hits"] / 14}
    rng = np.random.default_rng(c["seed"])
    ds = sorted(by_dialogue)
    boot = defaultdict(list)
    for _ in range(c["evaluation"]["bootstrap"]["reps"]):
        t = Counter()
        for di in rng.choice(ds, len(ds), replace=True):
            t.update(by_dialogue[di])
        for name, num, den in (("precision", "literal", "selected"), ("recall_39", "occurrence_hits", "all_onsets"),
                               ("new_recall_14", "new_occurrence_hits", "absent_onsets"),
                               ("unfiltered_emission_precision_HT", "unfiltered_literal_weight", "unfiltered_weight")):
            if t[den]:
                boot[name].append(t[num] / t[den])
    out = {"manifest_hash": m["manifest_hash"], "metrics": metrics,
           "label": I.decision(metrics, c["gates"]), "per_row": per_row,
           "dialogues": {k: dict(v) for k, v in by_dialogue.items()},
           "bootstrap_95": {k: np.quantile(v, [.025, .975]).tolist() for k, v in boot.items()},
           "candidate_taxonomy": dict(Counter(p["evaluation"]["label"] for r in per_row for p in r["candidate_labels"])),
           "primary_selected_taxonomy": dict(Counter(p["primary_evaluation"]["label"] for r in per_row for p in r["candidate_labels"] if "primary_evaluation" in p)),
           "adjudications_sha256": file_hash(args.adjudications),
           "oracle_union_is_not_deployable": True, "steering_authorized": False}
    atomic_json(ROOT / BASE / "evaluation.json", out)
    print(json.dumps({"metrics": metrics, "label": out["label"]}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "review-template", "evaluate"])
    p.add_argument("--primary-audit")
    p.add_argument("--sealed-ref")
    p.add_argument("--adjudications")
    a = p.parse_args()
    if a.stage == "prepare":
        prepare()
    else:
        if not a.primary_audit or not a.sealed_ref or not a.sealed_ref.startswith("origin/"):
            p.error("evaluate requires independent --primary-audit and pushed --sealed-ref origin/...")
        if a.stage == "review-template":
            review_template(a)
        elif not a.adjudications:
            p.error("evaluate requires complete blinded --adjudications")
        else:
            evaluate(a)


if __name__ == "__main__":
    main()
