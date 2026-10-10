#!/usr/bin/env python
"""R0 analysis (frozen configs/inference_cf/r0_region_vector.json, spec sections 7-9).

primary  (reference-free; cannot open oracle/CTC/role paths) completeness of 300 x 4 records, integrity, region
         durations / window abstentions, query mapping counts, per-layer construction validity / rank / failure
         reasons, numerical repeat identity, perturbation stability (S-gate inputs), shuffle and script diagnostics,
         v_prompt descriptive cosines.
oracle   (ONLY after the committed output seal + committed `R0_AUDIT: PASS (PRIMARY)`) reference-derived MMS-FA timing
         proxy -> oracle sample/frame masks on the SAME sealed queries/states/attention; identical construction;
         region quality, oracle coverage, paired agreement, shuffle advantage, dialogue bootstrap (B=10000, seed 240924,
         Bonferroni family 8), nested gates O/P/S/I and the terminal label; English-omission diagnostic.
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

from csasr.inference_cf.core import atomic_json, file_hash

CONFIG = "configs/inference_cf/r0_region_vector.json"
PANEL = "docs/inference_cf/R0_PANEL.json"
BASE = "results/inference_cf/r0"
SEAL = f"{BASE}/output_seal.json"
PRIMARY_AUDIT = f"{BASE}/primary_audit.json"
LAYERS = (3, 8, 16, 24)
LABELS = ("R0_INVALID", "R0_ORACLE_CONSTRUCTION_INSUFFICIENT", "R0_PREDICTED_REGION_INSUFFICIENT", "R0_VECTOR_UNSTABLE",
          "R0_REGION_SIGNAL_INSUFFICIENT", "R0_PARTIAL_FEASIBILITY", "R0_READY_FOR_R1")


def cfg() -> dict:
    return json.loads((ROOT / CONFIG).read_text())


def committed(rel: str) -> bool:
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    tracked = subprocess.run(["git", "ls-files", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip() == rel
    return tracked and "sha256:" + hashlib.sha256(blob).hexdigest() == file_hash(ROOT / rel)


def cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_rows(run: Path, man: dict) -> list[tuple[dict, dict]]:
    out = []
    for i in range(300):
        r = json.loads((run / "rows" / f"{i:03d}.json").read_text())
        if r["manifest_hash"] != man["manifest_hash"] or file_hash(run / "rows" / f"{i:03d}.npz") != r["arrays_sha256"]:
            raise ValueError(f"row hash {i}")
        with np.load(run / "rows" / f"{i:03d}.npz") as z:
            out.append((r, {k: z[k] for k in z.files}))
    return out


# ---- frozen gate logic (pure) ---------------------------------------------------------------------------------------

def gate_S(stab: dict, g: dict) -> bool:
    return (stab["n_primary_valid"] > 0 and stab["both_valid_fraction"] >= g["perturb_both_valid_fraction_min"]
            and stab["stable_fraction"] >= g["perturb_stable_fraction_min"]
            and stab["median_min_cos"] is not None and stab["median_min_cos"] >= g["perturb_median_min_signed_cosine"])


def stability(rows_l: list[dict], g: dict) -> dict:
    """rows_l: per primary-valid row {'erode_cos': float|None, 'dilate_cos': float|None} (None = perturbation invalid)."""
    n = len(rows_l)
    both = [r for r in rows_l if r["erode_cos"] is not None and r["dilate_cos"] is not None]
    mins = [min(r["erode_cos"], r["dilate_cos"]) for r in both]
    stable = sum(m >= g["perturb_row_min_signed_cosine"] for m in mins)
    return {"n_primary_valid": n, "n_both_valid": len(both), "both_valid_fraction": len(both) / n if n else 0.0,
            "stable_fraction": stable / n if n else 0.0, "median_min_cos": float(np.median(mins)) if mins else None}


def draws(keys: list[str], reps: int, seed: int) -> np.ndarray:
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(reps, len(keys)))
    W = np.zeros((reps, len(keys)))
    for j in range(len(keys)):
        W[:, j] = (idx == j).sum(axis=1)
    return W


def macro_boot(pairs: list[tuple[str, float]], keys: list[str], W: np.ndarray, lo_q: float, hi_q: float) -> dict:
    by = defaultdict(list)
    for d, v in pairs:
        by[d].append(v)
    m = np.array([np.mean(by[k]) if by.get(k) else np.nan for k in keys])
    has = ~np.isnan(m)
    out = {"dialogues": int(has.sum()), "rows": len(pairs), "estimate": None, "ci": None, "usable_draws": 0}
    if not has.any():
        return out
    out["estimate"] = float(m[has].mean())
    num, den = W[:, has] @ m[has], W[:, has].sum(axis=1)
    ok = den > 0
    dr = num[ok] / den[ok]
    out["usable_draws"] = int(ok.sum())
    out["ci"] = [float(np.quantile(dr, lo_q)), float(np.quantile(dr, hi_q))]
    return out


def decide(valid: bool, layers: dict, g: dict) -> dict:
    """layers[l] = {'O','P','S','I'} booleans (nested)."""
    if not valid:
        return {"label": "R0_INVALID", "passing_I": []}
    O = [l for l, x in layers.items() if x["O"]]
    if not O:
        return {"label": "R0_ORACLE_CONSTRUCTION_INSUFFICIENT", "passing_I": []}
    P = [l for l in O if layers[l]["P"]]
    if not P:
        return {"label": "R0_PREDICTED_REGION_INSUFFICIENT", "passing_I": []}
    S = [l for l in P if layers[l]["S"]]
    if not S:
        return {"label": "R0_VECTOR_UNSTABLE", "passing_I": []}
    I = [l for l in S if layers[l]["I"]]
    if not I:
        return {"label": "R0_REGION_SIGNAL_INSUFFICIENT", "passing_I": []}
    if len(I) < g["ready_layers_min"] or not ({16, 24} & set(int(l) for l in I)):
        return {"label": "R0_PARTIAL_FEASIBILITY", "passing_I": sorted(int(l) for l in I)}
    return {"label": "R0_READY_FOR_R1", "passing_I": sorted(int(l) for l in I), "requires_full_audit_PASS": True}


# ---- primary (reference-free) ----------------------------------------------------------------------------------------

def primary(run_rel: str) -> dict:
    c = cfg()
    run = ROOT / run_rel
    man = json.loads((run / "manifest.json").read_text())
    rt = json.loads((run / "runtime.json").read_text())
    missing = [i for i in range(300) if not (run / "rows" / f"{i:03d}.json").exists()]
    integ = {"runtime_completed": rt.get("status") == "completed", "rows_complete_300": not missing,
             "no_grad_no_training": bool(rt.get("model_grads_none")) and not rt.get("requires_grad_any") and not rt.get("training_mode"),
             "weights_unchanged": bool(rt.get("weights_unchanged_sample")),
             "no_autograd_optimizer_edits": all(rt["counters"].get(k, 0) == 0 for k in ("autograd_calls", "optimizer_steps", "edit_hooks"))}
    if missing:
        return {"schema": "r0_primary_v1", "valid": False, "integrity": integ, "missing_rows": missing, "references_used": False}
    rows = load_rows(run, man)
    P = json.loads((ROOT / PANEL).read_text())["rows"]
    integ["row_identity_order"] = [r["utterance_id"] for r, _ in rows] == [p["utterance_id"] for p in P]
    integ["records_4_layers"] = all(set(r["constructions"]) == {str(l) for l in LAYERS} for r, _ in rows)
    integ["row_integrity_all"] = all(all(v for k, v in r["integrity"].items() if k != "logits_sha256") for r, _ in rows)
    win_lab, win_reasons = Counter(), Counter()
    dur_full, dur_heard = Counter(), Counter()
    q = Counter()
    agree = []
    for r, _ in rows:
        for w in r["windows"]:
            win_lab[w["label"]] += 1
            for x in w["reasons"]:
                win_reasons[x] += 1
        dur_full.update(r["durations_full"])
        dur_heard.update(r["durations_heard"])
        q.update(r["counts"])
        agree += [x["argmax_agrees"] for x in r["queries"] if x["eligible"]]
    per_layer = {}
    for l in LAYERS:
        L = str(l)
        valid, ranks, reasons, dl = 0, Counter(), Counter(), set()
        stab_rows, shuf_cos, shuf_valid_counts, script_cos, vp_cos = [], [], [], [], []
        shuf_reasons = Counter()
        pert_valid = Counter()
        for r, z in rows:
            pr = r["constructions"][L]["primary"]
            for nm in ("erode", "dilate"):
                pert_valid[nm] += r["controls"][nm][L]["status"] == "ok"
            sh = r["controls"]["shuffle"][L]
            shuf_valid_counts.append(sum(x["status"] == "ok" for x in sh))
            for x in sh:
                for rr in x["reasons"][:1]:
                    shuf_reasons[rr] += 1
            if pr["status"] != "ok":
                for rr in pr["reasons"][:1]:
                    reasons[rr] += 1
                for rr in pr["reasons"][1:]:
                    reasons["detail:" + rr] += 1
                continue
            valid += 1
            dl.add(r["dialogue_id"])
            ranks[pr["rank"]] += 1
            v = z[f"v_L{l}"]
            e = cos(v, z[f"erode_L{l}"]) if f"erode_L{l}" in z else None
            d = cos(v, z[f"dilate_L{l}"]) if f"dilate_L{l}" in z else None
            stab_rows.append({"erode_cos": e, "dilate_cos": d})
            sc = [cos(v, z[f"shuf_L{l}_{k:02d}"]) for k in range(20) if f"shuf_L{l}_{k:02d}" in z]
            if sc:
                shuf_cos.append(float(np.mean(sc)))
            if f"script_L{l}" in z:
                script_cos.append(cos(v, z[f"script_L{l}"]))
            m = r["v_prompt"][L]["mean_delta_cos_primary"]
            if m is not None:
                vp_cos.append(m)
        st = stability(stab_rows, c["gates"])
        per_layer[L] = {"primary_valid": valid, "primary_valid_dialogues": len(dl), "rank_distribution": dict(ranks),
                        "first_failure_reasons": dict(reasons), "stability": st, "S_reference_free": gate_S(st, c["gates"]),
                        "perturb_valid": dict(pert_valid), "shuffle_valid_per_row": {"mean": float(np.mean(shuf_valid_counts)),
                                                                                     "rows_ge_10": int(sum(x >= 10 for x in shuf_valid_counts))},
                        "shuffle_first_failure_reasons": dict(shuf_reasons),
                        "mean_shuffle_vs_primary_signed_cos": {"rows": len(shuf_cos), "median": float(np.median(shuf_cos)) if shuf_cos else None},
                        "script_valid": sum(r["controls"]["script"][L]["status"] == "ok" for r, _ in rows),
                        "script_vs_primary_signed_cos_median": float(np.median(script_cos)) if script_cos else None,
                        "vprompt_mean_delta_cos_primary_median": float(np.median(vp_cos)) if vp_cos else None,
                        "repeat_identical": all(r["constructions"][L]["primary_repeat_identical"] for r, _ in rows)}
    integ["numerical_repeat_identical"] = all(per_layer[str(l)]["repeat_identical"] for l in LAYERS)
    valid = all(integ.values())
    return {"schema": "r0_primary_v1", "manifest_hash": man["manifest_hash"], "references_used": False, "valid": valid, "integrity": integ,
            "windows": {"labels": dict(win_lab), "abstention_reasons": dict(win_reasons), "total": sum(win_lab.values())},
            "durations_samples": {"full": dict(dur_full), "heard": dict(dur_heard)}, "queries": dict(q),
            "replay_argmax_agreement_eligible": float(np.mean(agree)) if agree else None,
            "shuffle_unavailable_rows": sum(not r["controls"]["shuffle_available"] for r, _ in rows),
            "per_layer": per_layer,
            "runtime": {k: rt.get(k) for k in ("job_id", "elapsed_sec", "peak_alloc", "peak_reserved", "counters", "gpu", "smoke", "smoke10")}}


# ---- oracle (post-seal) ----------------------------------------------------------------------------------------------

def oracle_tracks(rows: list[dict], c: dict) -> tuple[dict, dict]:
    """Reference-derived MMS-FA timing proxy -> full-waveform EN/ZH/U sample tracks. Evaluator-only."""
    import pyarrow.parquet as pq
    from csasr.nat5h.units import build_reference_units
    src = c["oracle"]["source"]
    pre = src["structural_preflight"]
    if file_hash(src["path"]) != src["file_sha256"] or file_hash(src["manifest_path"]) != src["manifest_sha256"]:
        raise ValueError("R0_DESIGN_BLOCKED:ORACLE_SOURCE")
    ids = [r["utterance_id"] for r in rows]
    t = pq.read_table(src["path"], filters=[("utterance_id", "in", ids)],
                      columns=["utterance_id", "reference_unit_index", "unit_id", "aligner_family", "aligner_variant", "is_valid", "start_sample",
                               "end_sample", "sample_rate", "coordinate_origin", "waveform_num_samples", "language", "surface", "mapping_method",
                               "metadata"]).to_pylist()
    role = pq.read_table(json.loads((ROOT / PANEL).read_text())["role_source"]["path"], filters=[("utterance_id", "in", ids)],
                         columns=["utterance_id", "transcript_raw"]).to_pylist()
    ref_units = {x["utterance_id"]: build_reference_units(x["utterance_id"], x["transcript_raw"]) for x in role}
    n_by = {r["utterance_id"]: r["n_samples"] for r in rows}
    st = Counter()
    keyseen: dict = {}
    by = defaultdict(list)
    if set(ids) - set(ref_units):
        raise ValueError("R0_INVALID:role_reference_missing")
    for x in t:
        st["records"] += 1
        k = (x["utterance_id"], x["reference_unit_index"], x["aligner_family"], x["aligner_variant"])
        val = (x["start_sample"], x["end_sample"], x["language"], x["surface"])
        if k in keyseen:
            if keyseen[k] != val:
                raise ValueError("R0_INVALID:conflicting_duplicate_oracle_record")
            st["duplicates_agree"] += 1
            continue
        keyseen[k] = val
        meta = x["metadata"] or {}
        if x["sample_rate"] != pre["sample_rate"] or x["coordinate_origin"] != pre["coordinate_origin"] or x["mapping_method"] != pre["mapping_method"] \
                or meta.get("normalization_version") != pre["normalization_version"]:
            raise ValueError("R0_INVALID:oracle_coordinate_metadata")
        units = ref_units[x["utterance_id"]]
        # historical nat5h normalization: per-utterance hash recomputed from the role transcript (the frozen preflight
        # value sha256:f781bcc1... is canonical row 0's per-utterance hash and is checked on that row)
        if not units or meta.get("normalization_hash") != units[0].normalization_hash:
            st["normalization_hash_mismatch"] += 1
            continue
        n = n_by[x["utterance_id"]]
        if x["waveform_num_samples"] != n:
            st["waveform_length_mismatch"] += 1
            continue
        if not x["is_valid"]:
            st["invalid"] += 1
            continue
        s, e = x["start_sample"], x["end_sample"]
        if s is None or e is None or not (0 <= s < e <= n):
            st["out_of_bounds"] += 1
            continue
        lang = x["language"]
        ui = x["unit_id"]
        if x["reference_unit_index"] != ui or not (0 <= ui < len(units)) or units[ui].surface != x["surface"] or units[ui].language != lang:
            st["language_surface_inconsistent"] += 1
            continue
        if lang not in ("EN", "ZH"):
            st["unknown_language"] += 1
            continue
        st[f"used_{lang}"] += 1
        by[x["utterance_id"]].append((s, e, lang, ui))
    row0 = rows[0]["utterance_id"]
    if ref_units[row0][0].normalization_hash != pre["normalization_hash"]:
        raise ValueError("R0_INVALID:pinned_row0_normalization_hash")
    tracks = {}
    for r in rows:
        n = r["n_samples"]
        e = np.zeros(n, bool)
        m = np.zeros(n, bool)
        for s, en, lang, _ in by[r["utterance_id"]]:
            (e if lang == "EN" else m)[s:en] = True
        tr = np.zeros(n, np.int8)
        tr[e & ~m] = 1
        tr[m & ~e] = 2
        st["conflict_samples"] += int((e & m).sum())
        tracks[r["utterance_id"]] = tr
    st["utterances_with_records"] = len(by)
    return tracks, {"stats": dict(st), "units": {u: v for u, v in by.items()},
                    "reference": {u: (v[0].normalized_transcript if v else "") for u, v in ref_units.items()}}


def boundaries(track: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.diff(track)) + 1


def boundary_match(ref_b: np.ndarray, pred_b: np.ndarray, tol: int) -> tuple[int, list[int]]:
    """One-to-one nearest matching within tol; oracle boundaries in order, earliest predicted on ties."""
    used = set()
    errs = []
    for b in ref_b:
        cand = [(abs(int(p) - int(b)), int(p)) for p in pred_b if int(p) not in used and abs(int(p) - int(b)) <= tol]
        if cand:
            d, p = min(cand)
            used.add(p)
            errs.append(d)
    return len(errs), errs


def oracle(run_rel: str) -> dict:
    from csasr.inference_cf import r0_regions as R
    from csasr.inference_cf import r0_unique as Q
    from csasr.evaluation.mer import ref_unit_status
    from experiments.inference_cf_r0 import unpack
    c = cfg()
    if not committed(SEAL):
        raise PermissionError("R0 output seal missing/uncommitted: oracle stays closed")
    seal = json.loads((ROOT / SEAL).read_text())
    if any(file_hash(ROOT / p) != h for p, h in seal["files"].items()):
        raise ValueError("sealed outputs changed")
    if not committed(PRIMARY_AUDIT) or json.loads((ROOT / PRIMARY_AUDIT).read_text())["verdict"] != c["audits"]["primary"]:
        raise PermissionError("R0_AUDIT: PASS (PRIMARY) required before oracle access")
    prim = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    run = ROOT / run_rel
    man = json.loads((run / "manifest.json").read_text())
    rows = load_rows(run, man)
    tracks, meta = oracle_tracks([r for r, _ in rows], c)          # oracle opened only here
    mc, uc, g, b = c["mapping"], c["unique"], c["gates"], c["bootstrap"]
    # ---- region quality (heard horizon known support; pooled) ----
    acc = Counter()
    rows_both, dl_both = 0, set()
    bmatch, bref, berr = 0, 0, []
    omitted = []
    full = Counter()
    for r, z in rows:
        heard = r["heard_samples"]
        pred = R.intervals_to_track([tuple(x) for x in r["intervals_full"]], r["n_samples"])
        orc = tracks[r["utterance_id"]]
        for scope, sl in (("heard", slice(0, heard)), ("full", slice(0, r["n_samples"]))):
            p, o = pred[sl], orc[sl]
            known = o > 0
            tgt = acc if scope == "heard" else full
            for k, code in (("EN", 1), ("ZH", 2)):
                tgt[f"tp_{k}"] += int(np.sum((p == code) & (o == code)))
                tgt[f"predpos_known_{k}"] += int(np.sum((p == code) & known))
                tgt[f"orc_{k}"] += int(np.sum(o == code))
                tgt[f"union_{k}"] += int(np.sum((p == code) | (o == code)))
            tgt["known"] += int(known.sum())
            tgt["classified_known"] += int(np.sum((p > 0) & known))
            tgt["pred_on_orcU_EN"] += int(np.sum((p == 1) & ~known))
            tgt["pred_on_orcU_ZH"] += int(np.sum((p == 2) & ~known))
            tgt["orcU"] += int(np.sum(~known))
        oh = orc[:heard]
        if np.any(oh == 1) and np.any(oh == 2):
            rows_both += 1
            dl_both.add(r["dialogue_id"])
        nb, errs = boundary_match(boundaries(oh), boundaries(pred[:heard]), 3200)
        bmatch += nb
        bref += len(boundaries(oh))
        berr += errs
        # English omitted by the baseline hypothesis (canonical ref/hyp alignment), predicted acoustic coverage
        hyp = json.loads((ROOT / json.loads((ROOT / PANEL).read_text())["rows"][r["canonical_index"]]["baseline"]["path"]).read_text())["theta0"]["text"]
        status = ref_unit_status(meta["reference"][r["utterance_id"]] or "", hyp)
        for s, e, lang, ui in meta["units"].get(r["utterance_id"], []):
            if lang == "EN" and 0 <= ui < len(status) and status[ui] == "del":
                hs, he = min(s, heard), min(e, heard)
                qa = [x for x in r["queries"] if x["eligible"] and x["mapped"] and s <= x["argmax_frame"] * 320 < e]
                omitted.append({"utterance_id": r["utterance_id"], "dialogue_id": r["dialogue_id"], "samples": e - s,
                                "pred_EN_fraction_full": float(np.mean(pred[s:e] == 1)),
                                "heard_fraction": (he - hs) / (e - s), "queries_attending": len(qa),
                                "queries_attending_assigned_EN": sum(x["label"] == 1 for x in qa)})

    def ratio(a, d):
        return a / d if d else None
    rq = {"EN_precision": ratio(acc["tp_EN"], acc["predpos_known_EN"]), "EN_recall": ratio(acc["tp_EN"], acc["orc_EN"]),
          "ZH_precision": ratio(acc["tp_ZH"], acc["predpos_known_ZH"]), "ZH_recall": ratio(acc["tp_ZH"], acc["orc_ZH"]),
          "classified_coverage": ratio(acc["classified_known"], acc["known"]), "rows_with_known_EN_and_ZH": rows_both,
          "dialogues_with_known_EN_and_ZH": len(dl_both),
          "EN_IoU": ratio(acc["tp_EN"], acc["union_EN"]), "ZH_IoU": ratio(acc["tp_ZH"], acc["union_ZH"]),
          "pred_on_oracle_U": {"EN": ratio(acc["pred_on_orcU_EN"], acc["orcU"]), "ZH": ratio(acc["pred_on_orcU_ZH"], acc["orcU"])},
          "boundary_match_200ms": {"matched": bmatch, "oracle_boundaries": bref, "rate": ratio(bmatch, bref),
                                   "median_abs_error_samples": float(np.median(berr)) if berr else None},
          "counts_samples_heard": dict(acc), "full_waveform_descriptive": {
              "EN_precision": ratio(full["tp_EN"], full["predpos_known_EN"]), "EN_recall": ratio(full["tp_EN"], full["orc_EN"]),
              "ZH_precision": ratio(full["tp_ZH"], full["predpos_known_ZH"]), "ZH_recall": ratio(full["tp_ZH"], full["orc_ZH"]),
              "classified_coverage": ratio(full["classified_known"], full["known"]),
              "oracle_EN_beyond_heard_samples": full["orc_EN"] - acc["orc_EN"]}}
    rq_pass = (all(rq[k] is not None for k in ("EN_precision", "EN_recall", "ZH_precision", "ZH_recall", "classified_coverage"))
               and rq["EN_precision"] >= g["en_precision_min"] and rq["EN_recall"] >= g["en_recall_min"]
               and rq["ZH_precision"] >= g["zh_precision_min"] and rq["ZH_recall"] >= g["zh_recall_min"]
               and rq["classified_coverage"] >= g["known_region_classified_coverage_min"]
               and rows_both >= g["known_region_scored_min_rows"] and len(dl_both) >= g["known_region_scored_min_dialogues"])
    rq["pass"] = rq_pass
    # ---- oracle constructions on the SAME sealed queries / states / attention ----
    recs = {str(l): [] for l in LAYERS}
    mapping_identical = True
    oracle_vectors = {}
    for r, z in rows:
        heard = r["heard_samples"]
        F = r["real_frames"]
        heads = unpack(z["heads"]).astype(np.float32)
        if heads.shape[2] != F:
            raise ValueError("sealed heads frame count")
        mp = R.query_mapping(heads, mc["raw_real_audio_attention_mass_min"])
        elig = np.array([x["eligible"] for x in r["queries"]])
        mapping_identical &= bool(np.array_equal(mp["raw_mass"], z["raw_mass"]) and np.array_equal(R.time_bins(mp["argmax_frame"], mc["independent_time_bin_samples"]), z["bins"]))
        we, wm = R.frame_masks(tracks[r["utterance_id"]][:heard])
        asg = R.assign_queries(mp["a"], mp["mapped"], elig, we, wm, mc)
        bins = z["bins"]
        for l in LAYERS:
            S = unpack(z[f"states_L{l}"])
            e, m = np.flatnonzero(asg["label"] == 1), np.flatnonzero(asg["label"] == 2)
            o = Q.construct(S[e], S[m], bins[e], bins[m], uc)
            pv = z.get(f"v_L{l}")
            ov = o["vector"]
            if ov is not None:
                oracle_vectors[f"{r['canonical_index']:03d}_L{l}"] = ov
            sh = [z[f"shuf_L{l}_{k:02d}"] for k in range(20) if f"shuf_L{l}_{k:02d}" in z]
            rec = {"q": r["canonical_index"], "dialogue_id": r["dialogue_id"], "oracle_status": o["status"], "oracle_rank": o["rank"],
                   "oracle_reasons": o["reasons"], "n_E": o["n_E"], "n_M": o["n_M"], "primary_valid": pv is not None,
                   "paired": pv is not None and ov is not None, "n_valid_shuffles": len(sh)}
            if rec["paired"]:
                rec["signed_cos"] = cos(pv, ov)
                rec["abs_cos"] = abs(rec["signed_cos"])
                if sh:
                    rec["shuffle_oracle_mean"] = float(np.mean([cos(s, ov) for s in sh]))
                    rec["advantage"] = rec["signed_cos"] - rec["shuffle_oracle_mean"]
                rec["signal_evaluable"] = len(sh) >= c["controls"]["minimum_valid_shuffles_per_pair"]
            if pv is not None:
                for nm in ("erode", "dilate"):
                    rec[f"{nm}_cos"] = cos(pv, z[f"{nm}_L{l}"]) if f"{nm}_L{l}" in z else None
            recs[str(l)].append(rec)
    # ---- gates + bootstrap ----
    keys = sorted({r["dialogue_id"] for r, _ in rows})
    W = draws(keys, b["replicates"], b["seed"])
    layers = {}
    for l in LAYERS:
        L = str(l)
        R_ = recs[L]
        ov = [x for x in R_ if x["oracle_status"] == "ok"]
        pa = [x for x in R_ if x["paired"]]
        pv = [x for x in R_ if x["primary_valid"]]
        O = len(ov) >= g["oracle_valid_min"] and len({x["dialogue_id"] for x in ov}) >= g["oracle_dialogues_min"]
        P_ = O and rq_pass and len(pa) >= g["paired_valid_min"] and len({x["dialogue_id"] for x in pa}) >= g["paired_dialogues_min"] \
            and (len(pa) / len(ov) if ov else 0) >= g["paired_fraction_of_oracle_min"]
        st = stability([{"erode_cos": x["erode_cos"], "dilate_cos": x["dilate_cos"]} for x in pv], g)
        S_ = P_ and gate_S(st, g)
        ev = [x for x in pa if x.get("signal_evaluable")]
        sm = macro_boot([(x["dialogue_id"], x["signed_cos"]) for x in ev], keys, W, b["lower_quantile"], b["upper_quantile"])
        ad = macro_boot([(x["dialogue_id"], x["advantage"]) for x in ev], keys, W, b["lower_quantile"], b["upper_quantile"])
        draws_ok = sm["usable_draws"] >= b["minimum_valid_draws"] and ad["usable_draws"] >= b["minimum_valid_draws"]
        I_ = (S_ and (len(ev) / len(pa) if pa else 0) >= g["signal_evaluable_fraction_of_paired_min"]
              and len(ev) >= g["paired_valid_min"] and len({x["dialogue_id"] for x in ev}) >= g["paired_dialogues_min"] and draws_ok
              and sm["estimate"] is not None and sm["estimate"] >= g["signed_oracle_mean_min"] and sm["ci"][0] > g["signed_oracle_adjusted_lower_min"]
              and ad["estimate"] >= g["shuffle_advantage_mean_min"] and ad["ci"][0] > g["shuffle_advantage_adjusted_lower_min"])
        pw95 = macro_boot([(x["dialogue_id"], x["signed_cos"]) for x in pa], keys, W, 0.025, 0.975)
        ab95 = macro_boot([(x["dialogue_id"], x["abs_cos"]) for x in pa], keys, W, 0.025, 0.975)
        layers[L] = {"O": bool(O), "P": bool(P_), "S": bool(S_), "I": bool(I_),
                     "oracle_valid": len(ov), "oracle_valid_dialogues": len({x["dialogue_id"] for x in ov}),
                     "primary_valid": len(pv), "paired": len(pa), "paired_dialogues": len({x["dialogue_id"] for x in pa}),
                     "paired_fraction_of_oracle": len(pa) / len(ov) if ov else None,
                     "oracle_rank_distribution": dict(Counter(x["oracle_rank"] for x in ov)),
                     "oracle_first_failure_reasons": dict(Counter(x["oracle_reasons"][0] for x in R_ if x["oracle_status"] != "ok")),
                     "stability": st, "signal_evaluable": len(ev), "signal_evaluable_fraction": len(ev) / len(pa) if pa else None,
                     "signed_cos_macro_adjusted": sm, "advantage_macro_adjusted": ad, "draws_ok": draws_ok,
                     "signed_cos_paired_pointwise95": pw95, "abs_cos_paired_pointwise95": ab95,
                     "signed_cos_median_paired": float(np.median([x["signed_cos"] for x in pa])) if pa else None,
                     "abs_cos_median_paired": float(np.median([x["abs_cos"] for x in pa])) if pa else None,
                     "shuffle_oracle_mean_median": float(np.median([x["shuffle_oracle_mean"] for x in ev])) if ev else None,
                     "paired_per_dialogue": dict(Counter(x["dialogue_id"] for x in pa)),
                     "oracle_valid_per_dialogue": dict(Counter(x["dialogue_id"] for x in ov))}
    valid = bool(prim["valid"]) and bool(mapping_identical)
    dec = decide(valid, layers, g)
    om_by_d = Counter(x["dialogue_id"] for x in omitted)
    return {"schema": "r0_oracle_v1", "output_seal_hash": seal["seal_hash"], "oracle_label": "REFERENCE-DERIVED TIMING PROXY; EVALUATOR-ONLY; NOT DEPLOYABLE",
            "oracle_source": meta["stats"], "mapping_identical_to_sealed": bool(mapping_identical), "valid": valid,
            "region_quality": rq, "layers": layers, "decision": dec, "label": dec["label"],
            "english_omission": {"omitted_EN_units": len(omitted), "dialogues": len(om_by_d),
                                 "with_pred_EN_fraction_ge_0.5": sum(x["pred_EN_fraction_full"] >= 0.5 for x in omitted),
                                 "fully_beyond_heard": sum(x["heard_fraction"] == 0 for x in omitted),
                                 "with_any_attending_query": sum(x["queries_attending"] > 0 for x in omitted),
                                 "with_attending_query_assigned_EN": sum(x["queries_attending_assigned_EN"] > 0 for x in omitted),
                                 "median_pred_EN_fraction": float(np.median([x["pred_EN_fraction_full"] for x in omitted])) if omitted else None,
                                 "units": omitted},
            "per_row": recs, "bootstrap": {"replicates": b["replicates"], "seed": b["seed"], "dialogues": keys,
                                           "adjusted_quantiles": [b["lower_quantile"], b["upper_quantile"]]}}, oracle_vectors


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("primary", "oracle"):
        p = sub.add_parser(nm)
        p.add_argument("--run", required=True)
        p.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("analysis exists; never overwrite")
    from experiments.inference_cf_p2dir_analyze import jsonable
    if args.cmd == "primary":
        res = jsonable(primary(args.run))
        atomic_json(out, res)
        print(json.dumps({k: res.get(k) for k in ("valid", "integrity")}, indent=1))
        for l, x in res.get("per_layer", {}).items():
            print(l, {k: x[k] for k in ("primary_valid", "primary_valid_dialogues", "rank_distribution", "S_reference_free")}, x["stability"])
    else:
        res, vecs = oracle(args.run)
        np.savez(out.with_name(out.stem + "_vectors.npz"), **vecs)
        res = jsonable(res)
        atomic_json(out, res)
        print(json.dumps({"label": res["label"], "decision": res["decision"], "region_quality": {k: v for k, v in res["region_quality"].items() if k not in ("counts_samples_heard",)}}, indent=1))
        for l, x in res["layers"].items():
            print(l, {k: x[k] for k in ("O", "P", "S", "I", "oracle_valid", "paired", "signal_evaluable")}, x["signed_cos_macro_adjusted"], x["advantage_macro_adjusted"])


if __name__ == "__main__":
    main()
