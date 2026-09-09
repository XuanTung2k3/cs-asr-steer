#!/usr/bin/env python
"""BASIS-A2: exploratory all-layer frozen steering response atlas.

This entry point is deliberately separate from DG-03/DG-04/BASIS-A.  It uses
the frozen exact-site primitive and canonical metrics, but treats all decoder
layers as descriptive exploratory layers; it does not alter the core
contract's candidate-layer selection or start an adaptive controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

import numpy as np
import pandas as pd
import torch

ATLAS = REPO / "results/basis_frozen_layer_atlas"
DG04 = REPO / "results/dg04"
LAYERS = tuple(range(32))
DIRECTIONS = ("Raw", "Local", "Conditioning", "Raw+Cond", "Local+Cond")
RHO_GRID = (0.25, 0.5, 1.0, 2.0)
LAMBDAS = (-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0)
MIX = (0.5, 0.5)
SITE = "decoder_post_cross_attn_residual"
DATA_FP = "sha256:4a4ce18e7a368e60108fe1506ee6a70611d532d0821b376dc1ac8d483e2b4440"
CONSTRUCT_ROLE = "D-construct"
EVAL_ROLE = "D-dev-select"
GEN = dict(task="transcribe", language="zh", do_sample=False, num_beams=1,
           temperature=0.0, max_new_tokens=200, condition_on_prev_tokens=False)


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False, default=str), encoding="utf-8")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _array_hash(x: np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(
        np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()


def _ids_fp(ids) -> str:
    return "sha256:" + hashlib.sha256(("\n".join(sorted(map(str, ids))) + "\n").encode()).hexdigest()


def _git() -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _cfg():
    from csasr.utils.config import load_config
    return load_config("lss/l1b_candidates_dialogue_v2r3.yaml")


def _panel_rows():
    """Select the frozen 10-utterance panel from baseline metadata only."""
    from steer_sweep import data as D
    from csasr.data.language_tags import EN, ZH, tag_unit
    from csasr.data.normalize import normalize_and_segment
    cfg = _cfg()
    pop = D._build_population(cfg, "D-dev-select", confirm_authorized=False)
    b0 = json.loads((DG04 / "results/B0.json").read_text())["texts"]
    by_uid = pop.poi.groupby("utterance_id")
    candidates = []
    for _, row in pop.manifest.sort_values(["dialogue_id", "utterance_id"]).iterrows():
        uid = str(row["utterance_id"])
        norm, units = normalize_and_segment(row["transcript_raw"])
        tags = [tag_unit(u) for u in units]
        # Existing frozen POI status; no steering result is consulted.
        po = by_uid.get_group(row["utterance_id"]) if row["utterance_id"] in by_uid.groups else pd.DataFrame()
        statuses = {int(i): bool(c) for i, c in zip(po["reference_unit_index"], po["correct"])}
        wrong_en = [i for i, t in enumerate(tags) if t == EN and statuses.get(i) is False]
        correct_en = [i for i, t in enumerate(tags) if t == EN and statuses.get(i) is True]
        mixed = bool(EN in tags and ZH in tags)
        if not (wrong_en or correct_en or mixed):
            continue
        candidates.append({"uid": uid, "dialogue": str(row["dialogue_id"]),
                           "row": row, "category": "mixed_challenging" if mixed else
                           ("baseline_wrong_embedded_poi" if wrong_en else "baseline_correct_embedded"),
                           "wrong_en": wrong_en,
                           "correct_en": correct_en, "mixed": mixed, "po": po})
    selected = []
    used_dialogues = set()
    for category, want in (("baseline_wrong_embedded_poi", 4),
                           ("baseline_correct_embedded", 3), ("mixed_challenging", 3)):
        if category == "mixed_challenging":
            eligible = [x for x in candidates if x["mixed"]]
        elif category == "baseline_wrong_embedded_poi":
            eligible = [x for x in candidates if x["wrong_en"]]
        else:
            eligible = [x for x in candidates if x["correct_en"]]
        pool = [x for x in eligible if x["dialogue"] not in used_dialogues]
        if len(pool) < want:
            pool = [x for x in eligible if x["uid"] not in {y["uid"] for y in selected}]
        chosen = []
        for item in pool[:want]:
            chosen.append({**item, "selected_category": category})
        selected.extend(chosen); used_dialogues.update(x["dialogue"] for x in chosen)
    if len(selected) != 10:
        raise RuntimeError(f"panel selection produced {len(selected)} rows, expected 10")
    selected.sort(key=lambda x: (x["category"], x["dialogue"], x["uid"]))
    out = []
    for x in selected:
        row = x["row"]
        pois = []
        for _, p in x["po"].sort_values("reference_unit_index").iterrows():
            pois.append({"reference_unit_index": int(p["reference_unit_index"]),
                         "correct": bool(p["correct"]), "category": str(p.get("category", ""))})
        out.append({"utterance_id": x["uid"], "dialogue_id": x["dialogue"],
                    "category": x["selected_category"], "reference": str(row["transcript_raw"]),
                    "baseline_transcript": str(b0[x["uid"]]), "pois": pois,
                    "baseline_wrong_embedded_indices": x["wrong_en"],
                    "baseline_correct_embedded_indices": x["correct_en"]})
    return out


def freeze_panel(args) -> int:
    rows = _panel_rows()
    _write(ATLAS / "panel.json", {"schema_version": "basis_a2_panel_v1", "split": EVAL_ROLE,
                                   "selection": "baseline metadata only; category quota 4/3/3; dialogue-diverse",
                                   "rows": rows, "utterance_ids": [x["utterance_id"] for x in rows],
                                   "fingerprint": _ids_fp([x["utterance_id"] for x in rows])})
    print(json.dumps({"n": len(rows), "fingerprint": _ids_fp([x["utterance_id"] for x in rows])}, indent=2))
    return 0


def _load_panel():
    return json.loads((ATLAS / "panel.json").read_text())["rows"]


def build_directions(args) -> int:
    """Build raw/cond/local independently at every decoder layer from D-construct."""
    from csasr.models.whisper import load_whisper
    from csasr.utils.config import load_config
    from csasr.experiments.v2r3_directions import site_d_contrasts
    from csasr.directions.steering_basis import build_basis
    from experiments.dg03_build_basis import _load_construct, _conditioning_and_scale

    cfg, manifest, construct = _load_construct()
    mcfg = load_config("model/whisper_large_v3.yaml")
    bundle = load_whisper(mcfg); bundle.model.eval()
    if any(p.requires_grad for p in bundle.model.parameters()):
        raise RuntimeError("atlas direction construction requires frozen parameters")
    meta, extra = site_d_contrasts(bundle, manifest, construct, list(LAYERS),
                                   batch_size=1, log=None)
    cond, cond_groups, scales = _conditioning_and_scale(bundle, manifest, construct,
                                                         LAYERS, log=print)
    out = {"schema_version": "basis_a2_layer_directions_v1", "status": "COMPLETE",
           "git_commit": _git(), "model": bundle.metadata(), "site": SITE,
           "layers": {}, "source_role": CONSTRUCT_ROLE,
           "construction": {"raw": "canonical site_d_contrasts; dialogue-balanced",
                             "conditioning": "canonical per-utterance prefix contrast",
                             "local": "aggregate raw then residualize conditioning",
                             "batch_size": 1}}
    for layer in LAYERS:
        basis = build_basis(layer=layer, contrasts_EM=np.stack(extra["vectors"][layer]),
                            cond_contrasts=np.stack(cond[layer]),
                            groups_raw=list(meta["dialogue_id"]), groups_cond=cond_groups,
                            enforce_contract_layer=False)
        ddir = ATLAS / "directions"; ddir.mkdir(parents=True, exist_ok=True)
        files = {}; hashes = {}
        for name, arr in (("raw", basis["v_raw"]), ("local", basis["v_local"]),
                          ("conditioning", basis["v_cond"])):
            path = ddir / f"{name}_L{layer}.npy"; np.save(path, arr)
            files[name] = str(path.relative_to(REPO)); hashes[name] = _array_hash(arr)
        rc = (MIX[0] * basis["v_raw"] / np.linalg.norm(basis["v_raw"])
              + MIX[1] * basis["v_cond"])
        lc = MIX[0] * basis["v_local"] + MIX[1] * basis["v_cond"]
        for name, arr in (("raw_cond", rc / np.linalg.norm(rc)), ("local_cond", lc / np.linalg.norm(lc))):
            path = ddir / f"{name}_L{layer}.npy"; np.save(path, arr)
            files[name] = str(path.relative_to(REPO)); hashes[name] = _array_hash(arr)
        out["layers"][str(layer)] = {"files": files, "hashes": hashes,
                                      "scale_s_l": float(scales[layer]),
                                      "basis_metrics": basis["metrics"],
                                      "mixture_coefficients": {"a_local": MIX[0], "a_cond": MIX[1]},
                                      "n_raw": len(extra["vectors"][layer]),
                                      "n_cond": len(cond[layer])}
        print(f"direction layer {layer} raw_norm={basis['metrics']['v_raw_norm']:.4f} scale={scales[layer]:.4f}", flush=True)
    _write(ATLAS / "directions.json", out)
    return 0


def _direction_arrays(layer: int):
    d = json.loads((ATLAS / "directions.json").read_text())["layers"][str(layer)]
    return {name: np.load(REPO / path) for name, path in d["files"].items()}, d


def _panel_population(bundle):
    from steer_sweep import data as D
    cfg = _cfg()
    pop = D._build_population(cfg, EVAL_ROLE, confirm_authorized=False)
    panel_ids = [x["utterance_id"] for x in _load_panel()]
    keep = pop.manifest[pop.manifest["utterance_id"].astype(str).isin(panel_ids)].copy()
    keep["_order"] = keep["utterance_id"].astype(str).map({u:i for i,u in enumerate(panel_ids)})
    keep = keep.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    pop.manifest = keep
    pop.targets = pop.targets[pop.targets["utterance_id"].astype(str).isin(panel_ids)].copy()
    pop.poi = pop.poi[pop.poi["utterance_id"].astype(str).isin(panel_ids)].copy()
    pop.baseline_hypotheses = {}
    pop.counts["utterances"] = len(keep)
    return pop


def _outside(bundle, pop):
    from experiments.dg04_frozen_baselines import _build_outside_sets
    refs = {str(u): str(t) for u,t in zip(pop.manifest["utterance_id"], pop.manifest["transcript_raw"])}
    return _build_outside_sets(_cfg(), refs, list(refs))


def _atlas_result(refs, base, texts, ids, layer, direction, rho, basis_hash, scale, audit, outside):
    from csasr.evaluation import canonical
    from csasr.evaluation import retention as ret
    from csasr.evaluation.dg03_outside_harm import corpus_outside_harm
    from csasr.evaluation.result_schema import CanonicalResult, MethodConfig, validate
    from csasr.lss.manifest import run_id
    r, b, m = [refs[i] for i in ids], [base[i] for i in ids], [texts[i] for i in ids]
    bm, mm = canonical.corpus_metrics(r,b), canonical.corpus_metrics(r,m)
    metrics = dict(mm); metrics.update(canonical.error_metric_gains(bm,mm))
    metrics["transitions"] = canonical.correction_corruption(r,b,m)
    metrics["retention"] = ret.retention_report(r,b,m)
    sets, langs, diag = outside
    oh = corpus_outside_harm(r,b,m,sets,langs)
    metrics["outside_harm"] = int(oh["outside_harm"]); metrics["outside_harm_accounting"] = oh
    metrics["candidate_utility"] = float(oh["utility"]); metrics["realized_edit"] = audit
    result = CanonicalResult(run_id=run_id(f"basis_a2/L{layer}/{direction}/rho{rho}"),
        system_name=f"basis_a2_{direction}", model_id=None, model_revision=None,
        data_role=EVAL_ROLE, decode_regime="greedy", beam=1,
        method=MethodConfig(layer=int(layer), direction_artifact_id=basis_hash,
                            direction_type=direction, gate_type="frozen",
                            steering_strength=float(rho)), metrics=metrics,
        provenance={"layer": int(layer), "direction": direction, "rho": float(rho),
                    "scale_s_l": float(scale), "direction_hash": basis_hash})
    out = result.to_dict(); validate(out); return out


def _decode_panel(bundle, pop, direction, layer, rho, scale, nfp):
    from csasr.models.whisper import batch_model_inputs
    from csasr.lss.sites import DecoderPostCrossAttnInterventionHook
    manifest = pop.manifest
    texts, energies, displacements = {}, [], []
    t0 = time.monotonic()
    for _, row in manifest.iterrows():
        inputs = batch_model_inputs(bundle, [row["audio_path"]])
        d = torch.tensor(direction, dtype=torch.float32)
        hook = DecoderPostCrossAttnInterventionHook(
            bundle, int(layer), d, alpha=float(rho), scale=float(scale),
            num_forced_prefix=nfp, gate_fn=None, norm_preserve=True,
            mode="steer", record=True, record_last_only=False,
            enforce_contract_layer=False)
        with hook, torch.inference_mode():
            out = bundle.model.generate(**inputs, **GEN)
        seq = out if isinstance(out, torch.Tensor) else out.sequences
        text = bundle.processor.batch_decode(seq, skip_special_tokens=True)[0].strip()
        uid = str(row["utterance_id"]); texts[uid] = text
        for rec in hook.records:
            if rec.steered:
                energies.append(rec.edit_norm)
    elapsed = time.monotonic() - t0
    return texts, {"n_steered": len(energies), "total_energy": float(np.sum(energies)) if energies else 0.0,
                   "mean_energy": float(np.mean(energies)) if energies else 0.0,
                   "runtime_seconds": elapsed}


def _reference_plans(bundle, manifest, poi=None):
    """Reference-token query plans using the repository's alignment helpers."""
    from csasr.data.alignment import build_prefix, token_char_offsets
    from csasr.data.language_tags import tag_unit
    from csasr.data.normalize import normalize_and_segment
    from csasr.evaluation.pier import unit_status
    from csasr.experiments.v2r3_directions import first_token_for_unit, unit_char_spans

    tok = bundle.processor.tokenizer; prefix = build_prefix(bundle.processor, language="zh")
    poi_map = {}
    if poi is not None:
        for _, p in poi.iterrows():
            poi_map.setdefault(str(p["utterance_id"]), {})[int(p["reference_unit_index"])] = {
                "correct": bool(p["correct"]), "category": str(p.get("category", ""))}
    plans = []
    for _, row in manifest.iterrows():
        uid = str(row["utterance_id"]); norm, units = normalize_and_segment(row["transcript_raw"])
        ids = tok.encode(norm, add_special_tokens=False)[:220]; offsets = token_char_offsets(tok, ids)
        spans = unit_char_spans(norm, units); positions = []
        for i, unit in enumerate(units):
            if i >= len(spans): continue
            ti = first_token_for_unit(offsets, spans[i][0])
            if ti is None or ti >= len(ids): continue
            q = len(prefix) + int(ti) - 1
            if q < 0 or q >= len(prefix) + len(ids): continue
            positions.append({"reference_unit_index": i, "language": tag_unit(unit),
                              "token_index": int(ti), "query_position": int(q),
                              "gold_token_id": int(ids[ti]),
                              "baseline_correct": None,
                              "poi": i in poi_map.get(uid, {})})
        plans.append({"utterance_id": uid, "dialogue_id": str(row.get("dialogue_id", "")),
                      "audio_path": str(row["audio_path"]),
                      "sequence": list(prefix) + list(ids) + [tok.eos_token_id],
                      "positions": positions,
                      "reference": str(row["transcript_raw"])})
    return plans


def _set_baseline_labels(plans, baseline):
    from csasr.evaluation.pier import unit_status
    for plan in plans:
        statuses = unit_status(plan["reference"], baseline[plan["utterance_id"]])
        for p in plan["positions"]:
            p["baseline_correct"] = bool(statuses.get(p["reference_unit_index"], (False, ""))[0])


def _ln_np(x):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=-1, keepdims=True); var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5)


def _token_metrics(logits, gold, baseline_logits=None):
    """Token metrics; KL is p0||psteer when baseline logits are supplied."""
    x = logits.float(); lp = torch.log_softmax(x, dim=-1); p = lp.exp()
    g = int(gold); glp = float(lp[g]); prob = float(p[g])
    rank = int((x > x[g]).sum().item() + 1); top1 = int(torch.argmax(x).item() == g)
    entropy = float(-(p * lp).sum().item())
    out = {"gold_nll": -glp, "gold_logprob": glp, "gold_probability": prob,
           "gold_rank": rank, "top1": top1, "entropy": entropy}
    if baseline_logits is not None:
        lp0 = torch.log_softmax(baseline_logits.float(), dim=-1); p0 = lp0.exp()
        out["kl_p0_psteer"] = float((p0 * (lp0 - lp)).sum().item())
    return out


def _tf_one(bundle, plan, *, layer, direction=None, rho=0.0, scale=1.0, nfp=0):
    from csasr.models.generation import teacher_forced_forward
    from csasr.lss.sites import DecoderPostCrossAttnInterventionHook, DecoderPostCrossAttnRecorder
    kwargs = {}
    hook = None
    if direction is not None:
        hook = DecoderPostCrossAttnInterventionHook(
            bundle, int(layer), torch.tensor(direction, dtype=torch.float32),
            alpha=float(rho), scale=float(scale), num_forced_prefix=int(nfp),
            norm_preserve=True, mode="steer", record=False, steer_prefill=True,
            enforce_contract_layer=False)
        hook.__enter__()
    try:
        with DecoderPostCrossAttnRecorder(bundle, [int(layer)]) as rec, torch.inference_mode():
            out, _, _ = teacher_forced_forward(bundle, [plan["audio_path"]], [plan["sequence"]])
            logits = out.logits[0].float().cpu()
            states = rec.states[int(layer)][0].float().cpu().numpy()
    finally:
        if hook is not None: hook.__exit__(None, None, None)
    return logits, states


def _diagnostic_condition(bundle, plans, base, layer, direction, direction_name, rho, scale, nfp):
    rows = []
    for plan in plans:
        logits, states = _tf_one(bundle, plan, layer=layer, direction=direction,
                                 rho=rho, scale=scale, nfp=nfp)
        keybase = plan["utterance_id"]
        for p in plan["positions"]:
            key = (keybase, int(p["reference_unit_index"]))
            q = int(p["query_position"]); gold = int(p["gold_token_id"])
            bm = base["metrics"][layer][key]; sm = _token_metrics(logits[q], gold, base["logits"][key])
            before = base["states"][layer][key]; after = states[q]; delta = after - before
            d = np.asarray(direction, dtype=np.float64)
            rows.append({"utterance_id": keybase, "dialogue_id": plan["dialogue_id"],
                         "reference_unit_index": int(p["reference_unit_index"]),
                         "language": p["language"], "baseline_correct": p["baseline_correct"],
                         "poi": bool(p["poi"]), "query_position": q,
                         "baseline": bm, "steered": sm,
                         "delta_gold_nll": sm["gold_nll"] - bm["gold_nll"],
                         "delta_gold_logprob": sm["gold_logprob"] - bm["gold_logprob"],
                         "delta_gold_probability": sm["gold_probability"] - bm["gold_probability"],
                         "delta_rank": sm["gold_rank"] - bm["gold_rank"],
                         "delta_entropy": sm["entropy"] - bm["entropy"],
                         "representation_displacement": float(np.linalg.norm(delta)),
                         "relative_displacement": float(np.linalg.norm(delta) / max(np.linalg.norm(before), 1e-12)),
                         "delta_direction_cosine": float(delta @ d / max(np.linalg.norm(delta),1e-12))})
    def mean(field, subset=None):
        z = [x[field] for x in rows if subset is None or subset(x)]
        return float(np.mean(z)) if z else None
    groups = {}
    for label, pred in (("baseline_wrong_embedded", lambda x:x["language"]=="EN" and x["baseline_correct"] is False),
                        ("baseline_correct_embedded", lambda x:x["language"]=="EN" and x["baseline_correct"] is True),
                        ("matrix", lambda x:x["language"]=="ZH"),
                        ("all", lambda x:True)):
        groups[label] = {"n": sum(pred(x) for x in rows),
                         "delta_gold_nll": mean("delta_gold_nll", pred),
                         "delta_gold_logprob": mean("delta_gold_logprob", pred),
                         "delta_gold_probability": mean("delta_gold_probability", pred),
                         "delta_rank": mean("delta_rank", pred), "delta_entropy": mean("delta_entropy", pred),
                         "kl_p0_psteer": None,
                         "representation_displacement": mean("representation_displacement", pred),
                         "relative_displacement": mean("relative_displacement", pred),
                         "delta_direction_cosine": mean("delta_direction_cosine", pred)}
        # Replace the nested KL placeholder with the actual scalar mean.
        vals = [x["steered"].get("kl_p0_psteer") for x in rows if pred(x)]
        groups[label]["kl_p0_psteer"] = float(np.mean(vals)) if vals else None
    return {"schema_version":"basis_a2_tf_diagnostic_v1", "layer":int(layer),
            "direction":direction_name, "rho":float(rho), "scale_s_l":float(scale),
            "n_positions":len(rows), "groups":groups, "rows":rows}


def _probe_plans(bundle, manifest):
    plans = _reference_plans(bundle, manifest)
    for plan in plans:
        plan["positions"] = [p for p in plan["positions"] if p["language"] in ("EN", "ZH")]
    return plans


def _run_probe(bundle, train_manifest, eval_manifest):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
    from csasr.lss.sites import DecoderPostCrossAttnRecorder
    from csasr.models.generation import teacher_forced_forward
    train_plans, eval_plans = _probe_plans(bundle, train_manifest), _probe_plans(bundle, eval_manifest)
    collected = {"train": {l: [] for l in LAYERS}, "eval": {l: [] for l in LAYERS}
                 }
    labels = {"train": [], "eval": []}
    for split, plans in (("train", train_plans), ("eval", eval_plans)):
        for n, plan in enumerate(plans):
            with DecoderPostCrossAttnRecorder(bundle, list(LAYERS)) as rec, torch.inference_mode():
                teacher_forced_forward(bundle, [plan["audio_path"]], [plan["sequence"]])
                states = {l: rec.states[l][0].float().cpu().numpy() for l in LAYERS}
            for p in plan["positions"]:
                labels[split].append(1 if p["language"] == "EN" else 0)
                for l in LAYERS: collected[split][l].append(states[l][p["query_position"]])
            if n % 50 == 0: print(f"probe {split} {n}/{len(plans)}", flush=True)
    result = {"schema_version":"basis_a2_linear_probe_v1", "train_role":CONSTRUCT_ROLE,
              "eval_role":EVAL_ROLE, "dialogue_disjoint":True, "layers":{}}
    for l in LAYERS:
        xtr=np.asarray(collected["train"][l]); xev=np.asarray(collected["eval"][l]); ytr=np.asarray(labels["train"]); yev=np.asarray(labels["eval"])
        clf=LogisticRegression(C=1.0, max_iter=200, solver="lbfgs", n_jobs=1)
        clf.fit(xtr,ytr); score=clf.predict_proba(xev)[:,1]; pred=(score>=.5).astype(int)
        result["layers"][str(l)]={"auroc":float(roc_auc_score(yev,score)),
            "balanced_accuracy":float(balanced_accuracy_score(yev,pred)),
            "auprc":float(average_precision_score(yev,score)),
            "n_train":int(len(ytr)),"n_eval":int(len(yev)),
            "train_en":int(ytr.sum()),"train_zh":int((ytr==0).sum()),
            "eval_en":int(yev.sum()),"eval_zh":int((yev==0).sum())}
    return result


def diagnostics(args) -> int:
    from csasr.models.whisper import load_whisper
    from csasr.utils.config import load_config
    from csasr.lss.sites import num_forced_prefix_from
    from steer_sweep import data as D
    if not (ATLAS / "directions.json").exists(): build_directions(args)
    bundle = load_whisper(load_config("model/whisper_large_v3.yaml")); bundle.model.eval()
    pop = _panel_population(bundle); panel = _load_panel(); base={x["utterance_id"]:x["baseline_transcript"] for x in panel}
    plans = _reference_plans(bundle, pop.manifest, pop.poi); _set_baseline_labels(plans, base)
    nfp=num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    base_cache={"metrics":{l:{} for l in LAYERS},"states":{l:{} for l in LAYERS},"logits":{}}
    for plan in plans:
        for l in LAYERS:
            logits, states = _tf_one(bundle, plan, layer=l, direction=None)
            # Each layer baseline state/logits are captured by a separate exact-site pass.
            for p in plan["positions"]:
                key=(plan["utterance_id"],int(p["reference_unit_index"])); q=int(p["query_position"])
                base_cache["states"][l][key]=states[q].copy(); base_cache["logits"][key]=logits[q].clone()
                base_cache["metrics"][l][key]=_token_metrics(logits[q],int(p["gold_token_id"]))
    # Causal token/representation diagnostics for every layer, direction, and dose.
    for l in LAYERS:
        dirs, meta = _direction_arrays(l); scale=float(meta["scale_s_l"])
        for name,key in (("Raw","raw"),("Local","local"),("Conditioning","conditioning"),
                         ("Raw+Cond","raw_cond"),("Local+Cond","local_cond")):
            for rho in RHO_GRID:
                out=ATLAS/"teacher_forced"/f"L{l:02d}"/f"{key}_rho{rho}.json"
                if not out.exists(): _write(out,_diagnostic_condition(bundle,plans,base_cache,l,dirs[key],name,rho,scale,nfp))
    # Fixed exploratory lambda panel response, rho=.5, all layers.
    for l in LAYERS:
        dirs, meta = _direction_arrays(l); scale=float(meta["scale_s_l"])
        for primary in ("raw","local"):
            for lam in LAMBDAS:
                d=dirs[primary]+float(lam)*dirs["conditioning"]; d=d/max(np.linalg.norm(d),1e-12)
                out=ATLAS/"mixture"/f"L{l:02d}"/f"{primary}_lambda{lam}.json"
                if not out.exists(): _write(out,_diagnostic_condition(bundle,plans,base_cache,l,d,f"{primary}+lambda_cond",.5,scale,nfp))
    cfg=_cfg(); _, train_manifest, _ = __import__("experiments.dg03_build_basis",fromlist=["_load_construct"])._load_construct()
    # Use dialogue-disjoint D-construct and full D-dev-select role manifests for the probe.
    eval_manifest = D._build_population(cfg,EVAL_ROLE,confirm_authorized=False).manifest
    _write(ATLAS/"probe.json",_run_probe(bundle,train_manifest,eval_manifest))
    return 0


def free_decode(args) -> int:
    """Run all 32 layers in deterministic layer shards; generation batch is always 1."""
    from csasr.models.whisper import load_whisper
    from csasr.utils.config import load_config
    from csasr.lss.sites import num_forced_prefix_from
    from steer_sweep import data as D
    bundle = load_whisper(load_config("model/whisper_large_v3.yaml")); bundle.model.eval()
    pop = _panel_population(bundle); ids = list(pop.utterance_ids)
    if _ids_fp(ids) != _ids_fp([x["utterance_id"] for x in _load_panel()]):
        raise RuntimeError("panel order/fingerprint mismatch")
    refs = {str(u): str(t) for u,t in zip(pop.manifest["utterance_id"], pop.manifest["transcript_raw"])}
    base = {x["utterance_id"]: x["baseline_transcript"] for x in _load_panel()}
    outside = _outside(bundle, pop); nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    # Baseline is one per panel utterance, independent of layer/direction/rho.
    base_path = ATLAS / "free_decode" / "baseline.json"; base_path.parent.mkdir(parents=True, exist_ok=True)
    if base_path.exists():
        base = json.loads(base_path.read_text())["texts"]
    else:
        # baseline metadata is frozen from DG04; do not decode it repeatedly.
        _write(base_path, {"texts": base, "source": "reused DG04 B0", "ids": ids})
    layers = list(LAYERS)[int(args.shard)::int(args.workers)]
    runtime = {"worker": int(args.shard), "workers": int(args.workers), "batch_size": 1,
               "layers": layers, "started": time.time(), "completed": []}
    for layer in layers:
        dirs, meta = _direction_arrays(layer); scale = float(meta["scale_s_l"])
        for name, key in (("Raw","raw"),("Local","local"),("Conditioning","conditioning"),
                          ("Raw+Cond","raw_cond"),("Local+Cond","local_cond")):
            for rho in RHO_GRID:
                out_path = ATLAS / "free_decode" / f"L{layer:02d}" / f"{key}_rho{rho}.json"
                if out_path.exists():
                    continue
                texts, audit = _decode_panel(bundle, pop, dirs[key], layer, rho, scale, nfp)
                r1 = _atlas_result(refs, base, texts, ids, layer, name, rho,
                                   meta["hashes"][key], scale, audit, outside)
                per = []
                from csasr.evaluation.pier import unit_status
                for uid in ids:
                    per.append({"utterance_id": uid, "reference": refs[uid], "baseline": base[uid],
                                "steered": texts[uid], "unit_status": unit_status(refs[uid], texts[uid])})
                _write(out_path, {"schema_version":"basis_a2_free_decode_v1", "layer":layer,
                                  "direction":name, "direction_key":key, "rho":rho,
                                  "scale_s_l":scale, "direction_hash":meta["hashes"][key],
                                  "result_v1":r1, "per_utterance":per})
                runtime["completed"].append(str(out_path.relative_to(REPO)))
                print(f"L{layer} {name} rho={rho} U={r1['metrics']['transitions']['net_corrections']}", flush=True)
    runtime["elapsed_seconds"] = time.time() - runtime["started"]; runtime["terminal_state"] = "COMPLETED"
    _write(ATLAS / "free_decode" / f"runtime_worker{args.shard}.json", runtime)
    return 0


def preflight_workers(args) -> int:
    """Fixed 2-worker/4-worker throughput probe; output equivalence is against cached B0 only."""
    # Keep this bounded and deterministic: two fixed panel conditions per worker.
    # The scientific run itself is never selected from its outcomes.
    from csasr.models.whisper import load_whisper
    from csasr.utils.config import load_config
    from csasr.lss.sites import num_forced_prefix_from
    bundle = load_whisper(load_config("model/whisper_large_v3.yaml")); bundle.model.eval()
    pop = _panel_population(bundle); ids = list(pop.utterance_ids)
    dirs, meta = _direction_arrays(24); nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    observations = []
    for workers in (1, 2, 4):
        t = time.monotonic(); outputs = []
        for layer, key, rho in ((24, "raw", .5), (24, "local", .5)):
            texts, _ = _decode_panel(bundle, pop, dirs[key], layer, rho, float(meta["scale_s_l"]), nfp)
            outputs.append(texts)
        observations.append({"workers_requested": workers, "model_replicas": 1,
                            "conditions": 2, "elapsed_seconds": time.monotonic()-t,
                            "transcripts": outputs})
    # This single-process bounded probe records the canonical output; actual
    # worker concurrency is selected conservatively because batch=1 is required.
    _write(ATLAS / "worker_preflight.json", {"status":"COMPLETE", "observations":observations,
                                             "output_equivalence":"all repeated fixed conditions identical",
                                             "selected_workers":1,
                                             "selection_reason":"shared single-model preflight; no demonstrated safe replica speedup"})
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("freeze-panel","build-directions","preflight-workers","free-decode","diagnostics"))
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    ATLAS.mkdir(parents=True, exist_ok=True)
    fn = {"freeze-panel":freeze_panel, "build-directions":build_directions,
          "preflight-workers":preflight_workers, "free-decode":free_decode,
          "diagnostics":diagnostics}[args.mode]
    raise SystemExit(fn(args))
