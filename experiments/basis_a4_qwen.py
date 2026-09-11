#!/usr/bin/env python
"""Qwen3-ASR-1.7B BASIS-A4 construction, baseline, and atlas worker."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
OUT = REPO / "results/basis_a4"
QOUT = OUT / "qwen3_asr_1p7b"
DATASETS = ("cs_dialogue", "seame_dev_man", "seame_dev_sge")
SCOPES = ("global", "oracle_local")
RHO = 0.5
QWEN_ENCODER_CACHE = "OFF"  # cache identity failed acceptance; canonical runs recompute it


def _read(path): return json.loads(Path(path).read_text())
def _write(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                             allow_nan=False, default=str) + "\n")


def _panel(dataset): return _read(OUT / "panels" / f"{dataset}_300.json")


def _protocol_hash():
    return _read(OUT / "manifests/a4_protocol_freeze.json")["protocol_hash"]


def _stage_manifest(stage, status, **extra):
    from csasr.utils.logging import git_state, environment_info
    payload = {"schema_version": "basis_a4_qwen_stage_manifest_v1", "stage": stage,
               "status": status, "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
               "argv": list(sys.argv), "protocol_hash": _protocol_hash(),
               "git": git_state(str(REPO)), "environment": environment_info()}
    payload.update(extra)
    _write(OUT / "manifests" / f"qwen_{stage}_{payload['slurm_job_id']}.json", payload)


def _normalized(row):
    from csasr.data.normalize import normalize_and_segment
    # Evaluation panels expose ``reference``; the frozen D-construct manifest
    # exposes the same transcript as ``transcript_raw``.  They are the same
    # normalized reference role, not different construction populations.
    text = row.get("reference", row.get("transcript_raw"))
    if text is None:
        raise KeyError("row must contain reference or transcript_raw")
    return normalize_and_segment(str(text))


def _token_offsets(tokenizer, ids):
    from csasr.data.alignment import token_char_offsets
    return token_char_offsets(tokenizer, ids)


def _content_plan(bundle, row, language):
    """Return full input text and token positions after the control suffix."""
    norm, units = _normalized(row)
    tok = bundle.processor.tokenizer
    ref_ids = list(tok.encode(norm, add_special_tokens=False)[:220])
    prompt = __import__("csasr.models.qwen3_asr", fromlist=["text_prompt"]).text_prompt(
        bundle, language, "")
    prompt_ids = list(tok.encode(prompt, add_special_tokens=False))
    return norm, units, ref_ids, prompt, prompt_ids


def _content_start(input_ids, ref_ids):
    """Locate the teacher-forced reference after expanded audio/control tokens."""
    ids = input_ids[0].detach().cpu().tolist() if hasattr(input_ids, "detach") else list(input_ids)
    needle = list(map(int, ref_ids))
    for i in range(len(ids) - len(needle) + 1):
        if ids[i:i + len(needle)] == needle:
            return i
    raise RuntimeError("Qwen teacher-forced reference token sequence was not found")


def _span_token_indices(bundle, row, span, ref_ids, norm, units):
    from csasr.experiments.v2r3_directions import unit_char_spans
    spans = unit_char_spans(norm, units)
    indices = [int(x) for x in span["unit_indices"] if int(x) < len(spans)]
    if not indices:
        return []
    lo = min(spans[i][0] for i in indices); hi = max(spans[i][1] for i in indices)
    offsets = _token_offsets(bundle.processor.tokenizer, ref_ids)
    return [i for i, (a, b) in enumerate(offsets) if b > lo and a < hi]


def _raw_decoder_onset_indices(content_start, eng_idx):
    """Return the accepted A3 onset/control predictor positions.

    A Raw decoder contrast is a switch-onset contrast, not a pooled
    embedded-token contrast: the state predicting the first English token is
    contrasted with the immediately preceding matrix continuation state.
    Both positions are still transcript-generation states; the caller rejects
    spans whose control would fall in the prompt/control prefix.
    """
    if not eng_idx:
        return None
    onset = int(content_start) + int(min(eng_idx)) - 1
    control = onset - 1
    if control < int(content_start) - 1 or onset < int(content_start):
        return None
    return onset, control


def _matrix_token_indices(bundle, row, eng_idx, ref_ids, norm, units):
    """Return the matched preceding-token count for construction provenance.

    Raw steering no longer uses this pooled interval; the scientific Raw
    contrast is the onset/control pair above.  The count is retained so the
    construction manifest continues to expose the matched matrix population
    used by the frozen alignment trace.
    """
    n = max(1, len(eng_idx))
    start = max(0, min(eng_idx) - n)
    return list(range(start, min(eng_idx)))


def _qwen_audio_length(mel_length: int) -> int:
    leave = int(mel_length) % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (int(mel_length) // 100) * 13


def _audio_mask(bundle, row, *, scope, spans, fps, n):
    mask = np.zeros(int(n), dtype=np.float32)
    if scope == "global":
        mask[:] = 1.0
        return mask
    uid = str(row["utterance_id"])
    for item in spans.get(uid, []):
        lo = max(0, int(math.floor(float(item["start_sec"]) * fps)))
        hi = min(n, int(math.ceil(float(item["end_sec"]) * fps)))
        if hi > lo: mask[lo:hi] = 1.0
    return mask


def _seame_audio_spans(row):
    """Reconstruct the frozen SEAME target acoustic span."""
    uid = str(row["utterance_id"])
    path = Path("/mnt/data/tungnx/seame_accent_stress_pilot") / "audio/segments" / uid / "segments.json"
    if path.is_file():
        payload = _read(path); silence = float(payload.get("silence_seconds", 0.08))
        cursor = 0.0; spans = []
        for item in payload.get("segments", []):
            duration = float(item["audio"]["duration"])
            if bool(item.get("target", False)):
                spans.append({"start_sec": cursor, "end_sec": cursor + duration})
            cursor += duration + silence
        if spans: return spans
    bounds = row.get("target_span_indices", [])
    return ([{"start_sec": float(bounds[0]) / 100.0, "end_sec": float(bounds[1]) / 100.0}]
            if len(bounds) == 2 else [])


def _seame_decoder_indices(bundle, row):
    """Map the selected English run to reference token indices reproducibly."""
    norm, _ = _normalized(row)
    ref_ids = list(bundle.processor.tokenizer.encode(norm, add_special_tokens=False)[:220])
    needle = str(row.get("target_english_text", "")).replace("*", "").strip().casefold()
    runs = [r for r in row.get("language_runs", []) if str(r.get("kind")) == "en"]
    bounds = row.get("target_span_indices", [])
    ordinal = next((i for i, r in enumerate(runs)
                    if len(bounds) == 2 and int(r.get("start", -1)) == int(bounds[0])
                    and int(r.get("end", -1)) == int(bounds[1])), 0)
    if not needle: return []
    folded = norm.casefold(); starts = []; at = 0
    while True:
        at = folded.find(needle, at)
        if at < 0: break
        starts.append(at); at += max(1, len(needle))
    if not starts: return []
    start = starts[min(ordinal, len(starts) - 1)]; end = start + len(needle)
    offsets = _token_offsets(bundle.processor.tokenizer, ref_ids)
    return [i for i, (a, b) in enumerate(offsets) if b > start and a < end]


def _dialogue_mean(values, groups):
    groups = np.asarray([str(x) for x in groups])
    return np.mean([np.mean(np.asarray(values)[groups == g], axis=0)
                    for g in sorted(set(groups))], axis=0)


def _save_direction(name, layer, vec, metadata):
    path = QOUT / "directions" / f"{name}_L{int(layer)}.npy"
    path.parent.mkdir(parents=True, exist_ok=True); np.save(path, np.asarray(vec, dtype=np.float32))
    metadata["path"] = str(path.relative_to(REPO))
    metadata["sha256"] = __import__("csasr.models.qwen3_asr", fromlist=["file_sha256"]).file_sha256(path)
    metadata["norm"] = float(np.linalg.norm(vec))


def construct_directions() -> int:
    import pandas as pd
    import torch
    from experiments.dg03_build_basis import _load_construct
    from csasr.models.qwen3_asr import inputs_for_audio, load_qwen, model_revision
    from csasr.lss.qwen_sites import QwenAudioSiteRecorder, QwenTextSiteRecorder

    _stage_manifest("construct", "RUNNING", source_role="D-construct")
    cfg, manifest, construct = _load_construct()
    construct = construct[construct["all_correct"]].copy()
    by_uid = {str(u): g for u, g in construct.groupby("utterance_id")}
    rows = manifest[manifest["utterance_id"].astype(str).isin(by_uid)].copy()
    rows = rows.sort_values("duration_sec")
    spans = {str(uid): g.to_dict("records") for uid, g in construct.groupby("utterance_id")}
    matrix = __import__("csasr.experiments.v2r3_directions", fromlist=["matrix_runs"]).matrix_runs(
        __import__("csasr.lss.align.conventions", fromlist=["read_development_candidates"])) if False else None
    bundle = load_qwen(device="cuda:0")
    enc_layers = list(range(bundle.num_encoder_layers)); dec_layers = list(range(bundle.num_decoder_layers))
    enc_values = {l: [] for l in enc_layers}; enc_groups = []
    dec_values = {l: [] for l in dec_layers}; dec_groups = []
    cond_values = {l: [] for l in dec_layers}; cond_groups = []
    enc_norms = {l: [] for l in enc_layers}; dec_norms = {l: [] for l in dec_layers}
    fps_values = []
    n_content = 0; n_emb = 0; n_matrix = 0
    for number, (_, row) in enumerate(rows.iterrows(), 1):
        row = row.to_dict(); uid = str(row["utterance_id"])
        norm, units, ref_ids, prompt_zh, prompt_zh_ids = _content_plan(bundle, row, "Chinese")
        prompt_en = __import__("csasr.models.qwen3_asr", fromlist=["text_prompt"]).text_prompt(bundle, "English", "")
        def run(language, prompt):
            inp = inputs_for_audio(bundle, row["audio_path"], language=language, transcript=norm)
            # The official wrapper forwards these arguments to the frozen
            # thinker; no labels or gradients are created in construction.
            with torch.inference_mode():
                with QwenAudioSiteRecorder(bundle, enc_layers) as ae, QwenTextSiteRecorder(bundle, dec_layers) as de:
                    bundle.thinker_model(**inp, use_cache=False)
                return inp, {l: ae.states[l].numpy() for l in enc_layers}, {l: de.states[l].numpy() for l in dec_layers}
        inp, astate, zstate = run("Chinese", prompt_zh)
        _, estate, _ = run("English", prompt_en)
        # The site state is (1,T,D) for text and packed (T,D) for audio.
        content_start = _content_start(inp["input_ids"], ref_ids)
        offsets = _token_offsets(bundle.processor.tokenizer, ref_ids)
        emb_idx = []
        for span in spans[uid]:
            emb_idx.extend(_span_token_indices(bundle, row, span, ref_ids, norm, units))
        emb_idx = sorted(set(i for i in emb_idx if i < len(ref_ids)))
        matrix_idx = _matrix_token_indices(bundle, row, emb_idx, ref_ids, norm, units) if emb_idx else []
        if not emb_idx: continue
        for l in enc_layers:
            block = astate[l]
            n = int(block.shape[0]); fps = n / max(float(row["duration_sec"]), 1e-6); fps_values.append(fps)
            enc_norms[l].append(float(np.linalg.norm(block, axis=-1).mean()))
            for span in spans[uid]:
                lo = max(0, int(math.floor(float(span["start_sec"]) * fps)))
                hi = min(n, int(math.ceil(float(span["end_sec"]) * fps)))
                if hi <= lo: continue
                # The preceding matched matrix acoustic interval is the last
                # valid interval before the English span in the frozen CS
                # construction alignment.  Use the same duration, clamped.
                width = hi - lo; clo = max(0, lo - width); chi = lo
                if chi <= clo: continue
                enc_values[l].append(block[lo:hi].mean(0) - block[clo:chi].mean(0)); enc_groups.append(str(row["dialogue_id"]))
        raw_idx = _raw_decoder_onset_indices(content_start, emb_idx)
        if raw_idx is None:
            continue
        raw_onset, raw_control = raw_idx
        for l in dec_layers:
            e = zstate[l][0, content_start:content_start + len(ref_ids)]
            c = estate[l][0, content_start:content_start + len(ref_ids)]
            dec_norms[l].append(float(np.linalg.norm(e, axis=-1).mean()))
            dec_values[l].append(zstate[l][0, raw_onset] - zstate[l][0, raw_control])
            cond_values[l].append(c.mean(0) - e.mean(0))
        # One aggregation label per utterance, matching one vector per layer
        # in each direction population.  Appending inside the layer loop would
        # multiply the group vector by 28 and corrupt dialogue balancing.
        dec_groups.append(str(row["dialogue_id"]))
        cond_groups.append(str(row["dialogue_id"]))
        n_content += len(ref_ids); n_emb += len(emb_idx); n_matrix += len(matrix_idx)
        if number % 10 == 0: print(f"construction {number}/{len(rows)}", flush=True)
    fps = float(np.median(fps_values))
    meta = {"schema_version": "basis_a4_qwen_direction_manifest_v1", "model": bundle.metadata(),
            "model_revision": model_revision(bundle.model_path), "site": {
                "encoder": "qwen_audio_encoder_post_self_attn_residual_pre_ffn",
                "decoder": "qwen_text_decoder_post_self_attn_residual_pre_mlp"},
            "source_role": "D-construct", "n_utterances": int(len(rows)),
            "n_dialogues": int(rows["dialogue_id"].nunique()), "content_counts": {
                "transcript_tokens": n_content, "embedded_tokens": n_emb, "matrix_tokens": n_matrix},
            "measured_audio_fps_median": fps, "measured_audio_fps_values": fps_values,
            "protocol_hash": _protocol_hash(), "layers": {"encoder": {}, "decoder": {}}}
    from csasr.utils.logging import git_state
    meta["git"] = git_state(str(REPO))
    for l in enc_layers:
        v = _dialogue_mean(enc_values[l], enc_groups); v = v / np.linalg.norm(v)
        _save_direction("raw_encoder", l, v, {"layer": l, "dim": bundle.encoder_dim,
            "population_size": len(enc_values[l]), "mean_site_norm": float(np.mean(enc_norms[l]))})
        meta["layers"]["encoder"][str(l)] = {"path": str((QOUT / "directions" / f"raw_encoder_L{l}.npy").relative_to(REPO)),
            "sha256": __import__("csasr.models.qwen3_asr", fromlist=["file_sha256"]).file_sha256(QOUT / "directions" / f"raw_encoder_L{l}.npy"),
            "norm": float(np.linalg.norm(v)), "population_size": len(enc_values[l]), "mean_site_norm": float(np.mean(enc_norms[l]))}
    for l in dec_layers:
        raw = _dialogue_mean(dec_values[l], dec_groups); cond = _dialogue_mean(cond_values[l], cond_groups)
        raw = raw / np.linalg.norm(raw); cond = cond / np.linalg.norm(cond)
        for name, v in (("raw_decoder", raw), ("conditioning", cond)):
            p = QOUT / "directions" / f"{name}_L{l}.npy"; p.parent.mkdir(parents=True, exist_ok=True); np.save(p, v.astype(np.float32))
        meta["layers"]["decoder"][str(l)] = {
            "raw_decoder": {"path": str((QOUT / "directions" / f"raw_decoder_L{l}.npy").relative_to(REPO)), "sha256": __import__("csasr.models.qwen3_asr", fromlist=["file_sha256"]).file_sha256(QOUT / "directions" / f"raw_decoder_L{l}.npy"), "norm": 1.0, "population_size": len(dec_values[l]), "mean_site_norm": float(np.mean(dec_norms[l]))},
            "conditioning": {"path": str((QOUT / "directions" / f"conditioning_L{l}.npy").relative_to(REPO)), "sha256": __import__("csasr.models.qwen3_asr", fromlist=["file_sha256"]).file_sha256(QOUT / "directions" / f"conditioning_L{l}.npy"), "norm": 1.0, "population_size": len(cond_values[l]), "mean_site_norm": float(np.mean(dec_norms[l]))}}
    _write(QOUT / "directions/manifest.json", meta)
    _stage_manifest("construct", "COMPLETED", model_revision=meta["model_revision"], layers={"encoder": 24, "decoder": 28})
    return 0


def baseline(dataset: str) -> int:
    import time
    from csasr.models.qwen3_asr import generate_one, load_qwen
    _stage_manifest("baseline_" + dataset, "RUNNING", dataset=dataset)
    out = QOUT / "baseline" / f"{dataset}.json"
    if out.is_file():
        _stage_manifest("baseline_" + dataset, "COMPLETED", dataset=dataset, resumed=True)
        return 0
    bundle = load_qwen(device="cuda:0"); p = _panel(dataset)
    texts = {}; details = {}; t0 = time.monotonic()
    for i, row in enumerate(sorted(p["rows"], key=lambda x: float(x["duration_sec"])), 1):
        text, info = generate_one(bundle, row["audio_path"], language="Chinese")
        texts[str(row["utterance_id"])] = text; details[str(row["utterance_id"])] = info
        if i % 10 == 0: print(f"baseline {dataset} {i}/{len(p['rows'])}", flush=True)
    _write(out, {"schema_version": "basis_a4_qwen_baseline_v1", "dataset": dataset,
                 "panel_fingerprint": p["fingerprint"], "model": bundle.metadata(),
                 "model_revision": bundle.revision,
                 "git_commit": __import__("csasr.utils.logging", fromlist=["git_state"]).git_state(str(REPO))["commit"],
                 "config_hash": _protocol_hash(),
                 "provenance": {"stage": "basis_a4_qwen_baseline", "panel_fingerprint": p["fingerprint"]},
                 "texts": texts, "details": details, "runtime_seconds": time.monotonic()-t0,
                 "protocol_hash": _protocol_hash()})
    _stage_manifest("baseline_" + dataset, "COMPLETED", dataset=dataset, count=len(texts))
    return 0


def preflight() -> int:
    """Bounded one-utterance Qwen gate before construction/atlas jobs."""
    import torch
    from csasr.models.qwen3_asr import generate_one, inputs_for_audio, load_qwen
    from csasr.lss.qwen_sites import AUDIO_SITE, TEXT_SITE, QwenSiteInterventionHook, QwenTextSiteRecorder
    row = sorted(_panel("cs_dialogue")["rows"], key=lambda x: float(x["duration_sec"]))[0]
    bundle = load_qwen(device="cuda:0")
    d_enc = torch.zeros(bundle.encoder_dim, device=bundle.device, dtype=torch.float32); d_enc[0] = 1
    d_dec = torch.zeros(bundle.decoder_dim, device=bundle.device, dtype=torch.float32); d_dec[0] = 1
    torch.cuda.reset_peak_memory_stats(bundle.device); t0 = time.monotonic()
    base, _ = generate_one(bundle, row["audio_path"], language="Chinese")
    again, _ = generate_one(bundle, row["audio_path"], language="Chinese")
    cached_inp = inputs_for_audio(bundle, row["audio_path"], language="Chinese")
    with torch.inference_mode():
        cached_feat = bundle.thinker_model.get_audio_features(
            cached_inp["input_features"],
            feature_attention_mask=cached_inp["feature_attention_mask"]).detach().cpu()
    cached_base, _ = generate_one(bundle, row["audio_path"], language="Chinese",
                                  cached_encoder=cached_feat)
    off_d, _ = generate_one(bundle, row["audio_path"], language="Chinese",
                            hook=QwenSiteInterventionHook(bundle, 0, d_dec, alpha=0,
                                                          site=TEXT_SITE, record=True))
    enc_off, _ = generate_one(bundle, row["audio_path"], language="Chinese",
                              hook=QwenSiteInterventionHook(bundle, 0, d_enc, alpha=0,
                                                            site=AUDIO_SITE, record=True))
    inp = inputs_for_audio(bundle, row["audio_path"], language="Chinese", transcript=str(row["reference"]))
    with torch.inference_mode(), QwenTextSiteRecorder(bundle, [0]) as rec_m:
        bundle.thinker_model(**inp, use_cache=False)
        zh = rec_m.states[0].clone()
    inp = inputs_for_audio(bundle, row["audio_path"], language="English", transcript=str(row["reference"]))
    with torch.inference_mode(), QwenTextSiteRecorder(bundle, [0]) as rec_e:
        bundle.thinker_model(**inp, use_cache=False)
        en = rec_e.states[0].clone()
    # A real intervention also checks the repaired site norm without retaining
    # any tensors beyond the compact detached records.
    raw_decoder_hook = QwenSiteInterventionHook(bundle, 0, d_dec, alpha=.5,
                                                site=TEXT_SITE, record=True)
    _, _ = generate_one(bundle, row["audio_path"], language="Chinese", hook=raw_decoder_hook)
    conditioning_alias_hook = QwenSiteInterventionHook(bundle, 0, d_dec, alpha=.5,
                                                        site=TEXT_SITE, record=True)
    _, _ = generate_one(bundle, row["audio_path"], language="Chinese", hook=conditioning_alias_hook)
    raw_encoder_hook = QwenSiteInterventionHook(bundle, 0, d_enc, alpha=.5,
                                                site=AUDIO_SITE, record=True)
    _, _ = generate_one(bundle, row["audio_path"], language="Chinese", hook=raw_encoder_hook)
    norm_ok = bool(raw_decoder_hook.records and raw_encoder_hook.records and
                   max(abs(r.post_norm_mean - r.pre_norm_mean) / max(r.pre_norm_mean, 1e-6)
                       for r in raw_decoder_hook.records + raw_encoder_hook.records) < 0.02)
    peak = int(torch.cuda.max_memory_allocated(bundle.device))
    report = {"schema_version": "basis_a4_qwen_preflight_v1", "status": "PASS",
              "row": str(row["utterance_id"]), "runtime_seconds": time.monotonic()-t0,
              "peak_vram_bytes": peak, "peak_vram_gb": peak / 2**30,
              "deterministic_decoding": base == again,
              "cached_encoder_identity": base == cached_base,
              "cache_policy": QWEN_ENCODER_CACHE,
              "decoder_rho0_identity": base == off_d,
              "encoder_rho0_identity": base == enc_off,
              "language_condition_states_distinct": bool(not torch.equal(en, zh)),
              "language_condition_max_abs_diff": float((en-zh).abs().max()),
              "cache_generation": False, "no_gradients": True,
              "norm_preserve_nonzero": norm_ok,
              "raw_encoder_tested": bool(raw_encoder_hook.records),
              "raw_decoder_tested": bool(raw_decoder_hook.records),
              "conditioning_alias_tested": bool(conditioning_alias_hook.records),
              "conditioning_avg": "NOT_RUN_REDUNDANT_ALIAS",
              "model": bundle.metadata(), "fps_measurement_deferred_to_construction": True}
    if not all(report[k] for k in ("deterministic_decoding", "decoder_rho0_identity",
                                   "encoder_rho0_identity", "language_condition_states_distinct",
                                   "norm_preserve_nonzero", "raw_encoder_tested",
                                   "raw_decoder_tested", "conditioning_alias_tested")):
        report["status"] = "FAIL"
    _write(OUT / "manifests/qwen_preflight.json", report)
    if report["status"] != "PASS": raise RuntimeError(f"Qwen preflight failed: {report}")
    return 0


def _direction(name, layer):
    from csasr.models.qwen3_asr import file_sha256
    p = QOUT / "directions" / f"{name}_L{layer}.npy"
    return __import__("torch").from_numpy(np.load(p)), file_sha256(p)


def atlas(dataset: str, *, side_filter=None, direction_filter=None,
          layer_start=0, layer_end=None) -> int:
    import torch
    _stage_manifest("atlas_" + dataset, "RUNNING", dataset=dataset, side=side_filter,
                    direction=direction_filter, layer_start=layer_start, layer_end=layer_end)
    from csasr.evaluation import canonical, retention
    from csasr.models.qwen3_asr import generate_one, inputs_for_audio, load_qwen
    from csasr.lss.qwen_sites import AUDIO_SITE, TEXT_SITE, QwenSiteInterventionHook
    if not (QOUT / "directions/manifest.json").is_file(): raise RuntimeError("Qwen directions are missing")
    baseline_file = QOUT / "baseline" / f"{dataset}.json"
    if not baseline_file.is_file(): raise RuntimeError("Qwen baseline must complete before atlas")
    p = _panel(dataset); base = _read(baseline_file)["texts"]; bundle = load_qwen(device="cuda:0")
    dmeta = _read(QOUT / "directions/manifest.json"); fps = float(dmeta["measured_audio_fps_median"])
    # Frozen reference spans from D-construct are mapped to evaluation rows by
    # their accepted CS IDs; SEAME carries its frozen target segment records.
    local_spans = {}
    local_decoder = {}
    encoder_cache = {}
    if dataset == "cs_dialogue":
        from experiments.dg03_build_basis import _load_construct
        _, _, c = _load_construct(); c = c[c["all_correct"]]
        for uid, g in c.groupby("utterance_id"): local_spans[str(uid)] = g.to_dict("records")
    else:
        for row in p["rows"]:
            uid = str(row["utterance_id"])
            local_spans[uid] = _seame_audio_spans(row)
            local_decoder[uid] = _seame_decoder_indices(bundle, row)
    results = []
    def emit(direction, side, layer, scope):
        key = f"{direction}_{side}_L{layer}_{scope}_rho0.5"
        path = QOUT / direction.lower() / dataset / f"L{layer:02d}" / f"{key}.json"
        if path.is_file(): return
        vecname = "raw_encoder" if side == "encoder" else ("raw_decoder" if direction == "Raw" else "conditioning")
        direction_vec, dhash = _direction(vecname, layer)
        texts = {}; edits = []; counts = []; records = []; t0 = time.monotonic()
        for i, row in enumerate(sorted(p["rows"], key=lambda x: float(x["duration_sec"])), 1):
            if side == "encoder":
                inp = inputs_for_audio(bundle, row["audio_path"], language="Chinese")
                n = _qwen_audio_length(int(inp["feature_attention_mask"].sum()))
                mask = _audio_mask(bundle, row, scope=scope, spans=local_spans,
                                   fps=fps, n=n)
                hook = QwenSiteInterventionHook(bundle, layer, direction_vec, alpha=RHO,
                                                site=AUDIO_SITE, scale=float(dmeta["layers"]["encoder"][str(layer)]["mean_site_norm"]),
                                                gain=torch.from_numpy(mask), record=True)
            else:
                inp = inputs_for_audio(bundle, row["audio_path"], language="Chinese")
                prompt_len = int(inp["input_ids"].shape[1])
                uid = str(row["utterance_id"])
                if QWEN_ENCODER_CACHE == "ON" and uid not in encoder_cache:
                    with torch.inference_mode():
                        encoder_cache[uid] = bundle.thinker_model.get_audio_features(
                            inp["input_features"],
                            feature_attention_mask=inp["feature_attention_mask"]).detach().cpu()
                # The prefill contains the system/user template, audio
                # placeholders, and the language/<asr_text> control suffix.
                # Generation positions begin at prompt_len; no prefill state
                # is eligible for decoder steering.
                allowed = set(range(prompt_len, prompt_len + 200))
                if scope == "oracle_local":
                    norm, units, ref_ids, _, _ = _content_plan(bundle, row, "Chinese")
                    idx = []
                    if dataset == "cs_dialogue":
                        for sp in local_spans.get(str(row["utterance_id"]), []): idx += _span_token_indices(bundle, row, sp, ref_ids, norm, units)
                    else:
                        idx = local_decoder.get(str(row["utterance_id"]), [])
                    # The state at position j produces logits for token j+1.
                    allowed = {prompt_len + int(x) - 1 for x in sorted(set(idx)) if int(x) > 0}
                excluded = set(getattr(bundle.processor.tokenizer, "all_special_ids", []))
                hook = QwenSiteInterventionHook(bundle, layer, direction_vec, alpha=RHO,
                                                site=TEXT_SITE, scale=float(dmeta["layers"]["decoder"][str(layer)][vecname]["mean_site_norm"]),
                                                allowed_positions=allowed, excluded_token_ids=excluded, record=True)
            text, _ = generate_one(bundle, row["audio_path"], language="Chinese", hook=hook,
                                   cached_encoder=(encoder_cache.get(str(row["utterance_id"]))
                                                   if side == "decoder" and QWEN_ENCODER_CACHE == "ON" else None))
            uid = str(row["utterance_id"]); texts[uid] = text
            edits.extend(r.edit_norm_sum for r in hook.records); counts.extend(r.active_positions for r in hook.records)
            records.extend(hook.records)
            if i % 10 == 0: print(f"atlas {dataset} {key} {i}/{len(p['rows'])}", flush=True)
        refs = [str(x["reference"]) for x in p["rows"]]; ids = [str(x["utterance_id"]) for x in p["rows"]]
        bm = canonical.corpus_metrics(refs, [base[x] for x in ids]); mm = canonical.corpus_metrics(refs, [texts[x] for x in ids])
        tr = canonical.correction_corruption(refs, [base[x] for x in ids], [texts[x] for x in ids])
        active = sum(r.active_positions for r in records)
        pre_active = sum(r.active_pre_norm_sum for r in records)
        n_seen = sum(r.n_positions for r in records)
        energy = float(sum(edits)); mean_pert = energy / active if active else 0.0
        metrics = dict(mm); metrics.update(canonical.error_metric_gains(bm, mm)); metrics.update({"transitions": tr,
            "poi_corrections": tr["corrections"], "poi_corruptions": tr["corruptions"],
            "poi_net_utility": tr["corrections"] - tr["corruptions"],
            "retention": retention.retention_report(refs, [base[x] for x in ids], [texts[x] for x in ids]),
            "outside_harm": None, "total_intervention_energy": energy,
            "edited_positions_or_frames": int(sum(counts)), "mean_intervention_norm": float(np.mean(edits) if edits else 0.0),
            "runtime_seconds": time.monotonic()-t0, "scope": scope, "side": side,
            "original_hidden_norm": float(pre_active / active) if active else 0.0,
            "mean_perturbation_norm": mean_pert,
            "relative_perturbation": float(energy / pre_active) if pre_active else 0.0,
            "edited_fraction": float(active / n_seen) if n_seen else 0.0})
        _write(path, {"schema_version": "basis_a4_qwen_result_v1", "dataset": dataset,
                      "direction": direction, "side": side, "layer": int(layer), "scope": scope,
                      "rho": RHO, "direction_hash": dhash, "site": AUDIO_SITE if side == "encoder" else TEXT_SITE,
                      "panel_fingerprint": p["fingerprint"], "model_revision": bundle.revision,
                      "protocol_hash": _protocol_hash(),
                      "git_commit": __import__("csasr.utils.logging", fromlist=["git_state"]).git_state(str(REPO))["commit"],
                      "config_hash": _protocol_hash(),
                      "site_hash": __import__("hashlib").sha256((AUDIO_SITE if side == "encoder" else TEXT_SITE).encode()).hexdigest(),
                      "mask_hash": __import__("hashlib").sha256(json.dumps({"dataset": dataset, "scope": scope, "side": side}, sort_keys=True).encode()).hexdigest(),
                      "provenance": {"stage": "basis_a4_qwen_atlas", "model_revision": bundle.revision,
                                     "panel_fingerprint": p["fingerprint"], "direction_hash": dhash,
                                     "cache_policy": QWEN_ENCODER_CACHE},
                      "metrics": metrics, "texts": texts, "baseline_texts": base})
    for side, layers in (("encoder", range(24)), ("decoder", range(28))):
        if side_filter and side != side_filter: continue
        layers = [l for l in layers if l >= int(layer_start) and (layer_end is None or l < int(layer_end))]
        for direction in (("Raw",) if side == "encoder" else ("Raw", "Conditioning")):
            if direction_filter and direction != direction_filter: continue
            for layer in layers:
                for scope in SCOPES: emit(direction, side, layer, scope)
    _stage_manifest("atlas_" + dataset, "COMPLETED", dataset=dataset, side=side_filter,
                    direction=direction_filter, layer_start=layer_start, layer_end=layer_end)
    return 0


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("construct")
    sub.add_parser("preflight")
    b = sub.add_parser("baseline"); b.add_argument("--dataset", choices=DATASETS, required=True)
    a = sub.add_parser("atlas"); a.add_argument("--dataset", choices=DATASETS, required=True)
    a.add_argument("--side", choices=("encoder", "decoder")); a.add_argument("--direction", choices=("Raw", "Conditioning"))
    a.add_argument("--layer-start", type=int, default=0); a.add_argument("--layer-end", type=int)
    args = ap.parse_args()
    try:
        if args.command == "construct": return construct_directions()
        if args.command == "preflight": return preflight()
        if args.command == "baseline": return baseline(args.dataset)
        return atlas(args.dataset, side_filter=args.side, direction_filter=args.direction,
                     layer_start=args.layer_start, layer_end=args.layer_end)
    except Exception as exc:
        stage = "construct" if args.command == "construct" else args.command
        _stage_manifest(stage, "FAILED", error=repr(exc))
        raise


if __name__ == "__main__": raise SystemExit(main())
