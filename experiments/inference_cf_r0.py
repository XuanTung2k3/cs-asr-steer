#!/usr/bin/env python
"""R0 runner — predicted acoustic regions and per-utterance unique-direction construction feasibility
(docs/inference_cf/R0_REGION_VECTOR_SPEC.md, R0_CODEX_DESIGN.md, R0_REFERENCE_FIREWALL.md, R0_PANEL.json,
configs/inference_cf/r0_region_vector.json; freeze a5929c6).

prepare   (CPU) verify freeze/panel/sources/model/tokenizer/audio/baseline/oracle-source byte pins and the window grid;
          write plan_sealed.json holding ONLY the runtime projection (rows: index/ID/dialogue/audio/baseline).
manifest  (CPU) resolved manifest after a committed PASS_TO_R0.
run       (GPU) ONE allocation, canonical row order: fixed-grid raw native LID (unchanged p0_r2.native_lid) ->
          EN/ZH/U windows, sample sweep, frame masks; encoder once; forced-ZH causal full replay of the sealed theta0
          content (unchanged p0_r2.full_replay) with a passive 4-layer DG-02 recorder + FFN-LN-input observer, repeated
          once for bit-identity; forced-EN full replay for the descriptive v_prompt comparator; reference-free mapping,
          groups and per-utterance small-rank constructions (primary + repeat, 20 circular shuffles, ERODE/DILATE,
          baseline-script diagnostic). First canonical row doubles as the in-allocation integrity/cost smoke.
seal      (CPU) immutable reference-free output seal.
No reference, CTC, oracle, evaluator or role-transcript input is opened here.
"""
from __future__ import annotations

import argparse
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

SCHEMA = "r0_v1"
CONFIG = "configs/inference_cf/r0_region_vector.json"
PANEL = "docs/inference_cf/R0_PANEL.json"
SPEC = "docs/inference_cf/R0_REGION_VECTOR_SPEC.md"
DESIGN = "docs/inference_cf/R0_CODEX_DESIGN.md"
FIREWALL = "docs/inference_cf/R0_REFERENCE_FIREWALL.md"
FREEZE_TEST = "tests/test_r0_freeze_contract.py"
BASE = "results/inference_cf/r0"
PLAN = f"{BASE}/plan_sealed.json"
LAYERS = (3, 8, 16, 24)
PROJECTION = ("canonical_index", "utterance_id", "dialogue_id", "audio", "baseline")
SAFETY_SEC = 900          # projection margin below the hard ceiling


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def committed(rel: str) -> bool:
    import hashlib
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return git("ls-files", rel) == rel and "sha256:" + hashlib.sha256(blob).hexdigest() == file_hash(ROOT / rel)


def runtime_projection(panel: dict) -> list[dict]:
    """The ONLY panel content the runner consumes (firewall: rows projection)."""
    return [{k: r[k] for k in PROJECTION} for r in panel["rows"]]


def model_hashes(model_dir: str, names) -> dict:
    return {n: file_hash(Path(model_dir) / n) for n in names}


# ---------------------------------------------------------------------------------------------------------------------

def cmd_prepare(args) -> None:
    import inspect
    import hashlib
    import soundfile as sf
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.inference_cf.r0_regions import window_grid
    cfg = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    checks = {"frozen_committed": all(committed(p) for p in (CONFIG, PANEL, SPEC, DESIGN, FIREWALL, FREEZE_TEST)),
              "panel_identity": digest({k: v for k, v in P.items() if k != "identity_hash"}) == P["identity_hash"] == cfg["panel_identity_hash"],
              "membership": digest([{k: r[k] for k in ("utterance_id", "dialogue_id")} for r in P["rows"]]) == P["membership_hash"] == cfg["membership_hash"],
              "sources": all(file_hash(ROOT / p) == h for p, h in cfg["source_sha256"].items()),
              "model_files": model_hashes(cfg["model"]["dir"], cfg["model"]["files"]) == cfg["model"]["files"]}
    gen = json.loads((Path(cfg["model"]["dir"]) / "generation_config.json").read_text())
    checks["language_map"] = digest(gen["lang_to_id"]) == cfg["regions"]["language_map_hash"] and sorted(set(gen["lang_to_id"].values())) == cfg["regions"]["language_ids"]
    checks["alignment_heads"] = gen["alignment_heads"] == cfg["mapping"]["alignment_heads"]
    tok = WhisperProcessor.from_pretrained(cfg["model"]["dir"], local_files_only=True).tokenizer
    checks["partition"] = tokenizer_partition(tok)["hash"] == cfg["secondary_predictor"]["partition_hash"]
    rows = runtime_projection(P)
    a_ok = b_ok = True
    n_win = 0
    for r in rows:
        a = r["audio"]
        info = sf.info(a["path"])
        a_ok &= ("sha256:" + hashlib.sha256(Path(a["path"]).read_bytes()).hexdigest() == a["full_sha256"] and info.frames == a["source_frames"]
                 and info.samplerate == a["source_sample_rate"] == 16000 and info.channels == a["channels"]
                 and a["resampled_num_samples"] == a["source_frames"] and a["baseline_heard_samples"] == min(480000, a["resampled_num_samples"]))
        b = r["baseline"]
        raw = json.loads((ROOT / b["path"]).read_text())
        b_ok &= (file_hash(ROOT / b["path"]) == b["file_sha256"] and raw["identity"] == r["utterance_id"] and raw["theta0"]["tokens"] == b["content_ids"]
                 and raw["theta0"]["terminated"] == b["terminated"] and digest(b["content_ids"]) == b["content_sha256"])
        n_win += len(window_grid(a["resampled_num_samples"], cfg["regions"]["window_samples"], cfg["regions"]["stride_samples"]))
    checks["audio_bytes_geometry"] = bool(a_ok)
    checks["baseline_theta0"] = bool(b_ok)
    checks["window_grid_10387"] = n_win == 10387
    # oracle source bytes are verified by the independent pre-run auditor; the runner never opens oracle_source.path
    import transformers
    from transformers.models.whisper.modeling_whisper import WhisperDecoderLayer
    st = json.loads((ROOT / "configs/inference_cf/st_loc0.json").read_text())["site_forward"]
    checks["site_forward"] = (transformers.__version__ == st["transformers_version"]
                              and "sha256:" + hashlib.sha256(inspect.getsource(WhisperDecoderLayer.forward).encode()).hexdigest() == st["forward_source_sha256"])
    plan = {"schema": SCHEMA + "_plan", "config_sha256": file_hash(ROOT / CONFIG), "config_hash": digest(cfg), "panel_identity_hash": P["identity_hash"],
            "checks": checks, "rows": rows, "layers": list(LAYERS), "windows_total": n_win, "references_used": False, "created_unix": time.time()}
    plan["plan_hash"] = digest(plan)
    out = ROOT / PLAN
    if out.exists():
        raise FileExistsError("plan exists; never overwrite")
    if not all(checks.values()):
        raise SystemExit("BLOCK: " + json.dumps([k for k, v in checks.items() if not v]))
    atomic_json(out, plan)
    print(json.dumps({"plan_hash": plan["plan_hash"], "checks": checks}))


SOURCES = (SPEC, DESIGN, FIREWALL, PANEL, CONFIG, PLAN, FREEZE_TEST, "src/csasr/inference_cf/r0_regions.py", "src/csasr/inference_cf/r0_unique.py",
           "experiments/inference_cf_r0.py", "experiments/inference_cf_r0_analyze.py", "experiments/inference_cf_r0_audit.py",
           "slurm/inference_cf_r0.sbatch", "tests/test_r0_impl.py", "experiments/inference_cf_p0_r2.py", "experiments/inference_cf_p2dir.py",
           "src/csasr/lss/sites.py", "src/csasr/models/whisper.py", "src/csasr/inference_cf/core_p1.py", "src/csasr/inference_cf/core_r2.py",
           "src/csasr/inference_cf/loc0_sites.py")


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    pre = f"{BASE}/prerun_audit.json"
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, pre)
    if dirty:
        raise ValueError("commit R0 sources before preparing a manifest:\n" + dirty)
    if json.loads((ROOT / pre).read_text())["verdict"] != "PASS_TO_R0":
        raise ValueError("pre-run audit did not pass")
    plan = json.loads((ROOT / PLAN).read_text())
    cfg = json.loads((ROOT / CONFIG).read_text())
    man = {"schema": SCHEMA, "stage": "run1", "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "config_sha256": file_hash(ROOT / CONFIG), "config_hash": digest(cfg), "panel_identity_hash": plan["panel_identity_hash"],
           "plan_hash": plan["plan_hash"], "layers": list(LAYERS), "output_root": args.out,
           "reference_access_boundary": "no oracle/CTC/reference/role transcript before output seal pushed + R0_AUDIT: PASS (PRIMARY)",
           "environment": prep.environment(), "model": {"dir": cfg["model"]["dir"], "files": model_hashes(cfg["model"]["dir"], cfg["model"]["files"])},
           "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
           "sources": {p: file_hash(ROOT / p) for p in SOURCES + (pre,)}, "trainable_parameters": 0, "edit_hooks": 0,
           "references_used": False, "created_unix": time.time()}
    man["manifest_hash"] = digest(man)
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("manifest exists; never overwrite")
    atomic_json(out / "manifest.json", man)
    print(json.dumps({"manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}))


# ---------------------------------------------------------------------------------------------------------------------

class FfnInputObserver:
    """Passive pre-hook on final_layer_norm: the tensor the FFN branch consumes (site audit). Returns None."""

    def __init__(self, bundle, layers):
        self.bundle, self.layers, self.inputs, self._h = bundle, list(layers), {}, []

    def __enter__(self):
        for l in self.layers:
            self._h.append(self.bundle.decoder_layer(l).final_layer_norm.register_forward_pre_hook(
                lambda _m, args, l=l: self.inputs.__setitem__(l, args[0].detach())))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h.clear()
        return False


def assert_no_hooks(bundle) -> None:
    from csasr.inference_cf.loc0_sites import assert_no_any_site_hooks
    assert_no_any_site_hooks(bundle)
    for i, lay in enumerate(bundle.model.model.decoder.layers):
        if lay.final_layer_norm._forward_pre_hooks or lay.final_layer_norm._forward_hooks:
            raise RuntimeError(f"final_layer_norm hook leaked on decoder[{i}]")


def pack(x) -> np.ndarray:
    """Exact bf16 values (float32 tensor from a bf16 model) stored losslessly as raw bf16 bits."""
    import torch
    t = x.detach().cpu().float()
    b = t.to(torch.bfloat16)
    if not torch.equal(b.float(), t):
        raise ValueError("tensor is not exactly bf16-representable")
    return b.view(torch.int16).numpy()


def unpack(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.int16).view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def replay(bundle, encoded, prompt, content, *, attention: bool, layers=LAYERS):
    """One unchanged p0_r2.full_replay under a passive 4-layer recorder and FFN-input observer."""
    import torch
    from csasr.lss.sites import DecoderPostCrossAttnRecorder
    from experiments.inference_cf_p0_r2 import full_replay
    with DecoderPostCrossAttnRecorder(bundle, layers) as rec, FfnInputObserver(bundle, layers) as ffn:
        logits, heads = full_replay(bundle, encoded, list(prompt), list(content), attention=attention)
    assert_no_hooks(bundle)
    site_ok = all(torch.equal(rec.states[l], ffn.inputs[l].to(rec.states[l].dtype)) for l in layers)
    states = {l: rec.states[l][0].float().cpu() for l in layers}
    return logits, heads, states, site_ok


def eligibility(tok, content, eos, suppress, byte_decoder):
    """Per query t in 0..T: eligible iff 1<=t<T, target non-special generation-valid, prior bytes complete UTF8."""
    from csasr.inference_cf.core_r2 import prefix_utf8_complete
    special = set(tok.all_special_ids)
    T = len(content)
    out = []
    for t in range(T + 1):
        reasons = []
        if t == 0:
            reasons.append("t0_forced_prefix_query")
        if t >= T:
            reasons.append("eos_slot")
        else:
            tgt = int(content[t])
            if tgt >= eos or tgt in special or tgt in suppress:
                reasons.append("special_or_invalid_target")
        if 0 < t <= T and not prefix_utf8_complete(tok, content[:t], byte_decoder):
            reasons.append("incomplete_utf8_prefix")
        out.append(reasons)
    return out


def assignment_for_track(track_heard, mp, elig, cfg):
    from csasr.inference_cf.r0_regions import assign_queries, frame_masks
    we, wm = frame_masks(track_heard)
    return assign_queries(mp["a"], mp["mapped"], elig, we, wm, cfg["mapping"]), we, wm


def groups(states_l, lab, bins):
    e, m = np.flatnonzero(lab == 1), np.flatnonzero(lab == 2)
    return states_l[e], states_l[m], bins[e], bins[m], e, m


def process_row(bundle, row: dict, cfg: dict, ctx: dict) -> tuple[dict, dict]:
    """All reference-free outputs of one utterance. ctx holds tokenizer/partition/suppress/counters."""
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.core_p1 import direction, processed_argmax
    from csasr.inference_cf import r0_regions as R
    from csasr.inference_cf import r0_unique as Q
    from csasr.models.whisper import batch_model_inputs, load_audio
    from experiments.inference_cf_p0_r2 import native_lid
    c = ctx["counters"]
    rc, mc, uc, cc = cfg["regions"], cfg["mapping"], cfg["unique"], cfg["controls"]
    uid, a = row["utterance_id"], row["audio"]
    t0 = time.time()
    wav = load_audio(a["path"], rc["sample_rate"])
    n = int(wav.shape[0])
    heard = int(a["baseline_heard_samples"])
    if n != a["resampled_num_samples"] or heard != min(n, 480000):
        raise ValueError("audio geometry mismatch")
    lang = tuple(rc["language_ids"])
    grid = R.window_grid(n, rc["window_samples"], rc["stride_samples"])
    probs = np.zeros((len(grid), len(lang)), dtype=np.float32)
    wins = []
    for k, (s, e) in enumerate(grid):
        crop = wav[s:e]
        d = native_lid(bundle, crop, lang)
        c["lid_calls"] += 1
        probs[k] = np.array([d[i] for i in lang], dtype=np.float32)
        wins.append({"start": s, "end": e, **R.classify_window(probs[k], lang, crop, rc)})
    t_lid = time.time() - t0
    codes = [w["code"] for w in wins]
    intervals = R.sample_intervals(grid, codes, n, rc["sample_vote_fraction_min"])
    track = R.intervals_to_track(intervals, n)
    track_h = track[:heard]
    F = -(-heard // R.FRAME)
    # ---- encoder once; replays ----
    inputs = batch_model_inputs(bundle, [a["path"]])
    with torch.inference_mode():
        h = bundle.model.model.encoder(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state
    c["encoder_passes"] += 1
    enc = BaseModelOutput(last_hidden_state=h.detach())
    content = list(row["baseline"]["content_ids"])
    T = len(content)
    cM, cE = cfg["conditions"]["cM"], cfg["conditions"]["cE"]
    qi = np.arange(len(cM) - 1, len(cM) + T)                    # query index for t = 0..T
    lM, hM, sM, okM = replay(bundle, enc, cM, content, attention=True)
    lM2, hM2, sM2, okM2 = replay(bundle, enc, cM, content, attention=True)
    lE, _, sE, okE = replay(bundle, enc, cE, content, attention=False)
    c["decoder_forwards"] += 3
    repeat_ok = bool(torch.equal(lM, lM2) and torch.equal(hM, hM2) and all(torch.equal(sM[l], sM2[l]) for l in LAYERS))
    finite = bool(torch.isfinite(lM).all() and torch.isfinite(hM).all() and all(torch.isfinite(sM[l]).all() and torch.isfinite(sE[l]).all() for l in LAYERS))
    if hM.shape != (len(mc["alignment_heads"]), len(cM) + T, 1500):
        raise ValueError(f"attention shape {tuple(hM.shape)}")
    heads = hM[:, qi, :].numpy()
    S = {l: sM[l][qi].numpy() for l in LAYERS}                  # (T+1, 1280) float32 exact bf16
    SE = {l: sE[l][qi].numpy() for l in LAYERS}
    if any(S[l].shape != (T + 1, bundle.d_model) for l in LAYERS):
        raise ValueError("state shape")
    elig_r = eligibility(ctx["tok"], content, cfg["conditions"]["eos"], ctx["suppress"], ctx["byte_decoder"])
    elig = np.array([not r for r in elig_r])
    argm = [processed_argmax(lM[int(q)], t, ctx["suppress"], ctx["begin"]) for t, q in enumerate(qi)]
    target = content + [cfg["conditions"]["eos"]]
    agree = [int(argm[t]) == int(target[t]) for t in range(T + 1)]
    R.check_attention(heads)
    heads_real = heads[:, :, :F].copy()
    mp = R.query_mapping(heads_real, mc["raw_real_audio_attention_mass_min"])
    bins = R.time_bins(mp["argmax_frame"], mc["independent_time_bin_samples"])
    asg, we, wm = assignment_for_track(track_h, mp, elig, cfg)
    lab = asg["label"]
    t_gpu = time.time() - t0 - t_lid
    # ---- constructions ----
    t_c = time.time()
    arrays: dict = {"lid_probs": probs, "heads": pack(torch.from_numpy(heads_real)), "wE": we, "wM": wm,
                    "raw_mass": mp["raw_mass"], "qE": asg["qE"], "qM": asg["qM"], "label": lab, "bins": bins, "track_heard_intervals":
                    np.array([(s, e, k) for s, e, k in R.track_to_intervals(track_h)], dtype=np.int64).reshape(-1, 3)}
    for l in LAYERS:
        arrays[f"states_L{l}"] = pack(torch.from_numpy(S[l]))
    cons: dict = {}
    for l in LAYERS:
        HE, HM, bE, bM, ie, im = groups(S[l], lab, bins)
        p1 = Q.construct(HE, HM, bE, bM, uc, keep=True)
        p2 = Q.construct(HE, HM, bE, bM, uc)
        c["constructions"] += 2
        rep = (p1["status"], p1["rank"], p1.get("winner"), p1["vector_sha256"]) == (p2["status"], p2["rank"], p2.get("winner"), p2["vector_sha256"])
        cons[l] = {"primary": Q.public(p1), "primary_repeat_identical": rep, "E_rows": ie.tolist(), "M_rows": im.tolist()}
        if p1["vector"] is not None:
            arrays[f"v_L{l}"] = p1["vector"]
        if "_basis" in p1:
            for k, v in p1["_basis"].items():
                arrays[f"{k}_L{l}"] = v.astype(np.float32) if k in ("U_E", "U_M") else v
            cons[l]["primary"]["basis_hashes_f64"] = {k: Q.array_hash(v) for k, v in p1["_basis"].items()}
        for k, v in p1.get("_spectra", {}).items():
            arrays[f"{k}_L{l}"] = v
    # shuffles (layer-invariant membership per offset)
    offs = R.shuffle_offsets(uid, heard, cc["shuffles"], cc["boundary_samples"], cc["shuffle_seed_key"])
    ctrl = {"shuffle_seed": R.shuffle_seed(uid, cc["shuffle_seed_key"]), "offsets": offs, "shuffle_available": offs is not None,
            "shuffle": {l: [] for l in LAYERS}, "erode": {}, "dilate": {}, "script": {}}
    if offs is not None:
        for k, off in enumerate(offs):
            sa, _, _ = assignment_for_track(np.roll(track_h, off), mp, elig, cfg)
            for l in LAYERS:
                HE, HM, bE, bM, _, _ = groups(S[l], sa["label"], bins)
                r_ = Q.construct(HE, HM, bE, bM, uc)
                c["constructions"] += 1
                ctrl["shuffle"][l].append({"offset": off, "n_E": r_["n_E"], "n_M": r_["n_M"], "status": r_["status"], "rank": r_["rank"],
                                           "winner": r_.get("winner"), "reasons": r_["reasons"], "vector_sha256": r_["vector_sha256"]})
                if r_["vector"] is not None:
                    arrays[f"shuf_L{l}_{k:02d}"] = r_["vector"]
    for name, fn in (("erode", R.erode), ("dilate", R.dilate)):
        pa, _, _ = assignment_for_track(fn(track_h, cc["boundary_samples"]), mp, elig, cfg)
        for l in LAYERS:
            HE, HM, bE, bM, _, _ = groups(S[l], pa["label"], bins)
            r_ = Q.construct(HE, HM, bE, bM, uc)
            c["constructions"] += 1
            ctrl[name][l] = Q.public(r_)
            if r_["vector"] is not None:
                arrays[f"{name}_L{l}"] = r_["vector"]
        arrays[f"{name}_label"] = pa["label"]
    # baseline-script diagnostic (never replaces primary)
    part = ctx["partition"]
    cls = np.array([1 if t < T and content[t] in part["embedded"] else (2 if t < T and content[t] in part["matrix"] else 0) for t in range(T + 1)])
    use = elig & mp["mapped"]
    votes = mp["a"][use]
    tot = votes.sum(axis=0)
    fe = np.where(tot > 0, votes[cls[use] == 1].sum(axis=0) / np.where(tot > 0, tot, 1), 0.0)
    fm = np.where(tot > 0, votes[cls[use] == 2].sum(axis=0) / np.where(tot > 0, tot, 1), 0.0)
    fr = cfg["regions"]["sample_vote_fraction_min"]
    swe, swm = (fe >= fr).astype(np.float64), (fm >= fr).astype(np.float64)
    from csasr.inference_cf.r0_regions import assign_queries
    sasg = assign_queries(mp["a"], mp["mapped"], elig, swe, swm, cfg["mapping"])
    for l in LAYERS:
        HE, HM, bE, bM, _, _ = groups(S[l], sasg["label"], bins)
        r_ = Q.construct(HE, HM, bE, bM, uc)
        c["constructions"] += 1
        ctrl["script"][l] = Q.public(r_)
        if r_["vector"] is not None:
            arrays[f"script_L{l}"] = r_["vector"]
    arrays["script_frame_E"], arrays["script_frame_M"], arrays["script_label"] = swe, swm, sasg["label"]
    # descriptive v_prompt comparator (forced-EN vs forced-ZH, same baseline prefix)
    vp = {}
    eidx = np.flatnonzero(elig)
    for l in LAYERS:
        per = []
        v = arrays.get(f"v_L{l}")
        for t in eidx:
            d = direction(torch.from_numpy(SE[l][t].copy()), torch.from_numpy(S[l][t].copy()))
            cos = None if (d["d"] is None or v is None) else float(np.dot(d["d"].numpy().astype(np.float64), v.astype(np.float64)) / np.linalg.norm(d["d"].numpy().astype(np.float64)))
            per.append({"t": int(t), "status": d["status"], "norm": d["norm"], "cos_primary": cos})
        md = direction(torch.from_numpy(SE[l][eidx].astype(np.float64).mean(axis=0)), torch.from_numpy(S[l][eidx].astype(np.float64).mean(axis=0))) if eidx.size else {"status": "no_eligible", "d": None, "norm": None}
        if md["d"] is not None:
            arrays[f"vprompt_mean_L{l}"] = md["d"].numpy().astype(np.float32)
        vp[l] = {"per_query": per, "mean_delta_status": md["status"], "mean_delta_norm": md["norm"],
                 "mean_delta_cos_primary": None if (md["d"] is None or v is None) else float(md["d"].numpy().astype(np.float64) @ v.astype(np.float64) / np.linalg.norm(md["d"].numpy().astype(np.float64)))}
    t_cons = time.time() - t_c
    queries = [{"t": int(t), "query_index": int(qi[t]), "target": int(target[t]), "replay_argmax": int(argm[t]), "argmax_agrees": agree[t],
                "eligible": bool(elig[t]), "excluded": elig_r[t], "raw_mass": float(mp["raw_mass"][t]), "mapped": bool(mp["mapped"][t]),
                "argmax_frame": int(mp["argmax_frame"][t]), "bin": int(bins[t]), "qE": None if np.isnan(asg["qE"][t]) else float(asg["qE"][t]),
                "qM": None if np.isnan(asg["qM"][t]) else float(asg["qM"][t]), "label": int(lab[t])} for t in range(T + 1)]
    dur = {k: int(np.sum(track == code)) for k, code in (("U", 0), ("EN", 1), ("ZH", 2))}
    dur_h = {k: int(np.sum(track_h == code)) for k, code in (("U", 0), ("EN", 1), ("ZH", 2))}
    rec = {"canonical_index": row["canonical_index"], "utterance_id": uid, "dialogue_id": row["dialogue_id"], "audio_sha256": a["full_sha256"],
           "n_samples": n, "heard_samples": heard, "real_frames": F, "content_sha256": digest(content), "T": T,
           "terminated": row["baseline"]["terminated"], "windows": wins, "intervals_full": [list(x) for x in intervals],
           "durations_full": dur, "durations_heard": dur_h, "queries": queries,
           "counts": {"eligible": int(elig.sum()), "mapped_eligible": int((elig & mp["mapped"]).sum()), "EN": int((lab == 1).sum()), "ZH": int((lab == 2).sum())},
           "integrity": {"site_equals_ffn_input": bool(okM and okM2 and okE), "repeat_replay_bitwise": repeat_ok, "finite": finite,
                         "primary_repeat_identical": all(cons[l]["primary_repeat_identical"] for l in LAYERS),
                         "logits_sha256": "sha256:" + __import__("hashlib").sha256(lM.numpy().tobytes()).hexdigest()},
           "constructions": {str(l): v for l, v in cons.items()},
           "controls": {"shuffle_seed": ctrl["shuffle_seed"], "offsets": offs, "shuffle_available": ctrl["shuffle_available"],
                        "shuffle": {str(l): v for l, v in ctrl["shuffle"].items()}, "erode": {str(l): v for l, v in ctrl["erode"].items()},
                        "dilate": {str(l): v for l, v in ctrl["dilate"].items()}, "script": {str(l): v for l, v in ctrl["script"].items()}},
           "v_prompt": {str(l): v for l, v in vp.items()},
           "timing": {"lid_sec": t_lid, "gpu_sec": t_gpu, "construct_sec": t_cons, "total_sec": time.time() - t0, "windows": len(grid)}}
    return rec, arrays


OPENED: set = set()


def _audit_open(event, args) -> None:
    """Runtime file-open log (firewall evidence): every opened data/repository path outside the Python install."""
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        if p.startswith("/mnt/") or (p.startswith(str(ROOT)) and "/.git/" not in p):
            OPENED.add(p)


def cmd_run(args) -> None:
    sys.addaudithook(_audit_open)
    import torch
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.models.whisper import load_whisper
    from csasr.utils.config import load_config
    from experiments.inference_cf_p2dir_analyze import jsonable
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
    cfg = json.loads((ROOT / CONFIG).read_text())
    if plan["plan_hash"] != m["plan_hash"] or digest(cfg) != m["config_hash"]:
        raise ValueError("plan/config")
    t_start = time.time()
    torch.manual_seed(240924)
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    bundle.model.eval()
    bundle.model.requires_grad_(False)
    if getattr(bundle.model.config, "_attn_implementation", "eager") != "eager":
        raise ValueError("attention implementation must be eager")
    w0 = {k: v.detach().clone() for k, v in bundle.model.state_dict().items() if k.endswith("final_layer_norm.weight")}
    tok = bundle.processor.tokenizer
    part = tokenizer_partition(tok)
    if part["hash"] != cfg["secondary_predictor"]["partition_hash"]:
        raise ValueError("partition")
    gen = bundle.model.generation_config
    if [list(x) for x in gen.alignment_heads] != cfg["mapping"]["alignment_heads"]:
        raise ValueError("alignment heads")
    counters = {"lid_calls": 0, "lid_cache_hits": 0, "encoder_passes": 0, "decoder_forwards": 0, "constructions": 0,
                "autograd_calls": 0, "optimizer_steps": 0, "edit_hooks": 0, "rows_complete": 0, "rows_failed": 0}
    ctx = {"tok": tok, "partition": {"embedded": set(part["embedded_ids"]), "matrix": set(part["matrix_ids"])},
           "suppress": list(gen.suppress_tokens or []), "begin": list(gen.begin_suppress_tokens or []),
           "byte_decoder": {v: k for k, v in bytes_to_unicode().items()}, "counters": counters}
    (out / "rows").mkdir(parents=True, exist_ok=True)
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
               "start_unix": t_start, "status": "running", "counters": counters, "smoke": None}
    atomic_json(out / "runtime.json", runtime)
    torch.cuda.reset_peak_memory_stats()
    rows = plan["rows"]
    total_windows = plan["windows_total"]
    done_windows = 0
    failures: list = []
    try:
        t_rows = time.time()
        for i, row in enumerate(rows):
            if (out / "rows" / f"{i:03d}.json").exists():
                raise FileExistsError("row exists; this runner never resumes over outputs")
            rec, arrays = process_row(bundle, row, cfg, ctx)
            rec = jsonable(rec)
            rec["manifest_hash"] = m["manifest_hash"]
            np.savez_compressed(out / "rows" / f"{i:03d}.npz", **arrays)
            rec["arrays_sha256"] = file_hash(out / "rows" / f"{i:03d}.npz")
            rec["arrays_index"] = {k: {"shape": list(np.asarray(v).shape), "dtype": str(np.asarray(v).dtype)} for k, v in arrays.items()}
            atomic_json(out / "rows" / f"{i:03d}.json", rec)
            counters["rows_complete"] += 1
            done_windows += rec["timing"]["windows"]
            bad = [k for k, v in rec["integrity"].items() if v is False]
            if bad:
                failures.append(f"{rec['utterance_id']}:{bad}")
                raise RuntimeError(f"integrity failure row {i}: {bad}")
            if i == 0 or i == 9:
                el = time.time() - t_rows
                proj = (time.time() - t_start) + el / done_windows * (total_windows - done_windows) * 1.25
                runtime["smoke" if i == 0 else "smoke10"] = {"row": i, "elapsed_rows_sec": el, "windows_done": done_windows,
                                                            "projected_total_sec": proj, "row_timing": rec["timing"]}
                atomic_json(out / "runtime.json", runtime)
                if proj > cfg["compute"]["hard_seconds"] - SAFETY_SEC:
                    raise RuntimeError(f"COMPUTE_STOP: projected {proj:.0f}s exceeds the hard ceiling; explicit compute revision required")
            print(f"R0 row {i + 1}/300 {rec['utterance_id']} T={rec['T']} win={rec['timing']['windows']} "
                  f"{rec['timing']['total_sec']:.1f}s", flush=True)
        status = "completed"
    except Exception as exc:
        import traceback
        status = "failed"
        counters["rows_failed"] += 1
        runtime["failure"] = {"reason": repr(exc), "traceback": traceback.format_exc()}
    assert_no_hooks(bundle)
    w1 = {k: v for k, v in bundle.model.state_dict().items() if k in w0}
    runtime.update(status=status, failures=failures, end_unix=time.time(), peak_alloc=int(torch.cuda.max_memory_allocated()),
                   peak_reserved=int(torch.cuda.max_memory_reserved()), model_grads_none=all(p.grad is None for p in bundle.model.parameters()),
                   requires_grad_any=any(p.requires_grad for p in bundle.model.parameters()),
                   weights_unchanged_sample=all(torch.equal(w0[k], w1[k]) for k in w0), training_mode=bool(bundle.model.training))
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    runtime["opened_paths"] = sorted(p for p in OPENED if not p.startswith(str(out)))
    atomic_json(out / "runtime.json", runtime)
    if status != "completed":
        raise SystemExit("R0 run failed: " + runtime["failure"]["reason"])


def cmd_seal(args) -> None:
    run = ROOT / args.run
    files = sorted(p for p in run.rglob("*") if p.is_file()) + [ROOT / BASE / "primary_analysis.json"]
    man = json.loads((run / "manifest.json").read_text())
    prim = json.loads((ROOT / BASE / "primary_analysis.json").read_text())
    doc = {"schema": SCHEMA + "_output_seal", "run": args.run, "files": {str(p.relative_to(ROOT)): file_hash(p) for p in files},
           "manifest_hash": man["manifest_hash"], "source_commit": man["git_commit"], "config_sha256": file_hash(ROOT / CONFIG),
           "panel_identity_hash": man["panel_identity_hash"], "plan_hash": man["plan_hash"], "primary_valid": prim["valid"],
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
