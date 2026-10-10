#!/usr/bin/env python
"""SRC-CF0-P runner — acoustic counterfactual direction construction and preliminary causal pulse screen
(docs/inference_cf/SRC_CF0_PILOT_SPEC.md, _DESIGN.md, _FIREWALL.md, SRC_CF0_CLAUDE_HANDOFF.md,
configs/inference_cf/src_cf0_pilot.json; freeze 2d23574).

prepare    (CPU, separate process) verifies freeze / config / panel / pins / model / audio / S1-R0 seals, re-derives every
           sealed S1 region from the sealed S1 attention + R0 heard intervals, and writes ONLY the allowlisted runtime
           projection run1/runtime.json (queries, utterances, regions, waveform / mask byte hashes, comparator hashes).
manifest   (CPU) resolved immutable manifest (stage construct, or stage pulse after a valid global P-B authorization).
construct  (GPU job 1; P-A) two independent sweeps over 80 original + 124 distinct S1-masked waveforms: cold encoder,
           cold forced-ZH owner cache through the identical prefix, passive DG-02 states at L16/L24 + pre-FFN input,
           raw logits; bitwise historical identities (S1 M / ST-LOC0 NONE logits, S1 heads, ST-LOC0 / R1 states, S1 masked
           logits) and bitwise repeat identity; CPU float64 v_AC, geometry, random controls and native-BF16 reachability
           ledger. No steering, no gradients, no lexical reference.
seal-a     (CPU) immutable direction seal (push before the construction audit and the evaluator-only Gate P-A).
pulse      (GPU job 2; P-B; requires the pushed minimal PB_authorization.json) NONE, zero dose at L16/L24, frozen L16
           historical apparatus (D0 +/- and D2 at e*), then the complete 24-cell matrix (8 v_AC + 8 off-target + 8 random)
           as single relative-dose pulses from the pristine cropped cache; full energy / consumption / restoration checks.
seal-b     (CPU) immutable reference-free pulse seal (push before the PRIMARY audit and any lexical reference).
smoke      (CPU engineering only) the construct / pulse code paths on SYNTHETIC audio with injected regions.
The runner never opens the panel file (only `prepare` does), strata, reference targets, competitors, evaluator or auditor
outputs, oracle regions or the R0 oracle arrays; GPU phases log every opened path (sys.addaudithook).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
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

SCHEMA = "src_cf0_pilot_v1"
FREEZE = "2d235742d5fe59915eb8cc3ba6ad8511f0fa1c9c"
CONFIG = "configs/inference_cf/src_cf0_pilot.json"
CONFIG_SHA = "sha256:f1881050b0718559d05a3ddaa0cb2832953e0fc7553be4c360d952d57f51aa87"
PANEL = "docs/inference_cf/SRC_CF0_PANEL.json"
FROZEN = (CONFIG, PANEL, "docs/inference_cf/SRC_CF0_PILOT_SPEC.md", "docs/inference_cf/SRC_CF0_PILOT_DESIGN.md",
          "docs/inference_cf/SRC_CF0_PILOT_FIREWALL.md", "docs/inference_cf/SRC_CF0_CLAUDE_HANDOFF.md",
          "docs/inference_cf/SRC_CF0_FULL300_SAFETY_PLAN.md", "tests/test_src_cf0_pilot_contract.py",
          "src/csasr/inference_cf/cf_pilot_contract.py")
BASE = "results/inference_cf/src_cf0_pilot"
RUN = f"{BASE}/run1"
PROJ = f"{RUN}/runtime.json"
PRERUN = f"{RUN}/prerun_audit.json"
DSEAL = f"{RUN}/direction_seal.json"
CAUDIT = f"{RUN}/construction_audit.json"
AUTH = f"{RUN}/PB_authorization.json"
PSEAL = f"{RUN}/pulse_seal.json"
STAGES = ("construct", "pulse")
REMOTE = "origin/cs-asr-steer-inf"
CB = [50258, 50260, 50360, 50364]
EOS = 50257
LAYERS = (16, 24)
R0_KEY = "track_heard_intervals"            # the only R0 primary array read (by `prepare` only)
S1_CAND = "results/inference_cf/s1/run1/candidates"
S1_ACO = "results/inference_cf/s1/run1/acoustic"
S1_SEAL = "results/inference_cf/s1/output_seal.json"
S1_CAND_SEAL = "results/inference_cf/s1/run1/candidates_sealed.json"
LOC0 = "results/inference_cf/st_loc0/run1"
LOC0_SEAL = "results/inference_cf/st_loc0/output_seal.json"
R1CAP = "results/inference_cf/st_prompt_r1/runA/capture"
R1_SEAL = "results/inference_cf/st_prompt_r1/output_seal_A.json"
R1_CONFIG = "configs/inference_cf/st_prompt_r1.json"
E_STAR = 1.1260757575454359
# frozen L16 historical apparatus (ST-LOC0 dir, ST-LOC0 arm id, vector family, sign) -- identical to ST-PROMPT-R1-A's barrier
APPARATUS = {"hist_D0_L16_plus": ("barrier", "v_prompt_L16_POST_CROSS_ATTENTION_PRE_FFN_plus", "D0", 1),
             "hist_D0_L16_minus": ("pulses", "v_prompt_L16_POST_CROSS_ATTENTION_PRE_FFN_minus", "D0", -1),
             "hist_D2_L16": ("barrier", "D2_L16_POST_CROSS_ATTENTION_PRE_FFN", "D2", 1)}
OWN = ("src/csasr/inference_cf/src_cf0_pilot.py", "experiments/inference_cf_src_cf0_pilot.py", "slurm/inference_cf_src_cf0_pilot.sbatch",
       "tests/test_src_cf0_pilot_impl.py", "docs/inference_cf/SRC_CF0_PILOT_IMPLEMENTATION_NOTES.md")
REUSED = ("src/csasr/inference_cf/prompt_r1.py", "src/csasr/inference_cf/core.py", "src/csasr/inference_cf/unique.py",
          "src/csasr/models/whisper.py", "src/csasr/utils/provenance.py")
EXEC_SOURCES = tuple(f for f in FROZEN if f != PANEL) + OWN + REUSED   # + config.pins; the panel is never opened by GPU phases
NOT_EXECUTED = ("experiments/inference_cf_src_cf0_pilot_evaluate.py", "experiments/inference_cf_src_cf0_pilot_audit.py")


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def blob_sha(commit: str, rel: str) -> str:
    return "sha256:" + hashlib.sha256(subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=ROOT, capture_output=True).stdout).hexdigest()


def committed(rel: str) -> bool:
    return git("ls-files", rel) == rel and blob_sha("HEAD", rel) == file_hash(ROOT / rel)


def on_remote(rel: str) -> bool:
    c = git("log", "-n1", "--format=%H", "--", rel)
    return bool(c) and subprocess.run(["git", "merge-base", "--is-ancestor", c, REMOTE], cwd=ROOT).returncode == 0


def arr_sha(a: np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def typed_sha(a: np.ndarray) -> str:
    from csasr.inference_cf.unique import array_hash
    return array_hash(np.ascontiguousarray(a))


def unpack(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.int16).view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def pack(x) -> np.ndarray:
    import torch
    t = x.detach().cpu()
    b = t.to(torch.bfloat16)
    if not torch.equal(b.float(), t.float()):
        raise ValueError("not exactly bf16-representable")
    return b.view(torch.int16).numpy()


def all_pins(cfg: dict) -> tuple:
    return tuple(sorted(set(EXEC_SOURCES) | set(cfg["pins"])))


# ---- CPU prepare ------------------------------------------------------------------------------------------------------

def cmd_prepare(args) -> None:
    import soundfile as sf
    import inspect
    import transformers
    from transformers import GenerationConfig
    from transformers.models.whisper.modeling_whisper import WhisperDecoderLayer
    from csasr.inference_cf.lexical_compatibility import hard_mask
    from csasr.inference_cf.s1_evidence import heard_attention, select_offtarget, select_target
    from csasr.inference_cf.src_cf0_pilot import forbidden_hits, mask_groups, runtime_projection
    from csasr.models.whisper import load_audio
    cfg = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    mdir = cfg["model"]["dir"]
    ck: dict = {}
    ck["frozen_committed_unchanged_since_freeze"] = all(committed(p) and blob_sha(FREEZE, p) == file_hash(ROOT / p) for p in FROZEN)
    ck["config_sha_matches_handoff"] = file_hash(ROOT / CONFIG) == CONFIG_SHA and CONFIG_SHA in (ROOT / "docs/inference_cf/SRC_CF0_CLAUDE_HANDOFF.md").read_text()
    ck["panel_file_and_identity"] = (file_hash(ROOT / PANEL) == cfg["panel"]["file_sha256"]
                                     and digest({k: v for k, v in P.items() if k != "identity_hash"}) == P["identity_hash"] == cfg["panel"]["identity_hash"])
    ck["runtime_membership"] = digest(P["runtime_queries"]) == cfg["panel"]["runtime_membership_hash"] == P["runtime_membership_hash"]
    ck["counts_180_80_20"] = (len(P["runtime_queries"]), len(P["utterances"]), len({u["dialogue_id"] for u in P["utterances"]})) == (180, 80, 20)
    ck["source_pins"] = all(file_hash(ROOT / p) == h for p, h in cfg["pins"].items())
    ck["historical_input_pins"] = all(file_hash(ROOT / p) == h for p, h in cfg["historical_inputs"].items())
    ck["model_files"] = all(file_hash(Path(mdir) / n) == h for n, h in cfg["model"]["files"].items())
    gen = GenerationConfig.from_pretrained(mdir, local_files_only=True)
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    ck["generation_config"] = ([list(x) for x in gen.alignment_heads] == cfg["region_policy"]["alignment_heads"] and gen.decoder_start_token_id == 50258
                               and gen.lang_to_id["<|zh|>"] == 50260 and gen.task_to_id["transcribe"] == 50360 and gen.no_timestamps_token_id == 50364)
    r1 = json.loads((ROOT / R1_CONFIG).read_text())
    site = r1["site_forward"]
    ck["installed_site_forward"] = (transformers.__version__ == site["transformers_version"]
                                    and "sha256:" + hashlib.sha256(inspect.getsource(WhisperDecoderLayer.forward).encode()).hexdigest() == site["forward_source_sha256"]
                                    and file_hash(inspect.getfile(WhisperDecoderLayer)) == site["module_file_sha256"])
    proj = runtime_projection(P, cfg)
    Q, U, R = proj["runtime_queries"], proj["utterances"], proj["regions"]
    by_u = {u["utterance_id"]: u for u in U}
    ok = True
    for u_full in P["utterances"]:
        b = json.loads((ROOT / u_full["baseline_row"]).read_text())["systems"][u_full["baseline_system"]]
        u = by_u[u_full["utterance_id"]]
        ok &= (file_hash(ROOT / u_full["baseline_row"]) == u_full["baseline_row_sha256"] and b["tokens"] == u_full["baseline_content_tokens"]
               and u["content_prefix_tokens"] == b["tokens"][:u["max_t"]])
    for q in Q:
        pre = by_u[q["utterance_id"]]["content_prefix_tokens"][:q["t"]]
        ok &= (q["t"] >= 1 and q["absolute_query"] == 4 + q["t"] - 1 and digest(pre) == q["content_prefix_sha256"]
               and digest(CB + pre) == q["forced_zh_query_input_sha256"])
    ck["runtime_prefixes"] = bool(ok)
    ck["projection_forbidden_free"] = not forbidden_hits(proj)
    # audio identity / geometry / waveform bytes
    wav, aok = {}, True
    for u in U:
        g = u["audio_geometry"]
        info = sf.info(u["audio_path"])
        x = load_audio(u["audio_path"], 16000)
        wav[u["utterance_id"]] = x
        aok &= (file_hash(u["audio_path"]) == u["audio_full_file_sha256"] == g["full_sha256"] and info.samplerate == g["source_sample_rate"] == 16000
                and info.frames == g["source_frames"] and info.channels == g["channels"] and x.dtype == np.float32
                and len(x) == g["resampled_num_samples"] and g["baseline_heard_samples"] == min(len(x), 480000))
    ck["audio_identity_geometry"] = bool(aok)
    # S1 sealed sources: region records verbatim + independent re-derivation from sealed S1 heads + R0 heard intervals
    s1seal, cseal = json.loads((ROOT / S1_SEAL).read_text()), json.loads((ROOT / S1_CAND_SEAL).read_text())
    rok, derived = True, True
    uidx = {u["utterance_id"]: i for i, u in enumerate(U)}
    cand_docs = {}
    for i, u in enumerate(U):
        p = f"{S1_CAND}/{i:03d}.json"
        rok &= cseal["files"].get(p) == file_hash(ROOT / p) and s1seal["files"].get(p) == file_hash(ROOT / p)
        d = json.loads((ROOT / p).read_text())
        rok &= d["utterance_id"] == u["utterance_id"] and d["waveform_sha256"] == arr_sha(wav[u["utterance_id"]])
        cand_docs[u["utterance_id"]] = d
    pu = {u["utterance_id"]: u for u in P["utterances"]}
    r0seal = json.loads((ROOT / "results/inference_cf/r0/output_seal.json").read_text())
    for j, (q, r) in enumerate(zip(Q, R)):
        d = cand_docs[q["utterance_id"]]
        cq = next(x for x in d["queries"] if x["j"] == j)
        inv = P["sealed_region_inventory"][j]
        rok &= (cq["t"] == q["t"] and r["target"] == cq["region"]["target"] and r["off_target"] == cq["region"]["off_target"]
                and r["paired_available"] == cq["region"]["paired_available"] and inv["source"] == f"{S1_CAND}/{uidx[q['utterance_id']]:03d}.json"
                and inv["source_sha256"] == file_hash(ROOT / inv["source"]))
        uf = pu[q["utterance_id"]]
        rok &= file_hash(ROOT / uf["r0_primary_arrays"]) == uf["r0_primary_arrays_sha256"] == r0seal["files"][uf["r0_primary_arrays"]]
        with np.load(ROOT / uf["r0_primary_arrays"]) as z:
            ivs = np.asarray(z[R0_KEY], dtype=np.int64)
        with np.load(ROOT / f"{S1_CAND}/{uidx[q['utterance_id']]:03d}.npz") as z:
            heads = unpack(z[f"q{j:03d}_M_heads"])
        x = wav[q["utterance_id"]]
        heard = min(len(x), 480000)
        att = heard_attention(heads, heard, 0.5)
        s1cfg = {"regions": cfg["region_policy"], "counterfactual": cfg["counterfactual"]}
        tg = select_target(att, ivs, x, heard, s1cfg)
        of = select_offtarget(att, ivs, x, heard, tg, s1cfg)
        derived &= tg == r["target"] and of == r["off_target"] and (tg["status"] == "OK" and of["status"] == "OK") == r["paired_available"]
    ck["s1_sealed_region_records"] = bool(rok)
    ck["s1_regions_rederived_from_sealed_heads"] = bool(derived)
    groups = mask_groups(proj)
    masks = {}
    for (uid, (a, b)), mem in sorted(groups.items()):
        xm = hard_mask(wav[uid], a, b)
        masks[f"{uid}|{a}|{b}"] = {"sha256": arr_sha(xm), "members": [list(m) for m in mem], "length": b - a}
    ck["mask_groups_124"] = len(masks) == 124
    s1m = {}
    for i, u in enumerate(U):
        p = f"{S1_ACO}/{i:03d}.json"
        if s1seal["files"].get(p) == file_hash(ROOT / p):
            for gk, g in json.loads((ROOT / p).read_text())["mask_groups"].items():
                s1m[f"{u['utterance_id']}|{gk.replace('_', '|')}"] = g["x_mask_sha256"]
    ck["mask_bytes_match_s1_acoustic"] = s1m == {k: v["sha256"] for k, v in masks.items()}
    ck["target_73_paired_59"] = (sum(r["target"]["status"] == "OK" for r in R), sum(r["paired_available"] for r in R)) == (73, 59)
    # historical comparators: sealed archives, same query / utterance order
    lseal, rseal = json.loads((ROOT / LOC0_SEAL).read_text()), json.loads((ROOT / R1_SEAL).read_text())
    comps = {f"{LOC0}/calibration/{n}": lseal for n in ("baseline_logits.npz", "states_L16_CROSS.npz", "states_L24_CROSS.npz", "v1_vectors.npz",
                                                       "v1_records.json", "d2_vectors.npz")}
    comps.update({f"{LOC0}/{d}/{i:03d}{s}": lseal for d in ("barrier", "pulses") for i in range(80) for s in (".json", "_logits.npz", "_arrays.npz")})
    comps.update({f"{R1CAP}/{n}": rseal for n in ("states_L16.npz", "states_L24.npz", "none_logits.npz")})
    comps.update({f"{S1_CAND}/{i:03d}.npz": s1seal for i in range(80)})
    comps.update({f"{S1_ACO}/{i:03d}.npz": s1seal for i in range(80)})
    ck["comparators_sealed"] = all(s["files"].get(p) == file_hash(ROOT / p) for p, s in comps.items())
    ck["comparator_utterance_order"] = all(json.loads((ROOT / f"{LOC0}/{d}/{i:03d}.json").read_text())["identity"] == u["utterance_id"]
                                           for d in ("barrier", "pulses") for i, u in enumerate(U))
    v1 = json.loads((ROOT / f"{LOC0}/calibration/v1_records.json").read_text())
    with np.load(ROOT / f"{LOC0}/calibration/v1_vectors.npz") as z:
        ck["D0_vectors_match_records"] = all(v1[f"q{j:03d}_L16_CROSS"]["status"] != "ok" or typed_sha(z[f"q{j:03d}_L16_CROSS"]) == v1[f"q{j:03d}_L16_CROSS"]["sha256"]
                                             for j in range(180))
    steps_clean = sum(1 + u["max_t"] for u in U)
    steps_mask = sum(1 + max(t for t, _, _ in mem) for mem in groups.values())
    fc = cfg["compute"]["construction_forward_bound"]
    forecast = {"construct": {"encoder_calls": 2 * (80 + len(groups)), "cached_decoder_steps": 2 * (steps_clean + steps_mask)},
                "pulse": {"clean_encoder_calls": 80, "clean_prefix_steps": steps_clean, "primary_control_query_forwards_max": 24 * 180,
                          "none_zero_forwards": 3 * 180, "apparatus_forwards": 3 * 180}}
    ck["construct_forecast_equals_frozen_bound"] = (forecast["construct"]["encoder_calls"] == fc["encoder_calls"]
                                                    and forecast["construct"]["cached_decoder_steps"] == fc["cached_decoder_steps"])
    pb = cfg["compute"]["pulse_forward_bound"]
    ck["pulse_forecast_within_frozen_bound"] = (steps_clean <= pb["clean_prefix_steps_max"] and 24 * 180 <= pb["primary_control_query_forwards"]
                                                and 3 * 180 <= pb["none_and_zero_query_forwards_max"])
    doc = {"schema": SCHEMA + "_runtime_projection", "freeze": FREEZE, "config_sha256": file_hash(ROOT / CONFIG), "config_hash": digest(cfg),
           "panel_identity_hash": P["identity_hash"], "projection": proj, "projection_hash": digest(proj),
           "artifacts": {"waveform_sha256": {u: arr_sha(x) for u, x in wav.items()}, "masks": masks},
           "comparators": {p: file_hash(ROOT / p) for p in sorted(comps)}, "suppression_hash": digest({"suppress": sup, "begin": beg}),
           "forecast": forecast, "checks": ck, "references_used": False, "created_unix": time.time()}
    hits = forbidden_hits({k: v for k, v in doc.items() if k not in ("checks", "references_used")})
    if hits:
        raise SystemExit(f"forbidden content in runtime.json: {hits}")
    doc["runtime_hash"] = digest(doc)
    if args.dry_run:
        print(json.dumps({"DRY_RUN": True, "checks": ck, "forecast": forecast, "projection_hash": doc["projection_hash"]}, indent=1))
        return
    out = ROOT / PROJ
    if out.exists():
        raise FileExistsError("runtime projection exists; never overwrite")
    if not all(ck.values()):
        raise SystemExit("SRC_CF0_PILOT blocked at prepare: " + json.dumps([k for k, v in ck.items() if not v]))
    atomic_json(out, doc)
    print(json.dumps({"runtime_hash": doc["runtime_hash"], "projection_hash": doc["projection_hash"], "checks": ck, "forecast": forecast}, indent=1))


# ---- manifest -------------------------------------------------------------------------------------------------------

def cmd_manifest(args) -> None:
    import inspect
    import experiments.inference_cf_p2dir_prepare as prep
    from transformers.models.whisper.modeling_whisper import WhisperDecoderLayer
    from csasr.utils.provenance import code_config_snapshot_hash, resolved_config_hash, test_snapshot_hash
    stage = args.stage
    cfg = json.loads((ROOT / CONFIG).read_text())
    srcs = all_pins(cfg) + NOT_EXECUTED + (PROJ, PANEL)
    need = [] if stage == "construct" else [PRERUN, DSEAL, CAUDIT, AUTH, f"{RUN}/manifest_construct.json"]
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *srcs, *need)
    if dirty:
        raise ValueError("commit sources before the manifest:\n" + dirty)
    proj = json.loads((ROOT / PROJ).read_text())
    man = {"schema": SCHEMA, "stage": stage, "freeze": FREEZE, "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "config": CONFIG, "config_sha256": file_hash(ROOT / CONFIG), "config_hash": digest(cfg), "resolved_config_hash": resolved_config_hash(cfg),
           "panel_identity_hash": cfg["panel"]["identity_hash"], "runtime_hash": proj["runtime_hash"], "projection_hash": proj["projection_hash"],
           "dataset_role": "D-dev-select (already exposed 180/80/20 diagnostic panel)", "seed": 240924,
           "environment": prep.environment(), "model": {"dir": cfg["model"]["dir"], "files": prep.model_hashes()},
           "site_forward_source_sha256": "sha256:" + hashlib.sha256(inspect.getsource(WhisperDecoderLayer.forward).encode()).hexdigest(),
           "generation": {"prompt": CB, "suppression_hash": proj["suppression_hash"], "alignment_heads": cfg["region_policy"]["alignment_heads"]},
           "provenance": {"code_config_snapshot_hash": code_config_snapshot_hash(ROOT), "test_snapshot_hash": test_snapshot_hash(ROOT)},
           "sources": {p: file_hash(ROOT / p) for p in all_pins(cfg) + (PROJ,)}, "not_executed_by_runner": {p: file_hash(ROOT / p) for p in NOT_EXECUTED},
           "prepare_only_inputs": {PANEL: file_hash(ROOT / PANEL)},
           "reference_access_boundary": "runner: runtime projection, audio, sealed reference-free S1/ST-LOC0/ST-PROMPT-R1 archives only; "
                                        "evaluator gate-a after direction seal + construction audit; evaluate-b after pushed pulse seal + PRIMARY PASS",
           "trainable_parameters": 0, "no_autograd": True, "no_optimizer": True, "references_used": False, "status": "FROZEN", "created_unix": time.time()}
    if stage == "pulse":
        auth = json.loads((ROOT / AUTH).read_text())
        if not (committed(AUTH) and on_remote(AUTH) and auth["gate_pass"]):
            raise ValueError("pulse manifest requires a committed, pushed passing PB authorization")
        man.update(authorization_hash=auth["authorization_hash"], direction_seal_sha256=file_hash(ROOT / DSEAL),
                   construct_manifest_hash=json.loads((ROOT / RUN / "manifest_construct.json").read_text())["manifest_hash"])
        man["sources"].update({p: file_hash(ROOT / p) for p in need})
    man["manifest_hash"] = digest(man)
    out = ROOT / RUN / f"manifest_{stage}.json"
    if out.exists():
        raise FileExistsError("manifest exists; never overwrite")
    atomic_json(out, man)
    print(json.dumps({"stage": stage, "manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}))


# ---- GPU-phase shared machinery ---------------------------------------------------------------------------------------

OPENED: set = set()


def _audit_open(event, args) -> None:
    """Runtime file-open log (firewall evidence): every opened data / repository path outside the Python install."""
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        if p.startswith("/mnt/") or (p.startswith(str(ROOT)) and "/.git/" not in p):
            OPENED.add(p)


def load_run(stage: str):
    m = json.loads((ROOT / RUN / f"manifest_{stage}.json").read_text())
    if m["schema"] != SCHEMA or m["stage"] != stage or digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("invalid manifest")
    for p, h in m["sources"].items():
        if file_hash(ROOT / p) != h:
            raise ValueError(f"source changed: {p}")
    head = git("rev-parse", "HEAD")
    if head != m["git_commit"]:
        changed = git("diff", "--name-only", m["git_commit"], head).splitlines()
        if subprocess.run(["git", "merge-base", "--is-ancestor", m["git_commit"], head], cwd=ROOT).returncode != 0 or \
                any(not c.startswith(BASE + "/") for c in changed):
            raise ValueError("HEAD differs from the manifest commit outside results/inference_cf/src_cf0_pilot/")
    cfg = json.loads((ROOT / CONFIG).read_text())
    proj = json.loads((ROOT / PROJ).read_text())
    if digest(cfg) != m["config_hash"] or proj["runtime_hash"] != m["runtime_hash"] or \
            digest({k: v for k, v in proj.items() if k != "runtime_hash"}) != proj["runtime_hash"]:
        raise ValueError("config / runtime projection")
    pre = json.loads((ROOT / PRERUN).read_text())
    if not (committed(PRERUN) and on_remote(PRERUN) and pre["verdict"] == "PASS_TO_SRC_CF0_PILOT" and pre["manifest_hash"] == (
            m["manifest_hash"] if stage == "construct" else m["construct_manifest_hash"])):
        raise SystemExit("pre-run audit PASS_TO_SRC_CF0_PILOT (committed + pushed, same manifest) required")
    return m, proj, cfg


def load_model():
    import torch
    from csasr.models.whisper import load_whisper
    from csasr.utils.config import load_config
    torch.manual_seed(240924)
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    if getattr(bundle.model.config, "_attn_implementation", "eager") != "eager":
        raise ValueError("attention implementation must be eager")
    bundle.model.eval()
    bundle.model.requires_grad_(False)
    return bundle


def weights_digest(bundle) -> str:
    import torch
    h = hashlib.sha256()
    for k, v in bundle.model.state_dict().items():
        h.update(k.encode())
        t = v.detach().contiguous().cpu()
        h.update((t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes())
    return "sha256:" + h.hexdigest()


def encode(bundle, x: np.ndarray, counters: dict, key: str):
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.lexical_compatibility import waveform_model_inputs
    feats = waveform_model_inputs(bundle, x)
    with torch.inference_mode():
        h = bundle.model.model.encoder(input_features=feats["input_features"], attention_mask=feats["attention_mask"]).last_hidden_state
    counters[key] += 1
    lineage = {"features_sha256": typed_sha(feats["input_features"].float().cpu().numpy()),
               "attention_mask_sha256": typed_sha(feats["attention_mask"].cpu().numpy()), "encoder_sha256": typed_sha(h.float().cpu().numpy())}
    return BaseModelOutput(last_hidden_state=h), lineage


class Engine:
    """Owner-scoped steps with a passive two-layer DG-02 recorder and pre-FFN input probe (+ optional pulse hook)."""

    def __init__(self, bundle, counters):
        import experiments.inference_cf_p2r as p2r
        self.p2r, self.bundle, self.c = p2r, bundle, counters

    def step(self, br, new, hook=None):
        from csasr.inference_cf.loc0_sites import Composite, assert_no_any_site_hooks
        from csasr.inference_cf.src_cf0_pilot import FFNInputProbe, ffn_hooks_clear
        from csasr.lss.sites import DecoderPostCrossAttnRecorder
        rec = DecoderPostCrossAttnRecorder(self.bundle, list(LAYERS), keep_last_only=True)
        probe = FFNInputProbe(self.bundle, LAYERS)
        logits, heads, _ = br.step(new, capture_layer=None, attention=True, hook=Composite(hook, rec, probe))
        self.c["decoder_forwards"] += 1
        assert_no_any_site_hooks(self.bundle)
        if not ffn_hooks_clear(self.bundle):
            raise RuntimeError("FFN probe hook leaked")
        st = {l: rec.states[l][0, -1].float().cpu() for l in LAYERS}
        ffn = {l: probe.states[l][0].float().cpu() for l in LAYERS}
        return logits, heads, st, ffn

    def pulse(self, br, L, new, layer, action):
        from csasr.inference_cf.loc0_sites import cross_pulse_hook
        br.crop(L)
        hook = cross_pulse_hook(self.bundle, layer, action, 4)
        logits, _, st, ffn = self.step(br, new, hook=hook)
        self.c["pulse_forwards"] += 1
        rec = hook.records[-1] if hook.records else None
        rec = rec.to_dict() if rec is not None and hasattr(rec, "to_dict") else rec
        return logits, st, ffn, rec


def feed(toks: list, t: int) -> list:
    return list(CB) if t == 0 else [int(toks[t - 1])]


# ---- P-A construction ------------------------------------------------------------------------------------------------

def capture_utterance(eng, *, u: dict, jq: list, groups: dict, x: np.ndarray, masks: dict | None) -> dict:
    """One independent sweep for one utterance: original + every distinct mask, each with a cold encoder and a cold
    owner forced-ZH cache through the identical prefix. Returns packed arrays keyed q{j}_{input}_{kind} + records."""
    import torch
    from csasr.inference_cf.lexical_compatibility import hard_mask
    toks, uid = u["content_prefix_tokens"], u["utterance_id"]
    heard = min(len(x), 480000)
    arrays, recs, fails = {}, defaultdict(dict), []

    def run(name: str, xin: np.ndarray, want: dict, enc_key: str):
        enc, lin = encode(eng.bundle, xin, eng.c, enc_key)
        br = eng.p2r.DiagBranch(eng.bundle, enc, CB, name)
        last = max(want)
        for t in range(last + 1):
            lg, heads, st, ffn = eng.step(br, feed(toks, t))
            eng.c["decoder_steps"] += 1
            if t not in want:
                continue
            for j, role in want[t]:
                q = jq[j]
                key = f"q{j:03d}_{role}"
                arrays[f"{key}_raw"] = pack(lg)
                for l in LAYERS:
                    arrays[f"{key}_L{l}"] = pack(st[l])
                if role == "clean":
                    arrays[f"{key}_heads"] = pack(heads)
                r = {"fed_sha256": digest(br.fed), "fed_ok": br.fed == CB + toks[:t] and digest(br.fed) == q["forced_zh_query_input_sha256"],
                     "query": br.length - 1, "query_ok": br.length - 1 == q["absolute_query"], "raw_finite": bool(torch.isfinite(lg).all()),
                     "states_finite": all(bool(torch.isfinite(st[l]).all()) for l in LAYERS),
                     "ffn_input_equals_site": {str(l): bool(torch.equal(ffn[l], st[l])) for l in LAYERS}, "encoder_lineage": lin}
                recs[j][role] = r
                if not (r["fed_ok"] and r["query_ok"] and r["raw_finite"] and r["states_finite"] and all(r["ffn_input_equals_site"].values())):
                    fails.append(f"{uid}:q{j:03d}:{role}:lineage")
        if br.positions != list(range(br.length)):
            fails.append(f"{uid}:{name}:positions")
        del br, enc

    run("clean", x, {q["t"]: [(j, "clean")] for j, q in jq.items()}, "encoder_calls")
    for (a, b), mem in sorted(groups.items()):
        xm = hard_mask(x, a, b)
        g = {"x_mask_sha256": arr_sha(xm), "outside_equal": xm[:a].tobytes() == x[:a].tobytes() and xm[b:].tobytes() == x[b:].tobytes(),
             "inside_positive_zero": bool(np.all(xm[a:b].view(np.uint32) == 0)), "within_heard": 0 <= a < b <= heard}
        g["matches_runtime"] = masks is None or masks.get(f"{uid}|{a}|{b}", {}).get("sha256") == g["x_mask_sha256"]
        if not all(g.values()):
            fails.append(f"{uid}:mask{a}_{b}:{g}")
        want = defaultdict(list)
        for t, j, role in mem:
            want[t].append((j, role))
        run(f"mask_{a}_{b}", xm, dict(want), "masked_encoder_calls")
        for t, j, role in mem:
            recs[j][role]["mask"] = {"bounds": [a, b], **g}
    return {"arrays": arrays, "records": dict(recs), "fails": fails}


def directions_for_query(eng, *, j: int, q: dict, region: dict, A: dict, ecfg: dict) -> tuple[dict, dict]:
    """CPU float64 v_AC (target / off), geometry, random control and the native-BF16 reachability ledger for one query."""
    import torch
    from csasr.inference_cf.src_cf0_pilot import ETAS, SIGNS, direction, pair_geometry, paired_energy_ok, random_direction, reach, unit_geometry
    out, arrays = {}, {}
    for l in LAYERS:
        h = unpack(A[f"q{j:03d}_clean_L{l}"])
        lay = {"clean_norm": float(np.linalg.norm(h.astype(np.float64)))}
        dirs = {}
        for fam, role in (("target", "target"), ("off", "off_target")):
            if region[role]["status"] != "OK":
                lay[fam] = {"status": "NO_" + ("TARGET_REGION" if fam == "target" else "MATCHED_OFFTARGET"), "region_status": region[role]["status"]}
                continue
            d = direction(h, unpack(A[f"q{j:03d}_{role}_L{l}"]))      # raises ValueError('INVALID_...') -> critical
            lay[fam] = {k: v for k, v in d.items() if k != "vector"}
            if d["vector"] is not None:
                dirs[fam] = d
                arrays[f"q{j:03d}_{fam}_L{l}_v"] = d["vector"]
        vr = random_direction(q["utterance_id"], q["t"], l)
        arrays[f"q{j:03d}_random_L{l}_v"] = vr
        lay["random"] = {"status": "OK", "seed_key": f"SRC_CF0_P-random-v1|240924|{q['utterance_id']}|{q['t']}|{l}",
                         "vector_sha256": arr_sha(vr), **unit_geometry(vr, h)}
        lay["pair"] = pair_geometry(dirs["target"], dirs["off"]) if ("target" in dirs and "off" in dirs) else None
        hn = torch.from_numpy(h.copy()).to(torch.bfloat16).to(eng.bundle.device)
        if not torch.equal(hn.float().cpu(), torch.from_numpy(h)):
            raise RuntimeError("native state not bf16-exact")
        ledger = {}
        vecs = {"target": dirs.get("target", {}).get("vector"), "off": dirs.get("off", {}).get("vector"),
                "random": vr if "target" in dirs else None}
        for fam, v in vecs.items():
            for s in SIGNS:
                for e in ETAS:
                    ledger[f"{fam}|{s}|{e:.2f}"] = reach(hn, v, e, s, ecfg, eng.p2r.solve_scale)
                    eng.c["solver_ledger_cells"] += 1
        lay["reach"] = ledger
        tang = all(fam in dirs for fam in ("target", "off")) and all(
            (lay[f]["tangent_ratio"] >= 0.25) for f in ("target", "off", "random"))
        pairs = {}
        for s in SIGNS:
            for e in ETAS:
                cells = [ledger[f"{f}|{s}|{e:.2f}"] for f in ("target", "off", "random")]
                pairs[f"{s}|{e:.2f}"] = all(c["reachable"] for c in cells) and paired_energy_ok(
                    [c["emulated_edit_norm"] for c in cells], ecfg["max_pairwise_relative_squared_error"])
        lay["paired_energy"] = pairs
        lay["joint_predicted"] = bool(tang and all(pairs.values()))
        out[str(l)] = lay
    return out, arrays


def cmd_construct(args) -> None:
    sys.addaudithook(_audit_open)
    import torch
    from csasr.inference_cf.loc0_sites import assert_no_any_site_hooks
    from csasr.inference_cf.src_cf0_pilot import energy_cfg
    from csasr.models.whisper import load_audio
    m, proj, cfg = load_run("construct")
    out = ROOT / RUN
    cdir = out / "construction"
    if cdir.exists() and any(cdir.iterdir()):
        raise FileExistsError("construction outputs exist; never overwrite (use a new numbered attempt)")
    cdir.mkdir(parents=True, exist_ok=True)
    P = proj["projection"]
    Q, U, R = P["runtime_queries"], P["utterances"], P["regions"]
    by_u = defaultdict(dict)
    for j, q in enumerate(Q):
        by_u[q["utterance_id"]][j] = q
    from csasr.inference_cf.src_cf0_pilot import mask_groups
    groups = defaultdict(dict)
    for (uid, bounds), mem in mask_groups(P).items():
        groups[uid][bounds] = mem
    bundle = load_model()
    w0 = weights_digest(bundle)
    counters = {"encoder_calls": 0, "masked_encoder_calls": 0, "decoder_steps": 0, "decoder_forwards": 0, "pulse_forwards": 0,
                "solver_ledger_cells": 0, "autograd_calls": 0, "optimizer_steps": 0, "steering_hooks": 0, "lid_calls": 0}
    eng = Engine(bundle, counters)
    status = {"stage": "construct", "manifest_hash": m["manifest_hash"], "job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
              "gpu": torch.cuda.get_device_name(0), "start_unix": time.time(), "status": "running", "counters": counters, "weights_before": w0}
    atomic_json(out / "construct_status.json", status)
    torch.cuda.reset_peak_memory_stats()
    sweeps, fails = {1: {}, 2: {}}, []
    fc = proj["forecast"]["construct"]
    try:
        for rep in (1, 2):
            for i, u in enumerate(U):
                t_u = time.time()
                x = load_audio(u["audio_path"], 16000)
                if arr_sha(x) != proj["artifacts"]["waveform_sha256"][u["utterance_id"]]:
                    fails.append(f"{u['utterance_id']}:waveform")
                res = capture_utterance(eng, u=u, jq=by_u[u["utterance_id"]], groups=groups[u["utterance_id"]], x=x, masks=proj["artifacts"]["masks"])
                sweeps[rep][i] = res
                fails += [f"rep{rep}:{f}" for f in res["fails"]]
                assert_no_any_site_hooks(bundle)
                print(f"P-A rep{rep} {i + 1}/80 {u['utterance_id']} masks={len(groups[u['utterance_id']])} {time.time() - t_u:.1f}s", flush=True)
                if rep == 1 and i == 0:
                    per = (time.time() - status["start_unix"]) / max(1, counters["decoder_steps"])
                    status["throughput_smoke"] = {"sec_per_step_incl_overhead": per, "forecast_construct_sec": per * fc["cached_decoder_steps"],
                                                  "forecast_pulse_sec": per * (proj["forecast"]["pulse"]["clean_prefix_steps"] + 31 * 180) * 2}
                    atomic_json(out / "construct_status.json", status)
                    if max(status["throughput_smoke"]["forecast_construct_sec"], status["throughput_smoke"]["forecast_pulse_sec"]) > 9000:
                        raise SystemExit("compute forecast > 2.5 h: STOP for an explicit compute revision (no matrix reduction)")
        # ---- historical comparators (sealed, reference-free) ----
        with np.load(ROOT / f"{LOC0}/calibration/baseline_logits.npz") as z:
            hist_none = {k: z[k] for k in z.files}
        hist_states = {}
        for l in LAYERS:
            with np.load(ROOT / f"{LOC0}/calibration/states_L{l}_CROSS.npz") as z:
                hist_states[("loc0", l)] = z["H_B"]
            with np.load(ROOT / f"{R1CAP}/states_L{l}.npz") as z:
                hist_states[("r1", l)] = z["H_M"]
        ecfg = energy_cfg(cfg)
        summary = defaultdict(int)
        for i, u in enumerate(U):
            uid = u["utterance_id"]
            A1, A2 = sweeps[1][i]["arrays"], sweeps[2][i]["arrays"]
            arrays = {}
            for k, v in A1.items():                     # sweep 1 (clean raw logits as q{j}_clean_r1_raw)
                arrays[k.replace("_raw", "_r1_raw") if k.endswith("_raw") else k] = v
            repeat_ok = set(A1) == set(A2) and all(np.array_equal(A1[k], A2[k]) for k in A1)
            for k, v in A2.items():                     # sweep 2 stored in full for independent repeat audit
                arrays[f"r2__{k}"] = v
            lin_ok = all(sweeps[1][i]["records"][j][r]["encoder_lineage"] == sweeps[2][i]["records"][j][r]["encoder_lineage"]
                         for j in sweeps[1][i]["records"] for r in sweeps[1][i]["records"][j])
            if not (repeat_ok and lin_ok):
                fails.append(f"{uid}:repeat_mismatch")
            with np.load(ROOT / f"{S1_CAND}/{i:03d}.npz") as z:
                s1c = {k: z[k] for k in z.files}
            with np.load(ROOT / f"{S1_ACO}/{i:03d}.npz") as z:
                s1a = {k: z[k] for k in z.files}
            qrecs = []
            for j, q in sorted(by_u[uid].items()):
                rec = {"j": j, "utterance_id": uid, "dialogue_id": q["dialogue_id"], "t": q["t"], "absolute_query": q["absolute_query"],
                       "region": {"target_status": R[j]["target"]["status"], "off_target_status": R[j]["off_target"]["status"],
                                  "target_bounds": R[j]["target"]["bounds"], "off_target_bounds": R[j]["off_target"]["bounds"],
                                  "paired_available": R[j]["paired_available"]},
                       "inputs": sweeps[1][i]["records"][j], "repeat_bitwise": bool(repeat_ok and lin_ok)}
                hid = {"clean_logits_vs_s1_M": bool(np.array_equal(A1[f"q{j:03d}_clean_raw"], s1c[f"q{j:03d}_M_raw"])),
                       "clean_logits_vs_st_loc0_none": bool(np.array_equal(A1[f"q{j:03d}_clean_raw"], hist_none[f"q{j:03d}"])),
                       "clean_heads_vs_s1": bool(np.array_equal(A1[f"q{j:03d}_clean_heads"], s1c[f"q{j:03d}_M_heads"]))}
                for l in LAYERS:
                    st = unpack(A1[f"q{j:03d}_clean_L{l}"])
                    hid[f"clean_L{l}_vs_st_loc0"] = bool(np.array_equal(st, hist_states[("loc0", l)][j]))
                    hid[f"clean_L{l}_vs_st_prompt_r1"] = bool(np.array_equal(st, hist_states[("r1", l)][j]))
                for role in ("target", "off_target"):
                    if f"q{j:03d}_{role}_raw" in A1:
                        hid[f"{role}_logits_vs_s1"] = bool(np.array_equal(A1[f"q{j:03d}_{role}_raw"], s1a.get(f"q{j:03d}_{role}_raw")))
                rec["historical_identity"] = hid
                if not all(hid.values()):
                    fails.append(f"{uid}:q{j:03d}:historical:{[k for k, v in hid.items() if not v]}")
                try:
                    geo, varr = directions_for_query(eng, j=j, q=q, region=R[j], A=A1, ecfg=ecfg)
                except ValueError as exc:
                    fails.append(f"{uid}:q{j:03d}:CRITICAL:{exc}")
                    geo, varr = {"critical": str(exc)}, {}
                rec["layers"] = geo
                arrays.update(varr)
                for l in LAYERS:
                    g = geo.get(str(l), {})
                    summary[f"L{l}_target_OK"] += int(g.get("target", {}).get("status") == "OK")
                    summary[f"L{l}_off_OK"] += int(g.get("off", {}).get("status") == "OK")
                    summary[f"L{l}_joint_predicted"] += int(bool(g.get("joint_predicted")))
                qrecs.append(rec)
            np.savez_compressed(cdir / f"{i:03d}.npz", **arrays)
            doc = {"schema": SCHEMA + "_construction_row", "manifest_hash": m["manifest_hash"], "index": i, "utterance_id": uid,
                   "dialogue_id": u["dialogue_id"], "waveform_sha256": proj["artifacts"]["waveform_sha256"][uid], "queries": qrecs,
                   "arrays_index": {k: {"dtype": str(v.dtype), "shape": list(v.shape), "sha256": typed_sha(v)} for k, v in arrays.items()},
                   "arrays_file_sha256": file_hash(cdir / f"{i:03d}.npz"), "references_used": False}
            atomic_json(cdir / f"{i:03d}.json", doc)
        status["summary_reference_free"] = dict(summary)
        status["status"] = "completed" if not fails else "failed_integrity"
    except SystemExit:
        status["status"] = "stopped_compute_forecast"
        raise
    except Exception as exc:
        import traceback
        status["status"] = "failed_exception"
        status["exception"] = {"reason": repr(exc), "traceback": traceback.format_exc()}
    finally:
        status["failures"] = fails[:500]
        status["n_failures"] = len(fails)
        finish(bundle, status, w0, out / "construct_status.json")
    print(json.dumps({"status": status["status"], "elapsed": status["elapsed_sec"], "counters": counters, "summary": status.get("summary_reference_free"),
                      "failures": fails[:10]}), flush=True)
    if status["status"] != "completed":
        raise SystemExit("SRC_CF0_PILOT_INVALID: construction integrity failure")


def finish(bundle, status: dict, w0: str, path: Path) -> None:
    import torch
    from csasr.lss.sites import assert_no_site_hooks
    assert_no_site_hooks(bundle)
    status.update(end_unix=time.time(), peak_vram_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                  peak_vram_reserved_bytes=int(torch.cuda.max_memory_reserved()), model_grads_none=all(p.grad is None for p in bundle.model.parameters()),
                  requires_grad_any=any(p.requires_grad for p in bundle.model.parameters()), training_mode=bool(bundle.model.training),
                  weights_after=weights_digest(bundle), top_forward_hooks=len(bundle.model._forward_hooks), opened_paths=sorted(OPENED))
    status["weights_unchanged"] = status["weights_after"] == w0
    status["elapsed_sec"] = status["end_unix"] - status["start_unix"]
    atomic_json(path, status)


def cmd_seal_a(args) -> None:
    run = ROOT / RUN
    st = json.loads((run / "construct_status.json").read_text())
    m = json.loads((run / "manifest_construct.json").read_text())
    if st["status"] != "completed" or st["manifest_hash"] != m["manifest_hash"]:
        raise ValueError("construction not completed")
    files = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted((run / "construction").iterdir())}
    for p in ("runtime.json", "manifest_construct.json", "construct_status.json", "prerun_audit.json"):
        files[f"{RUN}/{p}"] = file_hash(run / p)
    for p in sorted(run.glob("slurm-*")):
        if st.get("job_id") and st["job_id"] in p.name:
            files[str(p.relative_to(ROOT))] = file_hash(p)
    n = sum(len(json.loads((run / "construction" / f"{i:03d}.json").read_text())["queries"]) for i in range(80))
    if n != 180:
        raise ValueError("construction coverage")
    doc = {"schema": SCHEMA + "_direction_seal", "status": "SEALED", "manifest_hash": m["manifest_hash"], "source_commit": m["git_commit"],
           "source_commit_on_remote": subprocess.run(["git", "merge-base", "--is-ancestor", m["git_commit"], REMOTE], cwd=ROOT).returncode == 0,
           "config_sha256": file_hash(ROOT / CONFIG), "runtime_hash": m["runtime_hash"], "projection_hash": m["projection_hash"],
           "sources": m["sources"], "expected_coverage": {"queries": 180, "layers": list(LAYERS), "records": 360}, "query_records": n,
           "summary_reference_free": st["summary_reference_free"], "files": files, "references_used": False,
           "git_head_at_seal": git("rev-parse", "HEAD"), "created_unix": time.time()}
    doc["seal_hash"] = digest(doc)
    out = ROOT / DSEAL
    if out.exists():
        raise FileExistsError("direction seal exists; never overwrite")
    atomic_json(out, doc)
    print(json.dumps({"seal_hash": doc["seal_hash"], "files": len(files), "summary": doc["summary_reference_free"]}))


# ---- P-B pulses ------------------------------------------------------------------------------------------------------

def check_authorization(cfg: dict, m: dict) -> dict:
    auth = json.loads((ROOT / AUTH).read_text())
    keys = cfg["firewall"]["authorization_keys"]
    if sorted(auth) != sorted(keys) or digest({k: v for k, v in auth.items() if k != "authorization_hash"}) != auth["authorization_hash"]:
        raise SystemExit("PB authorization schema / hash invalid")
    if not (committed(AUTH) and on_remote(AUTH) and committed(DSEAL) and on_remote(DSEAL) and committed(CAUDIT) and on_remote(CAUDIT)):
        raise SystemExit("PB authorization / direction seal / construction audit must be committed and pushed")
    if not (auth["gate_pass"] is True and auth["config_sha256"] == file_hash(ROOT / CONFIG) and auth["direction_seal_sha256"] == file_hash(ROOT / DSEAL)
            and auth["construction_audit_sha256"] == file_hash(ROOT / CAUDIT) and auth["authorization_hash"] == m["authorization_hash"]
            and set(auth["qualified_layers"]) <= set(LAYERS) and auth["qualified_layers"]):
        raise SystemExit("PB authorization does not authorize P-B")
    if json.loads((ROOT / CAUDIT).read_text())["verdict"] != "SRC_CF0_PILOT_AUDIT: PASS (CONSTRUCTION)":
        raise SystemExit("construction audit did not pass")
    seal = json.loads((ROOT / DSEAL).read_text())
    if digest({k: v for k, v in seal.items() if k != "seal_hash"}) != seal["seal_hash"] or any(file_hash(ROOT / p) != h for p, h in seal["files"].items()):
        raise SystemExit("direction seal / sealed file changed")
    return auth


def pulse_utterance(eng, *, u: dict, jq: dict, A: dict, recs: dict, x: np.ndarray, cfg: dict, hist: dict | None) -> dict:
    """NONE, zero (L16/L24), L16 apparatus and the complete 24-cell matrix at every query of one utterance."""
    import torch
    from csasr.inference_cf.core_p1 import processed_argmax
    from csasr.inference_cf.loc0_sites import cache_fingerprint, pulse_action
    from csasr.inference_cf.prompt_r1 import relative_action
    from csasr.inference_cf.src_cf0_pilot import arms, energy_cfg, paired_energy_ok, pulse_checks
    gen = eng.bundle.model.generation_config
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    ecfg = energy_cfg(cfg)
    toks, uid = u["content_prefix_tokens"], u["utterance_id"]
    enc, lin = encode(eng.bundle, x, eng.c, "encoder_calls")
    B = eng.p2r.DiagBranch(eng.bundle, enc, CB, "B")
    jt = {q["t"]: j for j, q in jq.items()}
    rows, logits_out, arrays_out, fails = {}, {}, {}, []
    for t in range(max(jt) + 1):
        L = B.length
        new = feed(toks, t)
        if t not in jt:
            eng.step(B, new)
            eng.c["decoder_steps"] += 1
            continue
        j = jt[t]
        q = jq[j]
        query = q["absolute_query"]
        fp0 = cache_fingerprint(B.cache, L)
        if L + len(new) - 1 != query:
            raise RuntimeError("query index mismatch")
        # pre-edit reference = the SEALED P-A clean readout (bitwise); the live NONE forward runs LAST (it also advances the
        # pristine cache past this query) and must reproduce it bitwise. No extra forward beyond the frozen bound.
        lb = torch.from_numpy(unpack(A[f"q{j:03d}_clean_r1_raw"]).copy())
        sb = {l: torch.from_numpy(unpack(A[f"q{j:03d}_clean_L{l}"]).copy()) for l in LAYERS}
        pr = {"j": j, "t": t, "query": query, "zero": {}, "apparatus": {}, "cells": {}}
        for l in LAYERS:                                                     # zero dose at both layers
            info: dict = {}
            la, sa, ffn, rec = eng.pulse(B, L, new, l, pulse_action(query, lambda r: (None, "zero_dose"), 1.0, info, ecfg["max_relative_squared_error"],
                                                                      eng.p2r.solve_scale, eng.p2r.scaled_direction))
            eng.c["zero_dose_forwards"] += 1
            z_ok = bool(torch.equal(la, lb)) and all(torch.equal(sa[k], sb[k]) for k in LAYERS) and not (rec and rec["steered"])
            pr["zero"][str(l)] = z_ok
            if not z_ok:
                fails.append(f"{uid}:q{j:03d}:zero:L{l}")
        if hist is not None:                                                 # frozen L16 historical apparatus at e*
            for aid, (hd, haid, fam, sign) in APPARATUS.items():
                info = {}
                v = hist["D0"][f"q{j:03d}_L16_CROSS"] if fam == "D0" else hist["D2"][f"q{j:03d}"]
                vf = (lambda r, v=v, sign=sign: (torch.from_numpy((sign * v).astype(np.float32)), "ok"))
                act = pulse_action(query, vf, E_STAR, info, ecfg["max_relative_squared_error"], eng.p2r.solve_scale, eng.p2r.scaled_direction)
                la, sa, ffn, rec = eng.pulse(B, L, new, 16, act)
                eng.c["apparatus_forwards"] += 1
                hrec = hist[hd]["positions"][str(t)]["cells"][haid]
                s_hist = float(hrec["solver"].get("s", 0.0))
                bar = {"logits_bitwise": bool(np.array_equal(pack(la), hist[hd + "_lg"][f"q{j:03d}_{haid}"])),
                       "consumed_bitwise": bool(np.array_equal(sa[16].numpy(), hist[hd + "_ar"][f"q{j:03d}_{haid}_consumed"])) if hrec["steered"] else None,
                       "solver_s": abs(float(info.get("s", 0.0)) - s_hist) <= 1e-9 * max(1.0, abs(s_hist)),
                       "edit_norm_equal": (rec["edit_norm"] if rec and rec["steered"] else 0.0) == hrec["edit_norm"],
                       "steered_equal": bool(rec and rec["steered"]) == bool(hrec["steered"])}
                pr["apparatus"][aid] = bar
                logits_out[f"q{j:03d}_{aid}"] = pack(la)
                if not all(v_ for v_ in bar.values() if v_ is not None):
                    fails.append(f"{uid}:q{j:03d}:apparatus:{aid}:{bar}")
        for a in arms():                                                     # the complete 24-cell matrix
            l, eta, sign, fam = a["layer"], a["eta"], a["sign"], a["family"]
            key = "target" if fam == "random" else fam
            avail = f"q{j:03d}_{key}_L{l}_v" in A
            v = A.get(f"q{j:03d}_{fam}_L{l}_v")
            pred = recs[j]["layers"][str(l)]["reach"].get(f"{fam}|{sign}|{eta:.2f}", {}).get("reachable", False)
            c = {"arm": a["id"], "family": fam, "layer": l, "eta": eta, "sign": sign, "predicted_reachable": bool(pred), "steered": False,
                 "valid": False, "forward": False, "integrity_failures": []}
            if not avail or v is None:
                g = recs[j]["layers"][str(l)].get("target" if fam == "random" else fam, {})
                c["no_edit_reason"] = "INELIGIBLE_" + str(g.get("status", "NO_DIRECTION"))
                c["output"] = "BASELINE_REFERENCE"
                pr["cells"][a["id"]] = c
                continue
            info = {}
            vf = (lambda r, v=v, sign=sign: (torch.from_numpy((sign * v).astype(np.float32)), "ok"))
            act = relative_action(query, vf, eta, info, ecfg["max_relative_squared_error"], ecfg["minimum_state_norm"], eng.p2r.solve_scale,
                                  eng.p2r.scaled_direction)
            la, sa, ffn, rec = eng.pulse(B, L, new, l, act)
            eng.c["matrix_forwards"] += 1
            c["forward"] = True
            c["steered"] = bool(rec and rec["steered"])
            c["solver"] = {k: v_ for k, v_ in info.items() if k not in ("v", "_proposed")}
            c["top1"] = processed_argmax(la, t, sup, beg)
            c["ffn_equals_site"] = all(torch.equal(ffn[k], sa[k]) for k in LAYERS)
            if not c["ffn_equals_site"]:
                c["integrity_failures"].append("ffn_consumption")
            if c["steered"]:
                ch = pulse_checks(eta, info, rec, sb[l], sa[l], ecfg)
                c.update({k: v_ for k, v_ in ch.items()})
                c["realized_edit_norm"] = rec["edit_norm"]
                c["solver_v_sha256"] = arr_sha(info["v"].numpy())
                for k_ in ("consumed_vs_proposed", "solver_matches_hook"):
                    if not ch["checks"][k_]:
                        c["integrity_failures"].append(k_)              # consumed-site / solver mismatch is critical
                other = [k for k in LAYERS if k != l]
                if not all(torch.equal(sa[k], sb[k]) for k in other if k < l):
                    c["integrity_failures"].append("upstream_layer_changed")
                logits_out[f"q{j:03d}_{a['id']}"] = pack(la)
                arrays_out[f"q{j:03d}_{a['id']}_consumed"] = pack(sa[l])
                arrays_out[f"q{j:03d}_{a['id']}_proposed"] = info["_proposed"].numpy().astype(np.float32)
                c["output"] = "STORED"
            else:
                c["no_edit_reason"] = info.get("no_edit_reason")
                c["output"] = "BASELINE_REFERENCE"
                if c["no_edit_reason"] not in ("energy_unreachable", "solver_target_unattainable_within_8_evaluations"):
                    c["integrity_failures"].append(f"unexpected_no_edit:{c['no_edit_reason']}")
                if not (torch.equal(la, lb) and all(torch.equal(sa[k], sb[k]) for k in LAYERS)):
                    c["integrity_failures"].append("no_edit_not_baseline")
            if not bool(torch.isfinite(la).all()):
                c["integrity_failures"].append("nonfinite_logits")
            B.crop(L)
            c["cache_restored"] = cache_fingerprint(B.cache, L) == fp0 and B.positions == list(range(L)) and B.cache.get_seq_length() == L
            eng.c["cache_fingerprints"] += 1
            if not c["cache_restored"]:
                c["integrity_failures"].append("cache_lineage")
            if c["integrity_failures"]:
                fails.append(f"{uid}:q{j:03d}:{a['id']}:{c['integrity_failures']}")
            pr["cells"][a["id"]] = c
        pr["paired_energy"] = {}
        for l in LAYERS:
            for eta in (0.15, 0.30):
                for sign in (1, -1):
                    from csasr.inference_cf.src_cf0_pilot import arm_id
                    ids = [arm_id(l, eta, sign, f) for f in ("target", "off", "random")]
                    cs = [pr["cells"][i_] for i_ in ids]
                    pr["paired_energy"][arm_id(l, eta, sign)] = {
                        "all_valid": all(c_["valid"] for c_ in cs),
                        "ok": all(c_["valid"] for c_ in cs) and paired_energy_ok([c_["realized_edit_norm"] for c_ in cs], ecfg["max_pairwise_relative_squared_error"])}
        # restoration: pristine prefix fingerprint after the last crop, then the live NONE (prefix step) from that cache
        B.crop(L)
        pr["restore_fingerprint"] = cache_fingerprint(B.cache, L) == fp0 and B.positions == list(range(L))
        if not pr["restore_fingerprint"]:
            fails.append(f"{uid}:q{j:03d}:restore")
        ln, _, sn, ffn0 = eng.step(B, new)
        eng.c["decoder_steps"] += 1
        eng.c["none_forwards"] += 1
        pr["none_bitwise_sealed_clean"] = bool(torch.equal(ln, lb)) and all(torch.equal(sn[l], sb[l]) for l in LAYERS)
        pr["none_ffn_equals_site"] = all(torch.equal(ffn0[l], sn[l]) for l in LAYERS)
        pr["none_top1"] = processed_argmax(ln, t, sup, beg)
        if not (pr["none_bitwise_sealed_clean"] and pr["none_ffn_equals_site"]):
            fails.append(f"{uid}:q{j:03d}:none")
        rows[str(j)] = pr
    lineage_ok = B.positions == list(range(B.length))
    if not lineage_ok:
        fails.append(f"{uid}:positions")
    return {"rows": rows, "logits": logits_out, "arrays": arrays_out, "fails": fails, "encoder_lineage": lin}


def cmd_pulse(args) -> None:
    sys.addaudithook(_audit_open)
    import torch
    from csasr.inference_cf.loc0_sites import assert_no_any_site_hooks
    from csasr.models.whisper import load_audio
    m, proj, cfg = load_run("pulse")
    auth = check_authorization(cfg, m)
    out = ROOT / RUN
    pdir = out / "pulses"
    if pdir.exists() and any(pdir.iterdir()):
        raise FileExistsError("pulse outputs exist; never overwrite (use a new numbered attempt)")
    pdir.mkdir(parents=True, exist_ok=True)
    P = proj["projection"]
    Q, U = P["runtime_queries"], P["utterances"]
    by_u = defaultdict(dict)
    for j, q in enumerate(Q):
        by_u[q["utterance_id"]][j] = q
    bundle = load_model()
    w0 = weights_digest(bundle)
    counters = {"encoder_calls": 0, "decoder_steps": 0, "decoder_forwards": 0, "pulse_forwards": 0, "zero_dose_forwards": 0, "apparatus_forwards": 0,
                "matrix_forwards": 0, "none_forwards": 0, "cache_fingerprints": 0, "autograd_calls": 0, "optimizer_steps": 0, "lid_calls": 0}
    eng = Engine(bundle, counters)
    status = {"stage": "pulse", "manifest_hash": m["manifest_hash"], "authorization_hash": auth["authorization_hash"], "qualified_layers": auth["qualified_layers"],
              "job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "start_unix": time.time(),
              "status": "running", "counters": counters, "weights_before": w0}
    atomic_json(out / "pulse_status.json", status)
    torch.cuda.reset_peak_memory_stats()
    hist = {}
    with np.load(ROOT / f"{LOC0}/calibration/v1_vectors.npz") as z:
        hist["D0"] = {k: z[k] for k in z.files if k.endswith("_L16_CROSS")}
    with np.load(ROOT / f"{LOC0}/calibration/d2_vectors.npz") as z:
        hist["D2"] = {k: z[k] for k in z.files}
    fails = []
    try:
        for i, u in enumerate(U):
            t_u = time.time()
            uid = u["utterance_id"]
            cd = json.loads((out / "construction" / f"{i:03d}.json").read_text())
            if cd["utterance_id"] != uid or file_hash(out / "construction" / f"{i:03d}.npz") != cd["arrays_file_sha256"]:
                raise ValueError(f"sealed construction row {i}")
            with np.load(out / "construction" / f"{i:03d}.npz") as z:
                A = {k: z[k] for k in z.files}
            recs = {q["j"]: q for q in cd["queries"]}
            x = load_audio(u["audio_path"], 16000)
            if arr_sha(x) != proj["artifacts"]["waveform_sha256"][uid]:
                fails.append(f"{uid}:waveform")
            h = dict(hist)
            for d in ("barrier", "pulses"):
                h[d] = json.loads((ROOT / f"{LOC0}/{d}/{i:03d}.json").read_text())
                with np.load(ROOT / f"{LOC0}/{d}/{i:03d}_logits.npz") as z:
                    h[d + "_lg"] = {k: z[k] for k in z.files}
                with np.load(ROOT / f"{LOC0}/{d}/{i:03d}_arrays.npz") as z:
                    h[d + "_ar"] = {k: z[k] for k in z.files}
            res = pulse_utterance(eng, u=u, jq=by_u[uid], A=A, recs=recs, x=x, cfg=cfg, hist=h)
            fails += res["fails"]
            assert_no_any_site_hooks(bundle)
            np.savez_compressed(pdir / f"{i:03d}_logits.npz", **res["logits"])
            np.savez_compressed(pdir / f"{i:03d}_arrays.npz", **res["arrays"])
            doc = {"schema": SCHEMA + "_pulse_row", "manifest_hash": m["manifest_hash"], "authorization_hash": auth["authorization_hash"], "index": i,
                   "utterance_id": uid, "dialogue_id": u["dialogue_id"], "encoder_lineage": res["encoder_lineage"], "positions": res["rows"],
                   "logits_index": {k: typed_sha(v) for k, v in res["logits"].items()}, "arrays_index": {k: typed_sha(v) for k, v in res["arrays"].items()},
                   "logits_file_sha256": file_hash(pdir / f"{i:03d}_logits.npz"), "arrays_file_sha256": file_hash(pdir / f"{i:03d}_arrays.npz"),
                   "references_used": False}
            atomic_json(pdir / f"{i:03d}.json", doc)
            print(f"P-B {i + 1}/80 {uid} q={len(res['rows'])} fails={len(fails)} {time.time() - t_u:.1f}s", flush=True)
        status["status"] = "completed" if not fails else "failed_integrity"
    except Exception as exc:
        import traceback
        status["status"] = "failed_exception"
        status["exception"] = {"reason": repr(exc), "traceback": traceback.format_exc()}
    finally:
        status["failures"] = fails[:500]
        status["n_failures"] = len(fails)
        finish(bundle, status, w0, out / "pulse_status.json")
    print(json.dumps({"status": status["status"], "elapsed": status["elapsed_sec"], "counters": counters, "failures": fails[:10]}), flush=True)
    if status["status"] != "completed":
        raise SystemExit("SRC_CF0_PILOT_INVALID: pulse integrity failure")


def cmd_seal_b(args) -> None:
    run = ROOT / RUN
    st = json.loads((run / "pulse_status.json").read_text())
    m = json.loads((run / "manifest_pulse.json").read_text())
    if st["status"] != "completed" or st["manifest_hash"] != m["manifest_hash"]:
        raise ValueError("pulse phase not completed")
    files = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted((run / "pulses").iterdir())}
    for p in ("manifest_pulse.json", "pulse_status.json", "PB_authorization.json", "direction_seal.json", "construction_audit.json"):
        files[f"{RUN}/{p}"] = file_hash(run / p)
    for p in sorted(run.glob("slurm-*")):
        if st.get("job_id") and st["job_id"] in p.name:
            files[str(p.relative_to(ROOT))] = file_hash(p)
    doc = {"schema": SCHEMA + "_pulse_seal", "status": "SEALED", "manifest_hash": m["manifest_hash"], "source_commit": m["git_commit"],
           "authorization_hash": m["authorization_hash"], "direction_seal_sha256": m["direction_seal_sha256"], "config_sha256": file_hash(ROOT / CONFIG),
           "sources": m["sources"], "files": files, "references_used": False, "git_head_at_seal": git("rev-parse", "HEAD"), "created_unix": time.time()}
    doc["seal_hash"] = digest(doc)
    out = ROOT / PSEAL
    if out.exists():
        raise FileExistsError("pulse seal exists; never overwrite")
    atomic_json(out, doc)
    print(json.dumps({"seal_hash": doc["seal_hash"], "files": len(files)}))


# ---- CPU engineering smoke (synthetic audio; not scientific inference) -----------------------------------------------

def cmd_smoke(args) -> None:
    import torch
    from csasr.inference_cf.loc0_sites import assert_no_any_site_hooks
    from csasr.inference_cf.src_cf0_pilot import energy_cfg
    from csasr.models.whisper import load_whisper
    from csasr.utils.config import load_config
    cfg = json.loads((ROOT / CONFIG).read_text())
    mc = load_config(ROOT / "configs/model/whisper_large_v3.yaml")
    mc["model"]["device"] = "cpu"
    torch.set_num_threads(int(os.environ.get("SMOKE_THREADS", "32")))
    t0 = time.time()
    bundle = load_whisper(mc)
    bundle.model.eval()
    bundle.model.requires_grad_(False)
    tok = bundle.processor.tokenizer
    rng = np.random.default_rng(7)
    n = 16000 * 6
    tt = np.arange(n) / 16000
    x = (0.05 * np.sin(2 * np.pi * 220 * tt) * (tt > 1) + 0.01 * rng.standard_normal(n)).astype(np.float32)
    toks = tok.encode(" hello world 你好 thanks", add_special_tokens=False)[:5]
    u = {"utterance_id": "SMOKE", "dialogue_id": "SMOKE_D", "content_prefix_tokens": toks, "max_t": 4}
    jq = {j: {"utterance_id": "SMOKE", "dialogue_id": "SMOKE_D", "t": t, "absolute_query": 3 + t, "content_prefix_sha256": digest(toks[:t]),
              "forced_zh_query_input_sha256": digest(CB + toks[:t])} for j, t in ((0, 2), (1, 4))}
    # SMOKE ONLY injected regions: q0 target-only, q1 target + off-target (exercise every eligibility branch)
    R = {0: {"target": {"status": "OK", "bounds": [32000, 48000]}, "off_target": {"status": "NO_MATCHED_OFFTARGET", "bounds": None}, "paired_available": False},
         1: {"target": {"status": "OK", "bounds": [16000, 32000]}, "off_target": {"status": "OK", "bounds": [64000, 80000]}, "paired_available": True}}
    groups = defaultdict(list)
    for j, r in R.items():
        for role in ("target", "off_target"):
            if r[role]["status"] == "OK":
                groups[tuple(r[role]["bounds"])].append((jq[j]["t"], j, role))
    counters = defaultdict(int)
    eng = Engine(bundle, counters)
    t1 = time.time()
    s1 = capture_utterance(eng, u=u, jq=jq, groups=dict(groups), x=x, masks=None)
    s2 = capture_utterance(eng, u=u, jq=jq, groups=dict(groups), x=x, masks=None)
    rep = set(s1["arrays"]) == set(s2["arrays"]) and all(np.array_equal(s1["arrays"][k], s2["arrays"][k]) for k in s1["arrays"])
    t2 = time.time()
    A = {(k.replace("_raw", "_r1_raw") if k.endswith("_raw") else k): v for k, v in s1["arrays"].items()}
    recs = {}
    for j, q in jq.items():
        geo, varr = directions_for_query(eng, j=j, q=q, region=R[j], A=A, ecfg=energy_cfg(cfg))
        A.update(varr)
        recs[j] = {"layers": geo}
    t3 = time.time()
    pres = pulse_utterance(eng, u=u, jq=jq, A=A, recs=recs, x=x, cfg=cfg, hist=None)
    assert_no_any_site_hooks(bundle)
    t4 = time.time()
    cells = [c for r in pres["rows"].values() for c in r["cells"].values()]
    outd = {"schema": SCHEMA + "_cpu_smoke", "synthetic_audio": True, "panel_audio_used": False, "injected_regions": True, "device": str(bundle.device),
            "load_sec": t1 - t0, "capture_sec": t2 - t1, "geometry_sec": t3 - t2, "pulse_sec": t4 - t3, "counters": dict(counters),
            "capture_fails": s1["fails"] + s2["fails"], "repeat_bitwise": bool(rep), "pulse_fails": pres["fails"],
            "geometry": {str(j): {l: {k: (v if k != "reach" else {kk: vv.get("status") for kk, vv in v.items()}) for k, v in g.items()}
                                   for l, g in recs[j]["layers"].items()} for j in recs},
            "none_zero_restore": {j: {"none": r["none_bitwise_sealed_clean"], "zero": r["zero"], "restore": r["restore_fingerprint"]} for j, r in pres["rows"].items()},
            "cells": {"total": len(cells), "forward": sum(c["forward"] for c in cells), "steered": sum(c["steered"] for c in cells),
                      "valid": sum(c["valid"] for c in cells), "no_edit_reasons": sorted({str(c.get("no_edit_reason")) for c in cells if not c["steered"]}),
                      "integrity_failures": [c["integrity_failures"] for c in cells if c["integrity_failures"]],
                      "predicted_vs_actual_mismatch": sum(bool(c["predicted_reachable"]) != bool(c["valid"]) for c in cells),
                      "max_norm_rounding_rel": max([c.get("norm_rounding_rel", 0.0) for c in cells] or [0.0]),
                      "checks_all": {k: all(c["checks"][k] for c in cells if c["steered"]) for k in ("eta_rel", "sq_energy_rel", "solver_matches_hook", "consumed_vs_proposed")}},
            "paired_energy": {j: r["paired_energy"] for j, r in pres["rows"].items()}}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    atomic_json(Path(args.out) / "cpu_smoke.json", outd)
    print(json.dumps(outd, indent=1)[:6000])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare").add_argument("--dry-run", action="store_true")
    sub.add_parser("manifest").add_argument("--stage", required=True, choices=STAGES)
    for c in ("construct", "seal-a", "pulse", "seal-b"):
        sub.add_parser(c)
    sub.add_parser("smoke").add_argument("--out", required=True)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "manifest": cmd_manifest, "construct": cmd_construct, "seal-a": cmd_seal_a, "pulse": cmd_pulse,
     "seal-b": cmd_seal_b, "smoke": cmd_smoke}[args.cmd](args)


if __name__ == "__main__":
    main()
