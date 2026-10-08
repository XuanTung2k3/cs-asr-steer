#!/usr/bin/env python
"""S1 — Acoustic Evidence Feasibility runner (docs/inference_cf/S1_ACOUSTIC_EVIDENCE_SPEC.md, S1_CODEX_DESIGN.md,
S1_REFERENCE_FIREWALL.md, S1_PANEL.json, configs/inference_cf/s1_acoustic_evidence.json; freeze a89067b).

prepare         (CPU) freeze / panel / source / model / tokenizer / installed-AUTO / audio / R0-primary / LAC / historical
                NONE pins; plan_sealed.json holds the RUNTIME projection only (no strata, targets or competitors).
manifest        (CPU) per-stage immutable manifest (candidates after PASS_TO_S1; acoustic after the pushed candidate
                seal and S1_AUDIT: PASS (CANDIDATES)).
candidates      (GPU job 1) original audio only: one encoder per utterance, native AUTO detection once per utterance,
                M / E / AUTO own cold caches fed prompt + identical B0 prefix; at each query raw bf16 logits, processed
                float32 log-softmax, Top-5/20 per branch and unions; fresh M alignment-head attention -> predicted-EN
                target and matched off-target bounds (region choice sees no candidate, logit-from-mask or reference).
                Starts with an in-allocation reference-free cost smoke (first utterance) and stops if the forecast > 3 h.
candidate-seal  (CPU) immutable candidate seal (push before any mask).
acoustic        (GPU job 2) loads the SEALED unions only; hard-zero target / off-target waveform copies, whole masked
                encoder, cold M owner cache per mask through the identical prefix; fixed-candidate scores for all
                scorers incl. exact-reuse historical LAC masked distributions. No candidate is ever re-selected.
seal            (CPU) immutable reference-free output seal over both phases.
primary         (CPU) reference-free primary analysis (coverage, statuses, sizes, top-1 movement; no correctness).
The runner never opens reference targets, strata, competitors, evaluator outputs, oracle regions or script masks; the
GPU phases log every opened repository/data path (sys.addaudithook) for the independent PRIMARY audit.
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

SCHEMA = "s1_acoustic_evidence_v1"
FREEZE = "a89067b55ca89097ca5e0d38f429ed678d986179"
CONFIG = "configs/inference_cf/s1_acoustic_evidence.json"
PANEL = "docs/inference_cf/S1_PANEL.json"
FROZEN = (CONFIG, PANEL, "docs/inference_cf/S1_ACOUSTIC_EVIDENCE_SPEC.md", "docs/inference_cf/S1_CODEX_DESIGN.md",
          "docs/inference_cf/S1_REFERENCE_FIREWALL.md", "tests/test_s1_freeze_contract.py")
BASE = "results/inference_cf/s1"
RUN = f"{BASE}/run1"
PLAN = f"{BASE}/plan_sealed.json"
PRERUN = f"{BASE}/prerun_audit.json"
CAND_SEAL = f"{RUN}/candidates_sealed.json"
CAND_AUDIT = f"{BASE}/candidates_audit.json"
OUTPUT_SEAL = f"{BASE}/output_seal.json"
PRIMARY = f"{BASE}/primary_analysis.json"
LAC_SEAL = "results/inference_cf/p2sel_lac/candidates_sealed.json"
LAC_MANIFEST = "results/inference_cf/p2sel_lac/lac0_run1/manifest.json"
HIST_NONE = {"st_loc0": ("results/inference_cf/st_loc0/run1/calibration/baseline_logits.npz", "results/inference_cf/st_loc0/output_seal.json"),
             "st_prompt_r1": ("results/inference_cf/st_prompt_r1/runA/capture/none_logits.npz", "results/inference_cf/st_prompt_r1/output_seal_A.json")}
P2DIR_SEALED = "results/inference_cf/p2dir/exp1_run1/directions_sealed.json"
CB = [50258, 50260, 50360, 50364]
CE = [50258, 50259, 50360, 50364]
EOS = 50257
STAGES = ("candidates", "acoustic")
R0_KEY = "track_heard_intervals"            # the only R0 primary array this study reads
HEARD_MASS_MIN = 0.5                         # config regions.attention: "require raw heard mass >=0.5"


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def committed(rel: str) -> bool:
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return git("ls-files", rel) == rel and "sha256:" + hashlib.sha256(blob).hexdigest() == file_hash(ROOT / rel)


def at_freeze(rel: str) -> bool:
    blob = subprocess.run(["git", "show", f"{FREEZE}:{rel}"], cwd=ROOT, capture_output=True).stdout
    return "sha256:" + hashlib.sha256(blob).hexdigest() == file_hash(ROOT / rel)


def arr_sha(a: np.ndarray) -> str:
    """P2-SEL-LAC convention: SHA256 of the contiguous raw bytes (used for waveform and unmasked-vector identities)."""
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


def logsoftmax_processed(raw: np.ndarray, t: int, suppress, begin) -> tuple[np.ndarray, np.ndarray]:
    """Exact historical cached._processed then float32 full-vocabulary log_softmax (CPU torch)."""
    import torch
    from experiments.inference_cf_cached import _processed
    proc = _processed(torch.from_numpy(np.asarray(raw, dtype=np.float32).copy()), t, suppress, begin)
    return proc.numpy(), torch.log_softmax(proc, dim=-1).numpy()


# ---- CPU prepare -----------------------------------------------------------------------------------------------------

def native_auto_prompt(gen, detected: int, encoder=None) -> list[int]:
    """Installed WhisperGenerationMixin._retrieve_init_tokens with a stub detector (no model forward)."""
    import torch
    from types import SimpleNamespace
    from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
    calls = []

    def detect(**kw):
        calls.append(kw)
        return torch.tensor([int(detected)])
    stub = SimpleNamespace(device=torch.device("cpu"), detect_language=detect)
    enc = object() if encoder is None else encoder
    toks = WhisperGenerationMixin._retrieve_init_tokens(stub, None, 1, gen, SimpleNamespace(forced_decoder_ids=None), 3000,
                                                         {"encoder_outputs": enc})
    if len(calls) != 1 or calls[0]["encoder_outputs"] is not enc:
        raise ValueError("native AUTO detection not called exactly once with the original encoder")
    return [int(x) for x in toks[0].tolist()]


def auto_generation_config(mdir: str):
    from transformers import GenerationConfig
    gen = GenerationConfig.from_pretrained(mdir, local_files_only=True)
    gen.language, gen.task, gen.return_timestamps, gen.forced_decoder_ids = None, "transcribe", False, None
    return gen


def cmd_prepare(args) -> None:
    import soundfile as sf
    import transformers
    from transformers import GenerationConfig, WhisperProcessor
    from transformers.models.whisper import generation_whisper
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.inference_cf.s1_evidence import runtime_projection
    from csasr.models.whisper import load_audio
    import experiments.inference_cf_p2tta0 as t0run
    cfg = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    mdir = cfg["model"]["dir"]
    checks = {"frozen_committed_unchanged": all(committed(p) and at_freeze(p) for p in FROZEN),
              "panel_identity": digest({k: v for k, v in P.items() if k != "identity_hash"}) == P["identity_hash"] == cfg["panel_identity_hash"],
              "runtime_membership": digest(P["runtime_queries"]) == cfg["runtime_membership_hash"] == P["runtime_membership_hash"],
              "counts": (len(P["runtime_queries"]), len(P["utterances"]), len({u["dialogue_id"] for u in P["utterances"]})) == (180, 80, 20),
              "sources": all(file_hash(ROOT / p) == h for p, h in cfg["source_sha256"].items()),
              "model_files": all(file_hash(Path(mdir) / n) == h for n, h in cfg["model"]["files"].items()),
              "installed_generation_source": file_hash(cfg["conditions"]["installed_generation_source"]) == cfg["conditions"]["installed_generation_file_sha256"]
              and Path(generation_whisper.__file__).resolve() == Path(cfg["conditions"]["installed_generation_source"]).resolve()}
    proc = WhisperProcessor.from_pretrained(mdir, local_files_only=True)
    tok = proc.tokenizer
    gen = GenerationConfig.from_pretrained(mdir, local_files_only=True)
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    checks["generation_config"] = (gen.decoder_start_token_id == 50258 and gen.task_to_id["transcribe"] == 50360 and gen.no_timestamps_token_id == 50364
                                   and tok.eos_token_id == EOS and [list(x) for x in gen.alignment_heads] == cfg["regions"]["alignment_heads"]
                                   and sorted(gen.lang_to_id.values()) == cfg["regions"]["source_config"]["language_ids"]
                                   and gen.lang_to_id["<|en|>"] == 50259 and gen.lang_to_id["<|zh|>"] == 50260)
    checks["special_ids_above_eos"] = all(int(i) >= EOS for i in tok.all_special_ids)
    checks["prompts"] = cfg["conditions"]["M"] == CB and cfg["conditions"]["E"] == CE
    gauto = auto_generation_config(mdir)
    checks["native_auto_prompt_stub"] = all(native_auto_prompt(gauto, d) == [50258, d, 50360, 50364] for d in (50259, 50260, 50265, 50358))
    checks["off_target_guard_text"] = cfg["counterfactual"]["off_target"]["attention_mass_max"] == "min(0.10,target_mass/2)"
    checks["heard_mass_guard_text"] = "require raw heard mass >=0.5" in cfg["regions"]["attention"]
    rp = runtime_projection(P)
    by_u = {u["utterance_id"]: u for u in rp["utterances"]}
    ok = True
    for q in rp["runtime_queries"]:
        u = by_u[q["utterance_id"]]
        b = json.loads((ROOT / u["baseline_row"]).read_text())["systems"][u["baseline_system"]]
        prefix = b["tokens"][:q["t"]]
        ok &= (file_hash(ROOT / u["baseline_row"]) == u["baseline_row_sha256"] and u["content_prefix_tokens"] == b["tokens"][:u["max_t"]]
               and q["absolute_query"] == 4 + q["t"] - 1 and q["t"] >= 1 and digest(prefix) == q["content_prefix_sha256"]
               and digest(CB + prefix) == q["forced_zh_query_input_sha256"] and digest(CE + prefix) == q["forced_en_query_input_sha256"])
    checks["runtime_prefixes"] = bool(ok)
    audio_ok, geo, wav_sha = True, True, {}
    for u in rp["utterances"]:
        g = u["audio_geometry"]
        info = sf.info(u["audio_path"])
        x = load_audio(u["audio_path"], 16000)
        wav_sha[u["utterance_id"]] = arr_sha(x)
        audio_ok &= (t0run.audio_fingerprint_64k(u["audio_path"]) == u["audio_sha256"] and file_hash(u["audio_path"]) == u["audio_full_file_sha256"])
        geo &= (info.samplerate == g["source_sample_rate"] == 16000 and info.frames == g["source_frames"] and info.channels == g["channels"]
                and x.dtype == np.float32 and len(x) == g["resampled_num_samples"] and g["baseline_heard_samples"] == min(len(x), 480000))
    checks["audio_identity"], checks["audio_geometry"] = bool(audio_ok), bool(geo)
    # R0 primary arrays: hash, seal membership, only the heard-interval track is read
    r0seal = json.loads((ROOT / P["r0_output_seal"]["path"]).read_text())
    r0ok = file_hash(ROOT / P["r0_output_seal"]["path"]) == P["r0_output_seal"]["file_sha256"]
    for u in P["utterances"]:
        r0ok &= (file_hash(ROOT / u["r0_primary_arrays"]) == u["r0_primary_arrays_sha256"] == r0seal["files"][u["r0_primary_arrays"]]
                 and file_hash(ROOT / u["r0_primary_row"]) == u["r0_primary_row_sha256"] == r0seal["files"][u["r0_primary_row"]])
        with np.load(ROOT / u["r0_primary_arrays"]) as z:
            iv = np.asarray(z[R0_KEY], dtype=np.int64)
        h = u["audio_geometry"]["baseline_heard_samples"]
        r0ok &= bool(iv.ndim == 2 and iv.shape[1] == 3 and iv[0, 0] == 0 and iv[-1, 1] == h and np.all(iv[1:, 0] == iv[:-1, 1])
                     and set(iv[:, 2].tolist()) <= {0, 1, 2})
    checks["r0_primary_sources"] = bool(r0ok)
    checks["r0_primary_audit_pass"] = json.loads((ROOT / "results/inference_cf/r0/primary_audit.json").read_text())["verdict"] == "R0_AUDIT: PASS (PRIMARY)"
    # historical LAC exact-reuse pre-check (CPU, provenance only; masked-logit values are not inspected here)
    lseal = json.loads((ROOT / LAC_SEAL).read_text())
    lrows = {(r["utterance_id"], int(r["t"])): r for r in lseal["rows"]}
    lman = json.loads((ROOT / LAC_MANIFEST).read_text())
    lac_ok = (digest({k: v for k, v in lseal.items() if k != "seal_hash"}) == lseal["seal_hash"] == lman["seal_hash"]
              and lman["model"]["files"] == {k: v for k, v in cfg["model"]["files"].items() if k in lman["model"]["files"]}
              and set(lman["model"]["files"]) == set(cfg["model"]["files"])
              and lman["suppression"] == {"suppress": sup, "begin": beg} and lman["feature_extractor"] == proc.feature_extractor.to_dict())
    lac_src = {r["utterance_id"]: r for r in P["lac_sources"]}
    p2s = json.loads((ROOT / P2DIR_SEALED).read_text())
    for u in rp["utterances"]:
        src = lac_src[u["utterance_id"]]
        lac_ok &= file_hash(ROOT / src["row"]) == src["row_sha256"] and file_hash(ROOT / src["logits"]) == src["logits_sha256"]
        row = json.loads((ROOT / src["row"]).read_text())
        lac_ok &= row["status"] == "ok" and row["identity"] == u["utterance_id"] and row["manifest_hash"] == lman["manifest_hash"] and row["logits_sha256"] == src["logits_sha256"]
        with np.load(ROOT / src["logits"]) as z:
            names = set(z.files)
        for q in [q for q in rp["runtime_queries"] if q["utterance_id"] == u["utterance_id"]]:
            lr, pr = lrows[(u["utterance_id"], q["t"])], row["positions"][str(q["t"])]
            lac_ok &= (lr["input_ids"] == CB + u["content_prefix_tokens"][:q["t"]] and lr["query"] == q["absolute_query"] and pr["query"] == q["absolute_query"]
                       and pr["W_star"] == lr["W_star"] and pr["heard_samples"] == lr["heard_samples"] == u["audio_geometry"]["baseline_heard_samples"]
                       and pr["x_sha256"] == wav_sha[u["utterance_id"]] and pr["fed_equals_input_ids"] and pr["query_matches"] and pr["logits_finite"]
                       and pr["changed"] and pr["outside_left_equal"] and pr["outside_right_equal"] and pr["inside_positive_zero"] and pr["vocab"] == 51866
                       and f"t{q['t']}_masked" in names and p2s["files"].get(lr["unmasked_logits_file"].split("exp1_run1/")[1]) == lr["unmasked_logits_sha256"]
                       and file_hash(ROOT / lr["unmasked_logits_file"]) == lr["unmasked_logits_sha256"])
    checks["lac_exact_reuse_provenance"] = bool(lac_ok)
    # historical NONE comparators (sealed) agree bitwise with each other and with the LAC unmasked source vectors
    hist = {}
    for name, (path, seal) in HIST_NONE.items():
        sd = json.loads((ROOT / seal).read_text())
        if sd["files"].get(path) != file_hash(ROOT / path):
            checks[f"hist_none_sealed:{name}"] = False
            continue
        checks[f"hist_none_sealed:{name}"] = True
        with np.load(ROOT / path) as z:
            hist[name] = {k: z[k] for k in z.files}
    agree = len(hist) == 2 and all(np.array_equal(hist["st_loc0"][f"q{j:03d}"], hist["st_prompt_r1"][f"q{j:03d}"]) for j in range(180))
    for j, q in enumerate(rp["runtime_queries"]):
        lr = lrows[(q["utterance_id"], q["t"])]
        with np.load(ROOT / lr["unmasked_logits_file"]) as z:
            v = z[f"t{q['t']}_none"]
        agree &= bool(np.array_equal(v, hist["st_loc0"][f"q{j:03d}"]) and arr_sha(unpack(v)) == lr["unmasked_logit_vector_sha256"])
    checks["hist_none_bitwise_agreement"] = bool(agree)
    part = tokenizer_partition(tok)
    steps = sum(1 + u["max_t"] for u in rp["utterances"])
    plan = {"schema": SCHEMA + "_plan", "freeze": FREEZE, "config_sha256": file_hash(ROOT / CONFIG), "config_hash": digest(cfg),
            "panel_identity_hash": P["identity_hash"], "checks": checks, "runtime": rp, "waveform_sha256": wav_sha,
            "suppression": {"suppress": sup, "begin": beg}, "suppression_hash": digest({"suppress": sup, "begin": beg}),
            "partition_hash": part["hash"], "transformers": transformers.__version__,
            "forecast": {"original_encoders": 80, "original_cached_steps_max": 3 * steps, "auto_detection_queries": 80,
                         "masked_encoders_max": 360, "masked_cached_steps_max": 2 * sum(1 + q["t"] for q in rp["runtime_queries"])},
            "references_used": False, "created_unix": time.time()}
    plan["forecast"]["matches_config"] = (plan["forecast"]["original_cached_steps_max"] == cfg["compute"]["original_branch_cached_steps_max"]
                                          and plan["forecast"]["masked_cached_steps_max"] == cfg["compute"]["masked_cached_steps_max"])
    checks["compute_allocation_matches"] = plan["forecast"]["matches_config"]
    plan["plan_hash"] = digest(plan)
    out = ROOT / PLAN
    if out.exists():
        raise FileExistsError("plan exists; never overwrite")
    if not all(checks.values()):
        raise SystemExit("S1_BLOCKED/INVALID at prepare: " + json.dumps([k for k, v in checks.items() if not v]))
    atomic_json(out, plan)
    print(json.dumps({"plan_hash": plan["plan_hash"], "checks": checks}))


SOURCES = FROZEN + (PLAN, "src/csasr/inference_cf/s1_evidence.py", "src/csasr/inference_cf/r0_regions.py",
                    "src/csasr/inference_cf/lexical_compatibility.py", "src/csasr/inference_cf/core_r2.py", "src/csasr/inference_cf/core.py",
                    "src/csasr/models/whisper.py", "src/csasr/lss/sites.py", "experiments/inference_cf_cached.py", "experiments/acoustic_s1.py",
                    "experiments/acoustic_s1_evaluate.py", "experiments/acoustic_s1_audit.py", "slurm/acoustic_s1.sbatch", "tests/test_s1_impl.py")


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    stage = args.stage
    need = [PRERUN] + ([CAND_SEAL, CAND_AUDIT] if stage == "acoustic" else [])
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, *need)
    if dirty:
        raise ValueError("commit sources before the manifest:\n" + dirty)
    if json.loads((ROOT / PRERUN).read_text())["verdict"] != "PASS_TO_S1":
        raise ValueError("pre-run audit did not pass")
    cfg = json.loads((ROOT / CONFIG).read_text())
    plan = json.loads((ROOT / PLAN).read_text())
    man = {"schema": SCHEMA, "stage": stage, "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "config_sha256": file_hash(ROOT / CONFIG), "config_hash": digest(cfg), "panel_identity_hash": plan["panel_identity_hash"],
           "plan_hash": plan["plan_hash"], "output_root": RUN, "reference_access_boundary":
           "runner reads runtime projection, audio, R0 heard intervals, historical M/LAC logits only; evaluator only after pushed output seal + PRIMARY PASS",
           "environment": prep.environment(), "model": {"dir": cfg["model"]["dir"], "files": prep.model_hashes()},
           "sources": {p: file_hash(ROOT / p) for p in SOURCES + tuple(need)}, "trainable_parameters": 0, "references_used": False,
           "created_unix": time.time()}
    if stage == "acoustic":
        seal = json.loads((ROOT / CAND_SEAL).read_text())
        if json.loads((ROOT / CAND_AUDIT).read_text())["verdict"] != "S1_AUDIT: PASS (CANDIDATES)":
            raise ValueError("candidate audit did not pass")
        sc = git("log", "-n1", "--format=%H", "--", CAND_SEAL)
        if subprocess.run(["git", "merge-base", "--is-ancestor", sc, "origin/cs-asr-steer-inf"], cwd=ROOT).returncode != 0:
            raise ValueError("candidate seal is not on the remote branch")
        man.update(candidate_seal_hash=seal["seal_hash"], candidate_seal_commit=sc)
    man["manifest_hash"] = digest(man)
    out = ROOT / RUN / f"manifest_{stage}.json"
    if out.exists():
        raise FileExistsError("manifest exists; never overwrite")
    atomic_json(out, man)
    print(json.dumps({"stage": stage, "manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}))


# ---- GPU phases ------------------------------------------------------------------------------------------------------

OPENED: set = set()


def _audit_open(event, args) -> None:
    """Runtime file-open log (firewall evidence): every opened data/repository path outside the Python install."""
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
            raise ValueError("HEAD differs from the manifest commit outside results/inference_cf/s1/")
    plan = json.loads((ROOT / PLAN).read_text())
    cfg = json.loads((ROOT / CONFIG).read_text())
    if plan["plan_hash"] != m["plan_hash"] or digest(cfg) != m["config_hash"]:
        raise ValueError("plan/config")
    return m, plan, cfg


def load_model(cfg, plan):
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
    gen = bundle.model.generation_config
    if digest({"suppress": list(gen.suppress_tokens or []), "begin": list(gen.begin_suppress_tokens or [])}) != plan["suppression_hash"]:
        raise ValueError("suppression")
    if [list(x) for x in gen.alignment_heads] != cfg["regions"]["alignment_heads"]:
        raise ValueError("alignment heads")
    return bundle


def weight_probe(bundle) -> str:
    h = hashlib.sha256()
    for k, v in bundle.model.state_dict().items():
        if k.endswith(("final_layer_norm.weight", "proj_out.weight", "embed_positions.weight")) or ".layers.16." in k:
            h.update(k.encode())
            h.update(v.detach().float().cpu().numpy().tobytes())
    return "sha256:" + h.hexdigest()


def encode(bundle, x: np.ndarray, counters: dict, key: str = "encoder_calls"):
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.lexical_compatibility import waveform_model_inputs
    feats = waveform_model_inputs(bundle, x)
    with torch.inference_mode():
        h = bundle.model.model.encoder(input_features=feats["input_features"], attention_mask=feats["attention_mask"]).last_hidden_state
    counters[key] += 1
    lineage = {"features_sha256": typed_sha(feats["input_features"].float().cpu().numpy()),
               "attention_mask_sha256": typed_sha(feats["attention_mask"].cpu().numpy()),
               "encoder_sha256": typed_sha(h.float().cpu().numpy())}
    return BaseModelOutput(last_hidden_state=h), lineage


def finish_runtime(bundle, runtime: dict, w0: str, out: Path, name: str) -> None:
    import torch
    from csasr.lss.sites import assert_no_site_hooks
    assert_no_site_hooks(bundle)
    runtime.update(end_unix=time.time(), peak_vram_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                   peak_vram_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                   model_grads_none=all(p.grad is None for p in bundle.model.parameters()),
                   requires_grad_any=any(p.requires_grad for p in bundle.model.parameters()), training_mode=bool(bundle.model.training),
                   weights_unchanged_probe=weight_probe(bundle) == w0, top_forward_hooks=len(bundle.model._forward_hooks),
                   opened_paths=sorted(OPENED))
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / name, runtime)


def detect_auto(bundle, enc, gauto, counters) -> dict:
    """Native installed detection (one SOT query on the ORIGINAL encoder) through _retrieve_init_tokens; the SOT language
    logits are recorded passively by a top-level forward hook to verify the lowest-ID argmax."""
    import torch
    seen = []

    def grab(module, args, kwargs, output):
        seen.append({"decoder_input_ids": kwargs.get("decoder_input_ids").tolist(), "logits": output.logits[0, -1].float().cpu()})
    hd = bundle.model.register_forward_hook(grab, with_kwargs=True)
    try:
        with torch.inference_mode():
            toks = bundle.model._retrieve_init_tokens(None, 1, gauto, bundle.model.config, 3000, {"encoder_outputs": enc})
    finally:
        hd.remove()
    counters["auto_detection_queries"] += len(seen)
    ids = sorted(int(i) for i in gauto.lang_to_id.values())
    if len(seen) != 1 or seen[0]["decoder_input_ids"] != [[50258]]:
        raise ValueError("native detection did not run exactly one SOT query")
    lv = seen[0]["logits"][ids].numpy()
    lowest = ids[int(np.flatnonzero(lv == lv.max())[0])]
    prompt = [int(x) for x in toks[0].tolist()]
    return {"prompt": prompt, "detected": prompt[1], "lowest_id_argmax": lowest, "language_ids": ids,
            "language_logits": [float(v) for v in lv], "agrees": prompt[1] == lowest and prompt == [50258, prompt[1], 50360, 50364]}


def candidate_utterance(bundle, *, u: dict, jq: list, x: np.ndarray, ivs: np.ndarray, cfg: dict, sup, beg, gauto, counters: dict,
                        hist: dict | None) -> dict:
    """Original-audio candidate pass for one utterance (reference-free). jq = [(j, runtime query)] of this utterance."""
    from csasr.inference_cf.s1_evidence import BRANCHES, heard_attention, membership_digest, select_offtarget, select_target, topk, union
    from csasr.lss.sites import assert_no_site_hooks
    from experiments.inference_cf_cached import Branch
    heard = min(len(x), 480000)
    enc, lin = encode(bundle, x, counters)
    auto = detect_auto(bundle, enc, gauto, counters)
    alias = {50260: "M", 50259: "E"}.get(auto["detected"])
    prompts = {"M": CB, "E": CE, "AUTO": auto["prompt"]}
    run_b = [b for b in BRANCHES if not (b == "AUTO" and alias)]
    br = {b: Branch(bundle, enc, prompts[b], b) for b in run_b}
    counters["auto_alias"] += int(alias is not None)
    toks = u["content_prefix_tokens"]
    jt = {q["t"]: (j, q) for j, q in jq}
    arrays, qrecs, ident_fail = {}, [], []
    for t in range(u["max_t"] + 1):
        outs = {}
        for b in run_b:
            feed = prompts[b] if t == 0 else [toks[t - 1]]
            assert_no_site_hooks(bundle)
            lg, hd, _ = br[b].step(feed, attention=(b == "M"))
            counters["decoder_steps"][b] += 1
            outs[b] = (lg, hd)
        if t not in jt:
            continue
        j, q = jt[t]
        if alias:
            outs["AUTO"] = outs[alias]
        rec = {"j": j, "t": t, "query": q["absolute_query"], "branches": {}, "unions": {}}
        proc, logp, raw = {}, {}, {}
        for b in BRANCHES:
            fed = br[alias if (b == "AUTO" and alias) else b].fed
            lg = outs[b][0]
            raw[b] = lg.numpy().astype(np.float32)
            proc[b], logp[b] = logsoftmax_processed(raw[b], t, sup, beg)
            legal = np.isfinite(proc[b])
            rec["branches"][b] = {"prompt": prompts[b], "alias_of": alias if b == "AUTO" else None, "fed_sha256": digest(fed),
                                  "fed_is_prompt_plus_prefix": fed == prompts[b] + toks[:t], "query_index": len(fed) - 1,
                                  "raw_finite": bool(np.isfinite(raw[b]).all()), "legal_count": int(legal.sum()),
                                  "logp_finite_on_legal": bool(np.isfinite(logp[b][legal]).all()), "vocab": int(raw[b].size),
                                  "raw_sha256": typed_sha(pack(lg)), "logp_sha256": typed_sha(logp[b]),
                                  "top": {str(k): topk(proc[b], k) for k in cfg["candidates"]["budgets"]}}
            if not (b == "AUTO" and alias):
                arrays[f"q{j:03d}_{b}_raw"] = pack(lg)
                arrays[f"q{j:03d}_{b}_logp"] = logp[b]
        rec["branches"]["M"]["input_sha_matches_panel"] = digest(br["M"].fed) == q["forced_zh_query_input_sha256"]
        rec["branches"]["E"]["input_sha_matches_panel"] = digest(br["E"].fed) == q["forced_en_query_input_sha256"]
        rec["same_content_prefix"] = (all(rec["branches"][b]["fed_is_prompt_plus_prefix"] for b in BRANCHES) and digest(toks[:t]) == q["content_prefix_sha256"]
                                      and rec["branches"]["M"]["input_sha_matches_panel"] and rec["branches"]["E"]["input_sha_matches_panel"])
        rec["query_matches"] = all(rec["branches"][b]["query_index"] == q["absolute_query"] for b in BRANCHES)
        rec["M_bitwise_hist_none"] = None if hist is None else bool(np.array_equal(arrays[f"q{j:03d}_M_raw"], hist[f"q{j:03d}"]))
        if not (rec["same_content_prefix"] and rec["query_matches"] and rec["M_bitwise_hist_none"] is not False):
            ident_fail.append(j)
        for k in cfg["candidates"]["budgets"]:
            un = union({b: rec["branches"][b]["top"][str(k)] for b in BRANCHES}, EOS)
            ids = un["ids"]
            un["membership_sha256"] = membership_digest(ids)
            un["logp"] = {b: [float(logp[b][c]) for c in ids] for b in BRANCHES}
            un["raw"] = {b: [float(raw[b][c]) for c in ids] for b in BRANCHES}
            rec["unions"][str(k)] = un
        arrays[f"q{j:03d}_M_heads"] = pack(outs["M"][1])
        att = heard_attention(outs["M"][1].float().numpy(), heard, HEARD_MASS_MIN)
        tgt = select_target(att, ivs, x, heard, cfg)
        off = select_offtarget(att, ivs, x, heard, tgt, cfg)
        rec["region"] = {"heard": heard, "frames": att["frames"], "raw_heard_mass": att["raw_mass"], "mapped": att["mapped"],
                         "heads_sha256": typed_sha(arrays[f"q{j:03d}_M_heads"]), "target": tgt, "off_target": off,
                         "paired_available": tgt["status"] == "OK" and off["status"] == "OK"}
        qrecs.append(rec)
    lineage_ok = all(br[b].positions == list(range(br[b].length)) for b in run_b) and len({id(br[b].cache) for b in run_b}) == len(run_b)
    return {"queries": qrecs, "arrays": arrays, "ident_fail": ident_fail, "auto": auto, "alias": alias, "encoder_lineage": lin,
            "branch_lineage_ok": bool(lineage_ok), "heard": heard}


def cmd_candidates(args) -> None:
    sys.addaudithook(_audit_open)
    import torch
    from csasr.inference_cf.s1_evidence import BRANCHES
    from csasr.models.whisper import load_audio
    m, plan, cfg = load_run("candidates")
    out = ROOT / RUN
    cdir = out / "candidates"
    if cdir.exists() and any(cdir.iterdir()):
        raise FileExistsError("candidate outputs exist; never overwrite (use a new numbered run)")
    cdir.mkdir(parents=True, exist_ok=True)
    bundle = load_model(cfg, plan)
    gen = bundle.model.generation_config
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    gauto = auto_generation_config(cfg["model"]["dir"])
    w0 = weight_probe(bundle)
    counters = {"encoder_calls": 0, "decoder_steps": {b: 0 for b in BRANCHES}, "auto_detection_queries": 0, "auto_alias": 0,
                "masked_encoder_calls": 0, "autograd_calls": 0, "optimizer_steps": 0, "steering_hooks": 0, "lid_calls": 0}
    runtime = {"stage": "candidates", "manifest_hash": m["manifest_hash"], "job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
               "gpu": torch.cuda.get_device_name(0), "start_unix": time.time(), "status": "running", "counters": counters}
    atomic_json(out / "candidates_runtime.json", runtime)
    torch.cuda.reset_peak_memory_stats()
    with np.load(ROOT / HIST_NONE["st_loc0"][0]) as z:
        hist = {k: z[k] for k in z.files}
    Q, Ut = plan["runtime"]["runtime_queries"], plan["runtime"]["utterances"]
    by_u = defaultdict(list)
    for j, q in enumerate(Q):
        by_u[q["utterance_id"]].append((j, q))
    ident_fail, total_steps = [], 3 * sum(1 + u["max_t"] for u in Ut)
    for i, u in enumerate(Ut):
        t_u = time.time()
        uid = u["utterance_id"]
        x = load_audio(u["audio_path"], 16000)
        with np.load(ROOT / u["r0_primary_arrays"]) as z:
            ivs = np.asarray(z[R0_KEY], dtype=np.int64)
        res = candidate_utterance(bundle, u=u, jq=by_u[uid], x=x, ivs=ivs, cfg=cfg, sup=sup, beg=beg, gauto=gauto, counters=counters, hist=hist)
        ident_fail += res["ident_fail"]
        arrays = res["arrays"]
        np.savez_compressed(cdir / f"{i:03d}.npz", **arrays)
        doc = {"schema": SCHEMA + "_candidates_row", "manifest_hash": m["manifest_hash"], "index": i, "utterance_id": uid,
               "dialogue_id": u["dialogue_id"], "audio_full_file_sha256": u["audio_full_file_sha256"], "audio_sha256": u["audio_sha256"],
               "waveform_sha256": arr_sha(x), "waveform_matches_plan": arr_sha(x) == plan["waveform_sha256"][uid], "n_samples": int(len(x)),
               "heard": res["heard"], "r0_arrays": u["r0_primary_arrays"], "r0_arrays_sha256": file_hash(ROOT / u["r0_primary_arrays"]),
               "r0_track_heard_intervals": ivs.tolist(), "encoder_lineage": res["encoder_lineage"], "auto": res["auto"], "auto_alias": res["alias"],
               "branch_lineage_ok": res["branch_lineage_ok"], "queries": res["queries"],
               "arrays_index": {k: {"dtype": str(v.dtype), "shape": list(v.shape), "sha256": typed_sha(v)} for k, v in arrays.items()},
               "arrays_file_sha256": file_hash(cdir / f"{i:03d}.npz"), "elapsed_sec": time.time() - t_u}
        if not (doc["waveform_matches_plan"] and doc["branch_lineage_ok"] and res["auto"]["agrees"]):
            ident_fail.append(f"utterance:{uid}")
        atomic_json(cdir / f"{i:03d}.json", doc)
        print(f"S1 candidates {i + 1}/{len(Ut)} {uid} auto={res['auto']['detected']} alias={res['alias']} {time.time() - t_u:.1f}s", flush=True)
        if i == 0:      # in-allocation reference-free cost smoke -> forecast for both jobs
            per = (time.time() - runtime["start_unix"]) / max(1, sum(counters["decoder_steps"].values()))
            runtime["smoke"] = {"first_utterance_sec": time.time() - t_u, "sec_per_decoder_step_incl_overhead": per,
                                "forecast_job1_sec": per * total_steps, "forecast_job2_sec": per * plan["forecast"]["masked_cached_steps_max"] * 1.5}
            atomic_json(out / "candidates_runtime.json", runtime)
            if max(runtime["smoke"]["forecast_job1_sec"], runtime["smoke"]["forecast_job2_sec"]) > cfg["compute"]["hard_seconds_per_job"]:
                runtime["status"] = "stopped_compute_forecast"
                finish_runtime(bundle, runtime, w0, out, "candidates_runtime.json")
                raise SystemExit("compute forecast > 3 h: stop for an explicit pre-outcome compute revision")
    runtime["identity_failures"] = ident_fail
    runtime["status"] = "completed" if not ident_fail else "failed_identity"
    finish_runtime(bundle, runtime, w0, out, "candidates_runtime.json")
    print(json.dumps({"status": runtime["status"], "elapsed": runtime["elapsed_sec"], "counters": counters}), flush=True)
    if ident_fail:
        raise SystemExit(f"S1_INVALID: historical M identity / prefix / lineage failure at {ident_fail}")


def cmd_candidate_seal(args) -> None:
    run = ROOT / RUN
    rt = json.loads((run / "candidates_runtime.json").read_text())
    m = json.loads((run / "manifest_candidates.json").read_text())
    if rt["status"] != "completed" or rt["manifest_hash"] != m["manifest_hash"]:
        raise ValueError("candidate phase not completed")
    plan = json.loads((ROOT / PLAN).read_text())
    rows, members = [], {}
    for i, u in enumerate(plan["runtime"]["utterances"]):
        d = json.loads((run / "candidates" / f"{i:03d}.json").read_text())
        if d["utterance_id"] != u["utterance_id"] or d["manifest_hash"] != m["manifest_hash"] or file_hash(run / "candidates" / f"{i:03d}.npz") != d["arrays_file_sha256"]:
            raise ValueError(f"candidate row {i}")
        for q in d["queries"]:
            members[f"q{q['j']:03d}"] = {k: q["unions"][k]["membership_sha256"] for k in q["unions"]}
        rows.append(d)
    if sorted(members) != [f"q{j:03d}" for j in range(180)]:
        raise ValueError("candidate coverage")
    files = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted((run / "candidates").iterdir())}
    for p in ("manifest_candidates.json", "candidates_runtime.json"):
        files[f"{RUN}/{p}"] = file_hash(run / p)
    for p in sorted(run.glob("slurm-*")):
        if rt.get("job_id") and rt["job_id"] in p.name:
            files[str(p.relative_to(ROOT))] = file_hash(p)
    doc = {"schema": SCHEMA + "_candidate_seal", "status": "SEALED", "plan_hash": plan["plan_hash"], "manifest_hash": m["manifest_hash"],
           "config_sha256": file_hash(ROOT / CONFIG), "members": members, "files": files, "references_used": False,
           "git_head_at_seal": git("rev-parse", "HEAD"), "created_unix": time.time()}
    doc["seal_hash"] = digest(doc)
    out = ROOT / CAND_SEAL
    if out.exists():
        raise FileExistsError("candidate seal exists; never overwrite")
    atomic_json(out, doc)
    print(json.dumps({"seal_hash": doc["seal_hash"], "files": len(files)}))


def acoustic_utterance(bundle, *, u: dict, cd: dict, carr: dict, x: np.ndarray, cfg: dict, sup, beg, counters: dict, lac_ctx: dict,
                       members: dict, rq: list) -> dict:
    """Masked counterfactual readouts and fixed-candidate scores for one utterance. Only SEALED union IDs are scored."""
    from csasr.inference_cf.lexical_compatibility import hard_mask, waveform_model_inputs
    from csasr.inference_cf.s1_evidence import membership_digest, ranking, scores, shuffled
    from csasr.lss.sites import assert_no_site_hooks
    from experiments.inference_cf_cached import Branch
    uid = u["utterance_id"]
    heard = min(len(x), 480000)
    if arr_sha(x) != cd["waveform_sha256"] or cd["utterance_id"] != uid:
        raise ValueError("waveform lineage")
    f0 = waveform_model_inputs(bundle, x)          # zero-change audio: identical preprocessing, no encoder call
    zero_change = typed_sha(f0["input_features"].float().cpu().numpy()) == cd["encoder_lineage"]["features_sha256"]
    lseal, lrow, lac, lac_seal_ok = lac_ctx["lseal"], lac_ctx["lrow"], lac_ctx["lac"], lac_ctx["seal_ok"]
    groups = defaultdict(list)
    for q in cd["queries"]:
        for role in ("target", "off_target"):
            r = q["region"][role]
            if r["status"] == "OK":
                groups[tuple(r["bounds"])].append((q["t"], q["j"], role))
    masked, mrec, fails = {}, {}, []
    toks = u["content_prefix_tokens"]
    for (a, b) in sorted(groups):
        xm = hard_mask(x, a, b)
        g = {"bounds": [a, b], "x_mask_sha256": arr_sha(xm), "outside_left_equal": xm[:a].tobytes() == x[:a].tobytes(),
             "outside_right_equal": xm[b:].tobytes() == x[b:].tobytes(), "inside_positive_zero": bool(np.all(xm[a:b].view(np.uint32) == 0)),
             "changed_samples": int(np.count_nonzero(xm != x)), "within_heard": 0 <= a < b <= heard,
             "rms_before": float(np.sqrt(np.mean(x[a:b].astype(np.float64) ** 2))),
             "shape_dtype_ok": xm.shape == x.shape and xm.dtype == np.float32, "members": sorted(groups[(a, b)])}
        enc, lin = encode(bundle, xm, counters, "masked_encoder_calls")
        counters["mask_groups"] += 1
        g["lineage"] = lin
        br = Branch(bundle, enc, CB, f"mask_{a}_{b}")
        for t in sorted({t for t, _, _ in groups[(a, b)]}):
            lz = None
            while br.length < len(CB) + t:
                feed = CB if br.length == 0 else [toks[br.length - len(CB)]]
                assert_no_site_hooks(bundle)
                lz, _, _ = br.step(feed, attention=True)
                counters["masked_decoder_steps"] += 1
            if lz is None:
                raise RuntimeError("masked branch past the query")
            counters["query_readouts"] += 1
            raw = lz.numpy().astype(np.float32)
            _, lp = logsoftmax_processed(raw, t, sup, beg)
            for _, j, role in [z_ for z_ in groups[(a, b)] if z_[0] == t]:
                masked[(j, role)] = {"raw": pack(lz), "logp": lp, "fed_ok": br.fed == CB + toks[:t], "query": br.length - 1,
                                     "raw_finite": bool(np.isfinite(raw).all()), "group": f"{a}_{b}"}
        g["positions_ok"] = br.positions == list(range(br.length))
        mrec[f"{a}_{b}"] = g
        del br, enc
    arrays, qout = {}, []
    for q in cd["queries"]:
        j, t = q["j"], q["t"]
        Q = rq[j]
        lr, lp_ = lseal[(uid, t)], lrow["positions"][str(t)]
        lac_ok = bool(lr["input_ids"] == CB + toks[:t] and lr["query"] == Q["absolute_query"] == lp_["query"]
                      and arr_sha(unpack(carr[f"q{j:03d}_M_raw"])) == lr["unmasked_logit_vector_sha256"] and lp_["x_sha256"] == cd["waveform_sha256"]
                      and lp_["W_star"] == lr["W_star"] and lp_["fed_equals_input_ids"] and lp_["query_matches"] and lp_["logits_finite"]
                      and lp_["outside_left_equal"] and lp_["outside_right_equal"] and lp_["inside_positive_zero"] and lac_seal_ok
                      and f"t{t}_masked" in lac)
        _, lac_lp = logsoftmax_processed(unpack(lac[f"t{t}_masked"]), t, sup, beg)
        counters["lac_reused"] += 1
        tgt, off = q["region"]["target"], q["region"]["off_target"]
        have_t, have_o = (j, "target") in masked, (j, "off_target") in masked
        paired = tgt["status"] == "OK" and off["status"] == "OK" and have_t and have_o
        for role in ("target", "off_target"):
            if (j, role) in masked:
                arrays[f"q{j:03d}_{role}_raw"] = masked[(j, role)]["raw"]
                arrays[f"q{j:03d}_{role}_logp"] = masked[(j, role)]["logp"]
        rec = {"j": j, "t": t, "query": Q["absolute_query"], "target_status": tgt["status"], "off_target_status": off["status"],
               "target_bounds": tgt["bounds"], "off_target_bounds": off["bounds"], "paired_available": paired,
               "masked": {role: {k: v for k, v in masked[(j, role)].items() if k not in ("raw", "logp")}
                          | {"raw_sha256": typed_sha(masked[(j, role)]["raw"]), "logp_sha256": typed_sha(masked[(j, role)]["logp"])}
                          for role in ("target", "off_target") if (j, role) in masked},
               "lac": {"exact_reuse_ok": lac_ok, "W_star": lr["W_star"], "masked_raw_sha256": typed_sha(lac[f"t{t}_masked"]), "logp_sha256": typed_sha(lac_lp)},
               "K": {}}
        for k in map(str, cfg["candidates"]["budgets"]):
            un = q["unions"][k]
            ids = np.array(un["ids"], dtype=np.int64)
            if membership_digest(ids) != members[f"q{j:03d}"][k] or membership_digest(ids) != un["membership_sha256"]:
                raise ValueError("sealed union mismatch")
            l0 = carr[f"q{j:03d}_M_logp"][ids]
            sc = {"M_ORIGINAL": l0.astype(np.float64), "E_ORIGINAL": carr[f"q{j:03d}_E_logp"][ids].astype(np.float64),
                  "AUTO_ORIGINAL": carr[f"q{j:03d}_{cd['auto_alias'] or 'AUTO'}_logp"][ids].astype(np.float64),
                  "LAC_UNION": scores(l0, lac_lp[ids])}
            diag = {}
            if have_t:
                lt = masked[(j, "target")]["logp"][ids]
                diag["EN_REGION_TARGET_ONLY"] = scores(l0, lt)
                sup_t = l0.astype(np.float64) - lt.astype(np.float64)
            if have_o:
                diag["OFF_TARGET_ONLY"] = scores(l0, masked[(j, "off_target")]["logp"][ids])
            if paired:
                sc["EN_REGION"] = diag["EN_REGION_TARGET_ONLY"]
                sc["OFF_TARGET"] = diag["OFF_TARGET_ONLY"]
                sc["SHUFFLED_SUPPORT"] = l0.astype(np.float64) + shuffled(sup_t, uid, t, int(k))
            else:
                for s_ in ("EN_REGION", "OFF_TARGET", "SHUFFLED_SUPPORT"):
                    sc[s_] = sc["M_ORIGINAL"]
            sc["NO_CONTRAST"] = sc["M_ORIGINAL"]
            ranks = {s_: ranking(ids, v) for s_, v in sc.items()}
            rec["K"][k] = {"ids": ids.tolist(), "membership_sha256": membership_digest(ids), "fallback_to_M": not paired,
                           "l0": l0.astype(np.float64).tolist(),
                           "lm_target": masked[(j, "target")]["logp"][ids].astype(np.float64).tolist() if have_t else None,
                           "lm_off_target": masked[(j, "off_target")]["logp"][ids].astype(np.float64).tolist() if have_o else None,
                           "lm_lac": lac_lp[ids].astype(np.float64).tolist(),
                           "scores": {s_: np.asarray(v, dtype=np.float64).tolist() for s_, v in sc.items()},
                           "rankings": ranks, "top1": {s_: r_[0] for s_, r_ in ranks.items()},
                           "diagnostic_rankings": {s_: ranking(ids, v) for s_, v in diag.items()},
                           "finite": all(np.isfinite(v).all() for v in list(sc.values()) + list(diag.values()))}
            if not rec["K"][k]["finite"]:
                fails.append((j, k, "nonfinite"))
        if not lac_ok:
            fails.append((j, "lac_reuse"))
        for role, v in rec["masked"].items():
            if not (v["fed_ok"] and v["query"] == Q["absolute_query"] and v["raw_finite"]):
                fails.append((j, role, "masked_lineage"))
        qout.append(rec)
    for g in mrec.values():
        if not (g["outside_left_equal"] and g["outside_right_equal"] and g["inside_positive_zero"] and g["within_heard"] and g["shape_dtype_ok"] and g["positions_ok"]):
            fails.append((uid, g["bounds"], "mask_integrity"))
    if not zero_change:
        fails.append((uid, "zero_change_features"))
    return {"queries": qout, "arrays": arrays, "mask_groups": mrec, "fails": fails, "zero_change": zero_change, "heard": heard}


def cmd_acoustic(args) -> None:
    sys.addaudithook(_audit_open)
    import torch
    from csasr.models.whisper import load_audio
    m, plan, cfg = load_run("acoustic")
    run = ROOT / RUN
    seal = json.loads((ROOT / CAND_SEAL).read_text())
    if seal["seal_hash"] != m["candidate_seal_hash"] or digest({k: v for k, v in seal.items() if k != "seal_hash"}) != seal["seal_hash"]:
        raise ValueError("candidate seal")
    for p, h in seal["files"].items():
        if file_hash(ROOT / p) != h:
            raise ValueError(f"sealed candidate file changed: {p}")
    adir = run / "acoustic"
    if adir.exists() and any(adir.iterdir()):
        raise FileExistsError("acoustic outputs exist; never overwrite")
    adir.mkdir(parents=True, exist_ok=True)
    bundle = load_model(cfg, plan)
    gen = bundle.model.generation_config
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    w0 = weight_probe(bundle)
    counters = {"masked_encoder_calls": 0, "encoder_calls": 0, "masked_decoder_steps": 0, "mask_groups": 0, "query_readouts": 0,
                "lac_reused": 0, "autograd_calls": 0, "optimizer_steps": 0, "steering_hooks": 0, "lid_calls": 0, "auto_detection_queries": 0,
                "candidate_regenerations": 0}
    runtime = {"stage": "acoustic", "manifest_hash": m["manifest_hash"], "job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
               "gpu": torch.cuda.get_device_name(0), "start_unix": time.time(), "status": "running", "counters": counters}
    atomic_json(run / "acoustic_runtime.json", runtime)
    torch.cuda.reset_peak_memory_stats()
    lseal_doc = json.loads((ROOT / LAC_SEAL).read_text())
    lseal = {(r["utterance_id"], int(r["t"])): r for r in lseal_doc["rows"]}
    seal_ok = json.loads((ROOT / LAC_MANIFEST).read_text())["seal_hash"] == lseal_doc["seal_hash"]
    fails = []
    for i, u in enumerate(plan["runtime"]["utterances"]):
        t_u = time.time()
        uid = u["utterance_id"]
        cd = json.loads((run / "candidates" / f"{i:03d}.json").read_text())
        with np.load(run / "candidates" / f"{i:03d}.npz") as z:
            carr = {k: z[k] for k in z.files}
        x = load_audio(u["audio_path"], 16000)
        lrow = json.loads((ROOT / u["lac_row"]).read_text())
        with np.load(ROOT / u["lac_logits"]) as z:
            lac = {k: z[k] for k in z.files}
        res = acoustic_utterance(bundle, u=u, cd=cd, carr=carr, x=x, cfg=cfg, sup=sup, beg=beg, counters=counters,
                                 lac_ctx={"lseal": lseal, "lrow": lrow, "lac": lac, "seal_ok": seal_ok}, members=seal["members"],
                                 rq=plan["runtime"]["runtime_queries"])
        fails += res["fails"]
        arrays = res["arrays"]
        np.savez_compressed(adir / f"{i:03d}.npz", **arrays)
        doc = {"schema": SCHEMA + "_acoustic_row", "manifest_hash": m["manifest_hash"], "candidate_seal_hash": seal["seal_hash"], "index": i,
               "utterance_id": uid, "dialogue_id": u["dialogue_id"], "waveform_sha256": arr_sha(x), "heard": res["heard"],
               "zero_change_features_identical": res["zero_change"], "mask_groups": res["mask_groups"], "queries": res["queries"],
               "lac_source": {"row": u["lac_row"], "row_sha256": file_hash(ROOT / u["lac_row"]), "logits": u["lac_logits"], "logits_sha256": file_hash(ROOT / u["lac_logits"])},
               "arrays_index": {k: {"dtype": str(v.dtype), "shape": list(v.shape), "sha256": typed_sha(v)} for k, v in arrays.items()},
               "arrays_file_sha256": file_hash(adir / f"{i:03d}.npz"), "elapsed_sec": time.time() - t_u}
        atomic_json(adir / f"{i:03d}.json", doc)
        print(f"S1 acoustic {i + 1}/80 {uid} groups={len(res['mask_groups'])} {time.time() - t_u:.1f}s", flush=True)
    runtime["failures"] = [list(map(str, f)) for f in fails]
    runtime["status"] = "completed" if not fails else "failed_integrity"
    finish_runtime(bundle, runtime, w0, run, "acoustic_runtime.json")
    print(json.dumps({"status": runtime["status"], "elapsed": runtime["elapsed_sec"], "counters": counters}), flush=True)
    if fails:
        raise SystemExit(f"S1_INVALID: acoustic integrity failures {fails[:10]}")


def cmd_smoke(args) -> None:
    """CPU engineering smoke on SYNTHETIC audio (no panel audio, no S1 outcome): the real frozen model in bf16 on CPU runs
    the exact candidate_utterance / acoustic_utterance code paths (encoder, native AUTO detection, three branches, region
    choice, hard-zero masks, cold masked replay, scoring). Writes only to the given scratch directory."""
    import torch
    from csasr.inference_cf.s1_evidence import BRANCHES
    from csasr.models.whisper import load_whisper
    from csasr.utils.config import load_config
    cfg = json.loads((ROOT / CONFIG).read_text())
    mc = load_config(ROOT / "configs/model/whisper_large_v3.yaml")
    mc["model"]["device"] = "cpu"
    torch.set_num_threads(int(os.environ.get("SMOKE_THREADS", "32")))
    t0 = time.time()
    bundle = load_whisper(mc)
    bundle.model.requires_grad_(False)
    gen = bundle.model.generation_config
    sup, beg = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    tok = bundle.processor.tokenizer
    rng = np.random.default_rng(7)
    n = 16000 * 6
    tt = np.arange(n) / 16000
    x = (0.05 * np.sin(2 * np.pi * 220 * tt) * (tt > 1) + 0.01 * rng.standard_normal(n)).astype(np.float32)
    toks = tok.encode(" hello world 你好", add_special_tokens=False)[:5]
    ivs = np.array([[0, 16000, 0], [16000, 64000, 1], [64000, n, 2]], dtype=np.int64)
    u = {"utterance_id": "SMOKE", "content_prefix_tokens": toks, "max_t": 3}
    qs = []
    for j, t in enumerate((1, 3)):
        qs.append((j, {"utterance_id": "SMOKE", "t": t, "absolute_query": 3 + t, "content_prefix_sha256": digest(toks[:t]),
                       "forced_zh_query_input_sha256": digest(CB + toks[:t]), "forced_en_query_input_sha256": digest(CE + toks[:t])}))
    counters = {"encoder_calls": 0, "decoder_steps": {b: 0 for b in BRANCHES}, "auto_detection_queries": 0, "auto_alias": 0, "masked_encoder_calls": 0,
                "masked_decoder_steps": 0, "mask_groups": 0, "query_readouts": 0, "lac_reused": 0}
    t1 = time.time()
    cres = candidate_utterance(bundle, u=u, jq=qs, x=x, ivs=ivs, cfg=cfg, sup=sup, beg=beg, gauto=auto_generation_config(cfg["model"]["dir"]),
                               counters=counters, hist=None)
    t2 = time.time()
    import copy
    qsm = copy.deepcopy(cres["queries"])
    natural = [(q["region"]["target"]["status"], q["region"]["off_target"]["status"]) for q in qsm]
    # SMOKE ONLY: inject one OK target / off-target pair so the masked encoder, cold replay and scoring paths execute
    qsm[-1]["region"]["target"] = {"status": "OK", "bounds": [16000, 32000]}
    qsm[-1]["region"]["off_target"] = {"status": "OK", "bounds": [64000, 80000]}
    cd = {"utterance_id": "SMOKE", "waveform_sha256": arr_sha(x), "encoder_lineage": cres["encoder_lineage"], "queries": qsm,
          "auto_alias": cres["alias"]}
    lseal = {("SMOKE", q["t"]): {"input_ids": CB + toks[:q["t"]], "query": q["absolute_query"], "W_star": [0, 8000],
                                 "unmasked_logit_vector_sha256": arr_sha(unpack(cres["arrays"][f"q{j:03d}_M_raw"]))} for j, q in qs}
    lrow = {"positions": {str(q["t"]): {"query": q["absolute_query"], "x_sha256": arr_sha(x), "W_star": [0, 8000], "fed_equals_input_ids": True,
                                        "query_matches": True, "logits_finite": True, "outside_left_equal": True, "outside_right_equal": True,
                                        "inside_positive_zero": True} for _, q in qs}}
    lac = {f"t{q['t']}_masked": cres["arrays"][f"q{j:03d}_M_raw"] for j, q in qs}      # stand-in (smoke only)
    members = {f"q{q['j']:03d}": {k: q["unions"][k]["membership_sha256"] for k in q["unions"]} for q in cres["queries"]}
    ares = acoustic_utterance(bundle, u=u, cd=cd, carr=cres["arrays"], x=x, cfg=cfg, sup=sup, beg=beg, counters=counters,
                              lac_ctx={"lseal": lseal, "lrow": lrow, "lac": lac, "seal_ok": True}, members=members, rq=[q for _, q in qs])
    t3 = time.time()
    out = {"schema": SCHEMA + "_cpu_smoke", "synthetic_audio": True, "panel_audio_used": False, "device": str(bundle.device), "dtype": str(bundle.dtype),
           "load_sec": t1 - t0, "candidate_sec": t2 - t1, "acoustic_sec": t3 - t2, "counters": counters, "auto": cres["auto"],
           "ident_fail": cres["ident_fail"], "branch_lineage_ok": cres["branch_lineage_ok"],
           "natural_region_status": natural, "injected_smoke_region": {"t": qsm[-1]["t"], "target": [16000, 32000], "off_target": [64000, 80000]},
           "regions": [{"t": q["t"], "target": q["region"]["target"]["status"], "bounds": q["region"]["target"]["bounds"],
                        "off": q["region"]["off_target"]["status"], "off_bounds": q["region"]["off_target"]["bounds"],
                        "raw_mass": q["region"]["raw_heard_mass"]} for q in cres["queries"]],
           "union_sizes": {k: [len(q["unions"][k]["ids"]) for q in cres["queries"]] for k in ("5", "20")},
           "acoustic_fails": [list(map(str, f)) for f in ares["fails"]], "zero_change": ares["zero_change"],
           "mask_groups": {k: {kk: v[kk] for kk in ("outside_left_equal", "outside_right_equal", "inside_positive_zero", "positions_ok", "changed_samples")}
                           for k, v in ares["mask_groups"].items()},
           "top1": [{k: q["K"][k]["top1"] for k in q["K"]} for q in ares["queries"]],
           "masked_vs_original_changed": [bool(q["K"]["20"]["lm_target"] is not None and q["K"]["20"]["lm_target"] != q["K"]["20"]["l0"]) for q in ares["queries"]],
           "paired": [q["paired_available"] for q in ares["queries"]], "fallback": [q["K"]["20"]["fallback_to_M"] for q in ares["queries"]]}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    atomic_json(Path(args.out) / "cpu_smoke.json", out)
    print(json.dumps(out, indent=1)[:4000])


def cmd_seal(args) -> None:
    run = ROOT / RUN
    rt = json.loads((run / "acoustic_runtime.json").read_text())
    m = json.loads((run / "manifest_acoustic.json").read_text())
    if rt["status"] != "completed" or rt["manifest_hash"] != m["manifest_hash"]:
        raise ValueError("acoustic phase not completed")
    files = {}
    for p in sorted(run.rglob("*")):
        if p.is_file() and not p.name.endswith(".tmp"):
            files[str(p.relative_to(ROOT))] = file_hash(p)
    for p in (PLAN, PRERUN, CAND_AUDIT):
        files[p] = file_hash(ROOT / p)
    doc = {"schema": SCHEMA + "_output_seal", "status": "SEALED", "candidate_seal_hash": json.loads((ROOT / CAND_SEAL).read_text())["seal_hash"],
           "manifest_hashes": {s: json.loads((run / f"manifest_{s}.json").read_text())["manifest_hash"] for s in STAGES},
           "config_sha256": file_hash(ROOT / CONFIG), "sources": m["sources"], "files": files, "references_used": False,
           "git_head_at_seal": git("rev-parse", "HEAD"), "created_unix": time.time()}
    doc["seal_hash"] = digest(doc)
    out = ROOT / OUTPUT_SEAL
    if out.exists():
        raise FileExistsError("output seal exists; never overwrite")
    atomic_json(out, doc)
    print(json.dumps({"seal_hash": doc["seal_hash"], "files": len(files)}))


def cmd_primary(args) -> None:
    """Reference-free primary analysis: completeness, statuses, candidate sizes/types/branches, regions, top-1 movement."""
    run = ROOT / RUN
    seal = json.loads((ROOT / OUTPUT_SEAL).read_text())
    for p, h in seal["files"].items():
        if file_hash(ROOT / p) != h:
            raise ValueError(f"sealed file changed: {p}")
    crt, art = json.loads((run / "candidates_runtime.json").read_text()), json.loads((run / "acoustic_runtime.json").read_text())
    cand = [json.loads((run / "candidates" / f"{i:03d}.json").read_text()) for i in range(80)]
    aco = [json.loads((run / "acoustic" / f"{i:03d}.json").read_text()) for i in range(80)]
    cq = {q["j"]: q for d in cand for q in d["queries"]}
    aq = {q["j"]: q for d in aco for q in d["queries"]}
    status = defaultdict(int)
    ostatus = defaultdict(int)
    sizes = {k: [] for k in ("5", "20")}
    eos_in = {k: 0 for k in ("5", "20")}
    other = {k: 0 for k in ("5", "20")}
    origin_only = {k: defaultdict(int) for k in ("5", "20")}
    moved = {k: defaultdict(int) for k in ("5", "20")}
    durs, ratios, tmass, omass = [], [], [], []
    for j in range(180):
        c, a = cq[j], aq[j]
        status[c["region"]["target"]["status"]] += 1
        ostatus[c["region"]["off_target"]["status"]] += 1
        if c["region"]["target"]["status"] == "OK":
            durs.append(c["region"]["target"]["bounds"][1] - c["region"]["target"]["bounds"][0])
            tmass.append(c["region"]["target"]["integral"])
        if c["region"]["off_target"]["status"] == "OK":
            ratios.append(c["region"]["off_target"]["energy_ratio"])
            omass.append(c["region"]["off_target"]["integral"])
        for k in ("5", "20"):
            un = c["unions"][k]
            sizes[k].append(len(un["ids"]))
            eos_in[k] += int("TERMINATE" in un["types"])
            other[k] += int("OTHER_ACTION" in un["types"])
            for o in un["origin"]:
                origin_only[k][o] += 1
            for s_, top in a["K"][k]["top1"].items():
                moved[k][s_] += int(top != a["K"][k]["top1"]["M_ORIGINAL"])

    def dist(v):
        return None if not v else {"n": len(v), "min": float(np.min(v)), "median": float(np.median(v)), "max": float(np.max(v)), "mean": float(np.mean(v))}
    autos = defaultdict(int)
    for d in cand:
        autos[d["auto"]["detected"]] += 1
    doc = {"schema": SCHEMA + "_primary", "reference_free": True, "output_seal_hash": seal["seal_hash"],
           "completeness": {"candidate_queries": len(cq), "acoustic_queries": len(aq), "utterances": len(cand),
                            "all_candidate_outputs_valid": all(all(q["branches"][b]["raw_finite"] and q["branches"][b]["logp_finite_on_legal"] for b in q["branches"]) for q in cq.values()),
                            "M_bitwise_hist_none_180": sum(q["M_bitwise_hist_none"] for q in cq.values()),
                            "same_prefix_180": sum(q["same_content_prefix"] and q["query_matches"] for q in cq.values())},
           "auto_detected_language_counts": dict(autos), "target_status": dict(status), "off_target_status": dict(ostatus),
           "paired_available": sum(aq[j]["paired_available"] for j in range(180)),
           "union_sizes": {k: dist(v) for k, v in sizes.items()}, "unions_with_EOS": eos_in, "unions_with_other_action": other,
           "origin_bit_counts": {k: dict(v) for k, v in origin_only.items()}, "top1_changed_vs_M": {k: dict(v) for k, v in moved.items()},
           "target_duration_samples": dist(durs), "target_attention_integral": dist(tmass), "off_target_attention_integral": dist(omass),
           "off_target_energy_ratio": dist(ratios),
           "compute": {"candidates": {"elapsed_sec": crt["elapsed_sec"], "counters": crt["counters"], "peak_alloc": crt["peak_vram_allocated_bytes"],
                                      "peak_reserved": crt["peak_vram_reserved_bytes"], "job_id": crt["job_id"], "smoke": crt.get("smoke")},
                       "acoustic": {"elapsed_sec": art["elapsed_sec"], "counters": art["counters"], "peak_alloc": art["peak_vram_allocated_bytes"],
                                    "peak_reserved": art["peak_vram_reserved_bytes"], "job_id": art["job_id"]}},
           "created_unix": time.time()}
    doc["analysis_hash"] = digest(doc)
    if (ROOT / PRIMARY).exists():
        raise FileExistsError("primary analysis exists")
    atomic_json(ROOT / PRIMARY, doc)
    print(json.dumps({k: doc[k] for k in ("completeness", "target_status", "off_target_status", "paired_available")}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    mf = sub.add_parser("manifest")
    mf.add_argument("--stage", required=True, choices=STAGES)
    for c in ("candidates", "candidate-seal", "acoustic", "seal", "primary"):
        sub.add_parser(c)
    sub.add_parser("smoke").add_argument("--out", required=True)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "manifest": cmd_manifest, "candidates": cmd_candidates, "candidate-seal": cmd_candidate_seal,
     "acoustic": cmd_acoustic, "seal": cmd_seal, "primary": cmd_primary, "smoke": cmd_smoke}[args.cmd](args)


if __name__ == "__main__":
    main()
