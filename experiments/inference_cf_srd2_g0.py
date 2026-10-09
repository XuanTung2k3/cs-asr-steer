#!/usr/bin/env python
"""SRD2-G0 runner: acoustic-conflict-gated readout steering feasibility (frozen config
``configs/inference_cf/srd2_g0.json``; spec ``docs/inference_cf/SRD2_G0_SPEC.md``).

Modes (each refuses an absent/failed audit, changed source/config/model pin or a missing remote seal):

* ``prepare``       CPU  safe four-key runtime projection of the frozen 400-utterance selection.
* ``manifest``      CPU  immutable source/config/population/model/environment manifest.
* ``capture``       GPU  Job A: canonical cached forced-ZH B0, every actual query, original R2 full-replay
                         g_old (raw script mass, ten heads, native LID, null), cached/full compatibility and
                         causal-probe evidence. No D2, no steering, no reference.
* ``seal-a``        CPU  Job-A digest seal + durable content-addressed raw archive + reference-free predicates.
* ``permutation``   CPU  frozen within-utterance B3 gate permutation from the sealed gates (after audit A).
* ``authorize-b``   CPU  complete-inventory recost and the minimal eight-field global authorization.
* ``pulses``        GPU  Job B: historical old30 apparatus, then D2 once per structural query and the four
                         pristine single-query arms (B0 none, B1 e*, B2 e*g, B3 e*g_shuffled). No reference.
* ``seal-b``        CPU  Job-B digest seal + durable raw archive.

No reference transcript, target set, stratum, CTC/MMS-FA timing or evaluator quantity is read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, digest, file_hash
from csasr.inference_cf import srd2_g0 as S

SCHEMA = "srd2_g0_run_v1"
CONFIG = "configs/inference_cf/srd2_g0.json"
CONFIG_SHA = "sha256:218b3be997e8fafc102f7ec866b0a82b9925bb180e671f714c651bddee392eb9"
FREEZE = "docs/inference_cf/SRD2_G0_FREEZE.json"
POPULATION = "docs/inference_cf/SRD2_G0_POPULATION.json"
DESIGN_COMMIT = "3168508e5cfa460c370da3b82146678fb30df869"
RUN = "results/inference_cf/srd2_g0/run1"
REMOTE = "origin/cs-asr-steer-inf"
ARCHIVE = Path("/mnt/data/tungnx/cs-asr-steer/archives/srd2_g0/run1")
OWN = ("src/csasr/inference_cf/srd2_g0.py", "experiments/inference_cf_srd2_g0.py", "slurm/inference_cf_srd2_g0.sbatch")
NOT_EXECUTED = ("experiments/inference_cf_srd2_g0_evaluate.py", "experiments/inference_cf_srd2_g0_audit.py",
                "tests/test_inference_cf_srd2_g0.py")
REUSED = ("src/csasr/__init__.py", "src/csasr/inference_cf/__init__.py", "src/csasr/inference_cf/unique.py",
          "src/csasr/utils/__init__.py", "src/csasr/utils/config.py", "src/csasr/utils/hashing.py",
          "src/csasr/utils/logging.py", "src/csasr/data/__init__.py", "src/csasr/data/alignment.py",
          "src/csasr/models/__init__.py", "src/csasr/lss/__init__.py", ".gitignore")
APPARATUS = {"p2dir_sealed": "results/inference_cf/p2dir/exp1_run1/directions_sealed.json",
             "p2dir_rows": "results/inference_cf/p2dir/exp1_run1/rows",
             "baseline_panel": "results/inference_cf/p2_A_r1_L16/panel.json",
             "baseline_rows": "results/inference_cf/p2_A_r1_L16/rows",
             "audio_panel": "results/inference_cf/p0_r2/inference_panel.json"}
PANEL = f"{RUN}/runtime_panel.json"
MANIFEST = f"{RUN}/manifest.json"
PRE = f"{RUN}/audit_PRE.json"
GATE_SEAL = f"{RUN}/gate_seal.json"
AUDIT_A = f"{RUN}/audit_A.json"
PERM = f"{RUN}/permutation.json"
PERM_SEAL = f"{RUN}/permutation_seal.json"
FORECAST = f"{RUN}/resource_forecast.json"
AUTH = f"{RUN}/authorization_B.json"
PULSE_SEAL = f"{RUN}/pulse_seal.json"
PROVIDER_FAILURES = ("localizer_fail", "local_support_fail", "baseline_provider_fail", "nonfinite_signal")


# ---- git / provenance ----------------------------------------------------------------------------------

def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def blob_sha(commit: str, rel: str) -> str:
    out = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=ROOT, capture_output=True)
    return "sha256:" + hashlib.sha256(out.stdout).hexdigest() if out.returncode == 0 else ""


def committed(rel: str) -> bool:
    return git("ls-files", rel) == rel and blob_sha("HEAD", rel) == file_hash(ROOT / rel)


def on_remote(rel: str) -> bool:
    c = git("log", "-n1", "--format=%H", "--", rel)
    return bool(c) and subprocess.run(["git", "merge-base", "--is-ancestor", c, REMOTE], cwd=ROOT).returncode == 0


def load_cfg() -> dict:
    if file_hash(ROOT / CONFIG) != CONFIG_SHA:
        raise ValueError("frozen config bytes changed")
    return json.loads((ROOT / CONFIG).read_text())


def verify_freeze(cfg: dict) -> None:
    fr = json.loads((ROOT / FREEZE).read_text())
    for rel, h in fr["files"].items():
        if file_hash(ROOT / rel) != h:
            raise ValueError(f"frozen design file changed: {rel}")
    pop = cfg["population"]
    if file_hash(ROOT / pop["manifest"]) != pop["file_sha256"]:
        raise ValueError("population file hash")
    P = json.loads((ROOT / pop["manifest"]).read_text())
    if digest({k: v for k, v in P.items() if k != "manifest_hash"}) != P["manifest_hash"] or \
            P["manifest_hash"] != pop["manifest_hash"]:
        raise ValueError("population manifest hash")
    if digest(P["selected"]) != pop["selected_hash"] or \
            digest([r["utterance_id"] for r in P["selected"]]) != pop["selected_ids_hash"]:
        raise ValueError("population selection hash")


def pinned_sources(cfg: dict) -> list[str]:
    return sorted(set(cfg["source_sha256"]) | set(OWN) | set(NOT_EXECUTED) | set(REUSED) | {CONFIG, FREEZE, PANEL})


def environment() -> dict:
    import platform
    import torch
    import transformers
    return {"python": platform.python_version(), "interpreter": sys.executable, "torch": torch.__version__,
            "transformers": transformers.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
            "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}


def run_dir() -> Path:
    return ROOT / RUN


# ---- prepare / manifest -----------------------------------------------------------------------------------

def cmd_prepare(args) -> None:
    cfg = load_cfg()
    verify_freeze(cfg)
    out = ROOT / PANEL
    if out.exists():
        raise FileExistsError("runtime panel exists; never overwrite")
    P = json.loads((ROOT / cfg["population"]["manifest"]).read_text())
    panel = S.runtime_projection(P)
    if len(panel["rows"]) != cfg["population"]["utterances"]:
        raise ValueError("selected count")
    atomic_json(out, panel)
    print(json.dumps({"rows": len(panel["rows"]), "runtime_hash": panel["runtime_hash"]}))


def cmd_manifest(args) -> None:
    import inspect
    from transformers import WhisperTokenizer
    from transformers.models.whisper.modeling_whisper import WhisperDecoderLayer
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.utils.provenance import code_config_snapshot_hash, test_snapshot_hash
    cfg = load_cfg()
    verify_freeze(cfg)
    out = ROOT / MANIFEST
    if out.exists():
        raise FileExistsError("manifest exists; never overwrite")
    srcs = pinned_sources(cfg)
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *srcs)
    if dirty:
        raise ValueError("commit sources before the manifest:\n" + dirty)
    if not all(committed(s) and on_remote(s) for s in srcs):
        raise ValueError("every pinned source must be committed and pushed")
    for rel, h in cfg["source_sha256"].items():
        if file_hash(ROOT / rel) != h:
            raise ValueError(f"historical pinned source changed: {rel}")
    mdir = Path(cfg["model"]["dir"])
    model_files = {f: file_hash(mdir / f) for f in cfg["model"]["files"]}
    if model_files != cfg["model"]["files"]:
        raise ValueError("model/tokenizer/preprocessor file hashes differ from the freeze")
    if file_hash(cfg["environment"]["transformers_forward_source"]) != cfg["environment"]["transformers_forward_sha256"]:
        raise ValueError("installed Transformers forward source changed")
    tok = WhisperTokenizer.from_pretrained(str(mdir), local_files_only=True)
    part = tokenizer_partition(tok)
    counts = {"matrix": len(part["matrix_ids"]), "embedded": len(part["embedded_ids"]), "ambiguous": len(part["ambiguous_ids"])}
    if part["hash"] != cfg["gate"]["partition_hash"] or counts != cfg["gate"]["partition_counts"]:
        raise ValueError("tokenizer partition differs from the freeze")
    gen = json.loads((mdir / "generation_config.json").read_text())
    supp = digest({"suppress": list(gen.get("suppress_tokens") or []), "begin": list(gen.get("begin_suppress_tokens") or [])})
    if supp != cfg["model"]["suppression_hash"] or gen.get("alignment_heads") != cfg["gate"]["alignment_heads"]:
        raise ValueError("suppression lists / alignment heads differ from the freeze")
    panel = json.loads((ROOT / PANEL).read_text())
    S.validate_runtime_panel(panel)
    env = environment()
    if env["torch"] != cfg["environment"]["torch"] or env["transformers"] != cfg["environment"]["transformers"]:
        raise ValueError("library versions differ from the freeze")
    man = {"schema": SCHEMA, "design_freeze_commit": DESIGN_COMMIT, "git_commit": git("rev-parse", "HEAD"),
           "git_tree": git("rev-parse", "HEAD^{tree}"), "config": CONFIG, "config_sha256": CONFIG_SHA,
           "config_hash": digest(cfg), "freeze_index_hash": json.loads((ROOT / FREEZE).read_text())["freeze_index_hash"],
           "population": {k: cfg["population"][k] for k in ("file_sha256", "manifest_hash", "selected_hash",
                                                            "selected_ids_hash", "roster_hash")},
           "runtime_panel": PANEL, "runtime_hash": panel["runtime_hash"], "environment": env,
           "model": {"dir": str(mdir), "files": model_files, "precision": "bfloat16", "attention": "eager"},
           "transformers_forward_sha256": cfg["environment"]["transformers_forward_sha256"],
           "decoder_layer_forward_source_sha256": "sha256:" + hashlib.sha256(inspect.getsource(WhisperDecoderLayer.forward).encode()).hexdigest(),
           "partition_hash": part["hash"], "partition_counts": counts, "suppression_hash": supp,
           "alignment_heads": cfg["gate"]["alignment_heads"], "prompt": list(S.CB),
           "sources": {s: file_hash(ROOT / s) for s in srcs},
           "apparatus_inputs": {"p2dir_sealed": file_hash(ROOT / APPARATUS["p2dir_sealed"]),
                                "baseline_panel": file_hash(ROOT / APPARATUS["baseline_panel"]),
                                "audio_panel": file_hash(ROOT / APPARATUS["audio_panel"])},
           "archive_root": str(ARCHIVE),
           "provenance": {"code_config_snapshot_hash": code_config_snapshot_hash(ROOT), "test_snapshot_hash": test_snapshot_hash(ROOT)},
           "reference_access_boundary": "runner phases read the four-key runtime panel, audio, model and (Job B only) the "
                                        "hash-pinned old30 apparatus artifacts; references only in the separate evaluator "
                                        "after the pushed pulse seal and SRD2_G0_AUDIT_PRIMARY: PASS",
           "trainable_parameters": 0, "optimizer": None, "references_used": False, "status": "FROZEN",
           "created_unix": time.time()}
    man["manifest_hash"] = digest(man)
    atomic_json(out, man)
    print(json.dumps({"manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"], "sources": len(man["sources"])}))


# ---- GPU-phase guards -----------------------------------------------------------------------------------

OPENED: set = set()


def _audit_open(event, args) -> None:
    """Runtime file-open log (firewall evidence): every path opened outside the Python installation."""
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.path.abspath(os.fsdecode(args[0]))
        if not p.startswith((sys.prefix, sys.base_prefix, "/proc", "/sys", "/dev", "/usr", "/lib", "/etc",
                             "/tmp", "/run", "/var")) and "/.git/" not in p and "__pycache__" not in p:
            OPENED.add(p)


def load_run(stage: str) -> tuple[dict, dict, list[dict]]:
    m = json.loads((ROOT / MANIFEST).read_text())
    if m["schema"] != SCHEMA or digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError("invalid manifest")
    for p, h in m["sources"].items():
        if file_hash(ROOT / p) != h:
            raise ValueError(f"source changed after manifest: {p}")
    head = git("rev-parse", "HEAD")
    if head != m["git_commit"]:
        if subprocess.run(["git", "merge-base", "--is-ancestor", m["git_commit"], head], cwd=ROOT).returncode != 0:
            raise ValueError("HEAD does not descend from the manifest commit")
        changed = git("diff", "--name-only", m["git_commit"], head).splitlines()
        if any(not c.startswith(RUN + "/") for c in changed):
            raise ValueError("HEAD differs from the manifest commit outside the run directory: "
                             + ",".join(c for c in changed if not c.startswith(RUN + "/")))
    cfg = load_cfg()
    if digest(cfg) != m["config_hash"]:
        raise ValueError("config hash")
    panel = json.loads((ROOT / PANEL).read_text())
    rows = S.validate_runtime_panel(panel)
    if panel["runtime_hash"] != m["runtime_hash"]:
        raise ValueError("runtime panel hash")
    if not (committed(MANIFEST) and on_remote(MANIFEST)):
        raise SystemExit("the manifest must be committed and pushed")
    pre = json.loads((ROOT / PRE).read_text())
    if not (committed(PRE) and on_remote(PRE) and pre.get("verdict") == "PASS_TO_SRD2_G0"
            and pre.get("manifest_hash") == m["manifest_hash"]):
        raise SystemExit("independent PASS_TO_SRD2_G0 (committed + pushed, same manifest) required")
    if stage == "pulses":
        need = (GATE_SEAL, AUDIT_A, PERM, PERM_SEAL, FORECAST, AUTH)
        if not all(committed(x) and on_remote(x) for x in need):
            raise SystemExit("Job B requires pushed gate seal, audit A, permutation seal, forecast and authorization")
        auth = json.loads((ROOT / AUTH).read_text())
        if list(auth) != cfg["firewall"]["authorization_B_fields"] or auth["authorized"] is not True:
            raise SystemExit("authorization B absent, malformed or not authorized")
        expect = {"job_A_seal_hash": file_hash(ROOT / GATE_SEAL), "config_hash": CONFIG_SHA,
                  "population_hash": cfg["population"]["selected_ids_hash"], "permutation_hash": file_hash(ROOT / PERM_SEAL),
                  "audit_hash": file_hash(ROOT / AUDIT_A), "resource_forecast_hash": file_hash(ROOT / FORECAST)}
        if any(auth[k] != v for k, v in expect.items()):
            raise SystemExit("authorization B hashes do not match the sealed artifacts")
        if json.loads((ROOT / AUDIT_A).read_text()).get("verdict") != "SRD2_G0_AUDIT_A: PASS":
            raise SystemExit("SRD2_G0_AUDIT_A: PASS required")
    return m, cfg, rows


class Ctx:
    def __init__(self, bundle, cfg: dict):
        from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
        from csasr.inference_cf.core_r2 import tokenizer_partition
        from csasr.lss.sites import num_forced_prefix_from
        self.bundle, self.cfg = bundle, cfg
        self.tok = bundle.processor.tokenizer
        self.eos = int(self.tok.eos_token_id)
        gen = bundle.model.generation_config
        self.suppress, self.begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
        if digest({"suppress": self.suppress, "begin": self.begin}) != cfg["model"]["suppression_hash"]:
            raise ValueError("suppression lists changed")
        if [list(x) for x in gen.alignment_heads] != cfg["gate"]["alignment_heads"]:
            raise ValueError("alignment heads changed")
        self.partition = tokenizer_partition(self.tok)
        if self.partition["hash"] != cfg["gate"]["partition_hash"]:
            raise ValueError("tokenizer partition changed")
        self.nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
        if self.nfp != 4 or self.eos != S.EOS:
            raise ValueError("forced prefix / EOS mismatch")
        native = gen.lang_to_id
        self.language_ids = tuple(sorted(set(int(x) for x in native.values())))
        if len(self.language_ids) != 100:
            raise ValueError("unexpected language-token vocabulary")
        self.en_id, self.zh_id = int(native["<|en|>"]), int(native["<|zh|>"])
        self.byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
        self.e_star = float(cfg["dose"]["e_star"])
        self.prompt = list(S.CB)
        if self.prompt != cfg["baseline"]["prompt"]:
            raise ValueError("forced-ZH prompt differs from the freeze")
        self.max_new = int(cfg["baseline"]["max_new_tokens"])
        self.layer = int(cfg["model"]["layer_zero_based"])


def load_model(cfg: dict):
    import torch
    from csasr.models.whisper import load_whisper
    from csasr.utils.config import load_config
    torch.manual_seed(S.SEED)
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    if getattr(bundle.model.config, "_attn_implementation", "eager") != "eager":
        raise ValueError("attention implementation must be eager")
    bundle.model.eval()
    bundle.model.requires_grad_(False)
    torch.cuda.synchronize()
    return bundle, time.perf_counter() - t0


def weights_digest(bundle) -> str:
    import torch
    h = hashlib.sha256()
    for k, v in bundle.model.state_dict().items():
        h.update(k.encode())
        t = v.detach().contiguous().cpu()
        h.update((t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes())
    return "sha256:" + h.hexdigest()


def sync() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def encode(bundle, audio_path: str):
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.models.whisper import batch_model_inputs
    t0 = time.perf_counter()
    inputs = batch_model_inputs(bundle, [audio_path])
    sync()
    t1 = time.perf_counter()
    with torch.inference_mode():
        h = bundle.model.model.encoder(input_features=inputs["input_features"],
                                       attention_mask=inputs["attention_mask"], return_dict=True).last_hidden_state
    sync()
    t2 = time.perf_counter()
    return BaseModelOutput(last_hidden_state=h), {"features_sec": t1 - t0, "encoder_sec": t2 - t1,
                                                   "encoder_sha256": S.tensor_sha(h)}


def write_npz(path: Path, arrays: dict) -> tuple[str, int, float]:
    """Uncompressed lossless archive, fsynced; returns (sha256, bytes, write seconds)."""
    tmp = path.with_name(path.name + ".tmp.npz")
    t0 = time.perf_counter()
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    dt = time.perf_counter() - t0
    return file_hash(path), path.stat().st_size, dt


# ---- Job A: capture -----------------------------------------------------------------------------------

def cached_baseline(ctx: Ctx, enc, *, teacher: list[int] | None = None) -> dict:
    """Canonical pristine cached B0 (explicit lowest-ID processed argmax). With ``teacher`` the same cached
    path is replayed on fixed content (bitwise identity evidence); otherwise greedy generation."""
    import experiments.inference_cf_p2r as p2r
    from csasr.inference_cf.core_p1 import processed_argmax
    B = p2r.DiagBranch(ctx.bundle, enc, list(ctx.prompt), "B")
    new, content, terminated = list(ctx.prompt), [], "cap"
    logits, sites, qsec, argmax = [], [], [], []
    for t in range(ctx.max_new):
        site = S.NativeSite(ctx.bundle, ctx.layer)
        sync()
        t0 = time.perf_counter()
        lg, _, _ = B.step(new, capture_layer=None, attention=True, hook=site)
        sync()
        qsec.append(time.perf_counter() - t0)
        S.assert_no_hooks_anywhere(ctx.bundle)
        logits.append(lg)
        sites.append(site.r[0].float().cpu())
        nxt = processed_argmax(lg, t, ctx.suppress, ctx.begin)
        argmax.append(int(nxt))
        if teacher is not None:
            if t == len(teacher):
                terminated = "eos" if nxt == ctx.eos else "replay_mismatch"
                break
            nxt_fed = int(teacher[t])
        else:
            if nxt == ctx.eos:
                terminated = "eos"
                break
            nxt_fed = int(nxt)
        content.append(nxt_fed)
        new = [nxt_fed]
    P = len(ctx.prompt)
    lineage = B.fed[:P] == list(ctx.prompt) and B.fed[P:] == content[:len(B.fed) - P] and B.positions == list(range(len(B.fed)))
    return {"content": content, "terminated": terminated, "logits": logits, "sites": sites, "query_sec": qsec,
            "argmax": argmax, "lineage_ok": bool(lineage)}


def capture_utterance(ctx: Ctx, row: dict, null_probs: dict, primitives: dict, out_dir: Path, m: dict) -> dict:
    import torch
    import experiments.inference_cf_p0_r2 as r2
    from csasr.inference_cf.core_p1 import processed_argmax
    from csasr.inference_cf.core_r2 import conflict_from_logits, prefix_utf8_complete
    from csasr.lss.sites import DecoderPostCrossAttnRecorder
    from csasr.models.whisper import load_audio
    uid = row["utterance_id"]
    t_io = time.perf_counter()
    if file_hash(row["audio_path"]) != row["audio_full_sha256"]:
        raise ValueError("audio bytes changed after the population freeze")
    waveform = load_audio(row["audio_path"], ctx.bundle.sample_rate)
    io_sec = time.perf_counter() - t_io
    OPENED.add(os.path.abspath(row["audio_path"]))
    enc, enc_t = encode(ctx.bundle, row["audio_path"])
    base = cached_baseline(ctx, enc)
    content, terminated = base["content"], base["terminated"]
    inv = S.inventory(content, terminated)
    if len(base["logits"]) != len(inv):
        raise RuntimeError("inventory/cached query count mismatch")
    replay = cached_baseline(ctx, enc, teacher=content)
    replay_bitwise = (replay["terminated"] == terminated and replay["content"] == content and len(replay["logits"]) == len(inv)
                      and all(torch.equal(a, b) for a, b in zip(replay["logits"], base["logits"]))
                      and all(torch.equal(a, b) for a, b in zip(replay["sites"], base["sites"])))
    # ---- original R2 full causal replay (no cache) on the exact cached-B0 content -------------------------
    fm = S.FutureMassProbe(ctx.bundle)
    rec = DecoderPostCrossAttnRecorder(ctx.bundle, [ctx.layer], to_dtype=torch.bfloat16)
    sync()
    t0 = time.perf_counter()
    with fm, rec:
        full_logits, heads = r2.full_replay(ctx.bundle, enc, list(ctx.prompt), content, attention=True)
    sync()
    full_sec = time.perf_counter() - t0
    S.assert_no_hooks_anywhere(ctx.bundle)
    full_site = rec.states[ctx.layer][0].float().cpu()
    if full_logits.shape[0] != len(ctx.prompt) + len(content) or heads.shape[1] != full_logits.shape[0]:
        raise RuntimeError("full replay shape")
    lid_cache: dict = {}
    lid_order: list = []
    lid_counts = {"calls": 0, "cache_hits": 0, "sec": 0.0}

    def lid(start: int, end: int) -> dict:
        key = (int(start), int(end))
        if key in lid_cache:
            lid_counts["cache_hits"] += 1
            return lid_cache[key]
        sync()
        a = time.perf_counter()
        lid_cache[key] = r2.native_lid(ctx.bundle, waveform[key[0]:key[1]], ctx.language_ids)
        sync()
        lid_counts["sec"] += time.perf_counter() - a
        lid_counts["calls"] += 1
        lid_order.append(key)
        return lid_cache[key]

    queries, structural_ts = [], []
    for i, t in enumerate(inv):
        q = S.query_index(t, len(ctx.prompt))
        prefix = content[:t]
        lc = base["logits"][i]
        utf8 = prefix_utf8_complete(ctx.tok, prefix, ctx.byte_decoder)
        finite_site, finite_logits = bool(torch.isfinite(base["sites"][i]).all()), bool(torch.isfinite(lc).all())
        ok, why = S.structural(t, prefix, utf8, finite_site, finite_logits, ctx.eos)
        gate = S.r2_gate(utf8_complete=utf8, heads_q=heads[:, q].numpy(), heard_samples=len(waveform),
                         raw_logits=full_logits[q], partition=ctx.partition, lid=lid, null_probs=null_probs,
                         en_id=ctx.en_id, zh_id=ctx.zh_id, primitives=primitives)
        qr = {"t": t, "query": q, "prefix_sha256": S.prefix_hash(ctx.prompt, prefix),
              "expected_action": S.expected_action(content, t, ctx.eos), "is_eos_query": t == len(content),
              "remaining_content_tokens": len(content) - t, "utf8_complete": bool(utf8),
              "structural": {"eligible": ok, "reasons": why},
              "cached_argmax": base["argmax"][i], "gate": gate, "g": float(gate["g"]), "g_bits": S.float_bits(gate["g"])}
        if base["argmax"][i] != qr["expected_action"]:
            raise RuntimeError("cached baseline argmax differs from its own trajectory")
        if ok:
            structural_ts.append(t)
            lf = full_logits[q]
            cf, cc = conflict_from_logits(lf, ctx.partition), conflict_from_logits(lc, ctx.partition)
            geo = S.site_geometry(full_site[q].numpy(), base["sites"][i].numpy())
            qr["compat"] = {"TV": S.total_variation(lf.numpy(), lc.numpy()),
                            "PE_abs_diff": abs(cf["P_E"] - cc["P_E"]), "PM_abs_diff": abs(cf["P_M"] - cc["P_M"]),
                            "full_processed_argmax": processed_argmax(lf, t, ctx.suppress, ctx.begin),
                            "argmax_agree": processed_argmax(lf, t, ctx.suppress, ctx.begin) == base["argmax"][i],
                            "site_rel_L2": geo["rel_L2"], "site_cos": geo["cos"]}
        queries.append(qr)
    # ---- fixed causal probes: prefix-only vs full replay (first / median / last structural query) ------------
    probes, probe_logits, probe_heads = [], [], []
    if structural_ts:
        picks = sorted({structural_ts[0], structural_ts[(len(structural_ts) - 1) // 2], structural_ts[-1]})
        for t in picks:
            q = S.query_index(t, len(ctx.prompt))
            pf = S.FutureMassProbe(ctx.bundle)
            with pf:
                pl, ph = r2.full_replay(ctx.bundle, enc, list(ctx.prompt), content[:t], attention=True)
            S.assert_no_hooks_anywhere(ctx.bundle)
            a_full = S.heard_attention(heads[:, q].numpy(), len(waveform))
            a_pre = S.heard_attention(ph[:, -1].numpy(), len(waveform))
            probes.append({"t": t, "query": q, "TV": S.total_variation(pl[-1].numpy(), full_logits[q].numpy()),
                           "attention_L1": float(np.abs(a_full - a_pre).sum()), "future_mass_max": pf.max_mass,
                           "prefix_len": int(pl.shape[0])})
            probe_logits.append(S.bf16_bits(pl[-1]))
            probe_heads.append(S.bf16_bits(ph[:, -1]))
    qidx = [S.query_index(t, len(ctx.prompt)) for t in inv]
    arrays = {"cached_logits": np.stack([S.bf16_bits(x) for x in base["logits"]]),
              "full_logits": S.bf16_bits(full_logits[qidx]),
              "heads": S.bf16_bits(heads[:, qidx].permute(1, 0, 2).contiguous()),
              "cached_site": np.stack([S.bf16_bits(x) for x in base["sites"]]),
              "full_site": S.bf16_bits(full_site[qidx]),
              "lid_keys": np.asarray(lid_order, dtype=np.int64).reshape(-1, 2),
              "lid_probs": np.asarray([[lid_cache[k][i] for i in ctx.language_ids] for k in lid_order],
                                      dtype=np.float64).reshape(-1, len(ctx.language_ids)),
              "probe_logits": np.stack(probe_logits) if probe_logits else np.zeros((0, full_logits.shape[1]), np.int16),
              "probe_heads": np.stack(probe_heads) if probe_heads else np.zeros((0, heads.shape[0], heads.shape[2]), np.int16)}
    npz_sha, nbytes, wsec = write_npz(out_dir / f"{uid}.npz", arrays)
    return {"schema": SCHEMA, "phase": "A", "identity": uid, "utterance_id": uid, "canonical_index": row["canonical_index"],
            "manifest_hash": m["manifest_hash"], "status": "ok",
            "audio": {"full_sha256": row["audio_full_sha256"], "samples": int(len(waveform)), "io_sec": io_sec},
            "encoder": enc_t, "baseline": {"content_ids": content, "terminated": terminated, "T": len(content),
                                           "lineage_ok": base["lineage_ok"], "replay_bitwise": bool(replay_bitwise)},
            "inventory": inv, "structural_ts": structural_ts, "queries": queries,
            "full_replay": {"sec": full_sec, "future_mass_max": fm.max_mass, "self_attn_calls": fm.calls,
                            "input_ids_sha256": S.prefix_hash(ctx.prompt, content), "shared_encoder": True},
            "probes": probes, "lid": {**lid_counts, "language_ids_sha256": digest(list(ctx.language_ids)),
                                      "language_order": list(ctx.language_ids)},
            "timing": {"cached_query_sec": base["query_sec"], "replay_query_sec": replay["query_sec"]},
            "arrays": {"file": f"{uid}.npz", "sha256": npz_sha, "bytes": nbytes, "write_sec": wsec,
                       "keys": {k: [str(v.dtype), list(v.shape)] for k, v in arrays.items()}}}


def cmd_capture(args) -> None:
    import torch
    import experiments.inference_cf_p0_r2 as r2
    from csasr.inference_cf.core_r2 import conflict_from_logits, gate_values, local_support, max_attention_window
    sys.addaudithook(_audit_open)
    m, cfg, rows = load_run("capture")
    out = run_dir() / "gate"
    if out.exists():
        raise FileExistsError("gate directory exists; a capture attempt is never overwritten")
    out.mkdir(parents=True)
    bundle, load_sec = load_model(cfg)
    ctx = Ctx(bundle, cfg)
    w0 = weights_digest(bundle)
    torch.cuda.reset_peak_memory_stats()
    rt = {"schema": SCHEMA, "phase": "A", "job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
          "gpu": torch.cuda.get_device_name(0), "manifest_hash": m["manifest_hash"], "start_unix": time.time(),
          "status": "running", "model_load_sec": load_sec, "weights_digest_start": w0}
    atomic_json(run_dir() / "capture_runtime.json", rt)
    null = r2.native_lid(bundle, np.zeros(30 * bundle.sample_rate, dtype=np.float32), ctx.language_ids)
    rt["null"] = {"EN": null[ctx.en_id], "ZH": null[ctx.zh_id],
                  "EN_abs_error": abs(null[ctx.en_id] - cfg["gate"]["null_EN_historical"]),
                  "ZH_abs_error": abs(null[ctx.zh_id] - cfg["gate"]["null_ZH_historical"]),
                  "probs_sha256": digest([null[i] for i in ctx.language_ids])}
    primitives = {"max_attention_window": max_attention_window, "conflict_from_logits": conflict_from_logits,
                  "local_support": local_support, "gate_values": gate_values, "reason": r2._reason}
    failures = 0
    for row in rows:
        t0 = time.time()
        try:
            res = capture_utterance(ctx, row, null, primitives, out, m)
        except Exception as exc:
            import traceback
            failures += 1
            res = {"schema": SCHEMA, "phase": "A", "identity": row["utterance_id"], "utterance_id": row["utterance_id"],
                   "canonical_index": row["canonical_index"], "manifest_hash": m["manifest_hash"], "status": "failure",
                   "reason": repr(exc), "traceback": traceback.format_exc()}
        res["elapsed_sec"] = time.time() - t0
        atomic_json(out / f"{row['utterance_id']}.json", res)
        print(f"A {row['canonical_index'] + 1}/{len(rows)} {row['utterance_id']} {res['status']} "
              f"q={len(res.get('inventory', []))} {res['elapsed_sec']:.1f}s", flush=True)
    S.assert_no_hooks_anywhere(bundle)
    rt.update(weights_digest_end=weights_digest(bundle), end_unix=time.time(), failures=failures,
              status="completed" if failures == 0 else "failed",
              peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
              peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(),
              parameters_with_grad=sum(p.grad is not None for p in bundle.model.parameters()),
              parameters_requiring_grad=sum(p.requires_grad for p in bundle.model.parameters()),
              opened_paths=sorted(OPENED))
    rt["elapsed_sec"] = rt["end_unix"] - rt["start_unix"]
    rt["weights_unchanged"] = rt["weights_digest_end"] == w0
    atomic_json(run_dir() / "capture_runtime.json", rt)
    if failures:
        raise SystemExit(f"{failures} utterance failures")


# ---- seal / archive helpers --------------------------------------------------------------------------------

def archive_copy(src: Path, sha: str) -> str:
    """Content-addressed immutable copy on durable project storage; returns the archive path."""
    dest = ARCHIVE / "sha256" / (sha.split(":", 1)[1] + src.suffix)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_name(dest.name + ".tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(dest)
        os.chmod(dest, 0o444)
    if file_hash(dest) != sha:
        raise ValueError(f"archive copy hash mismatch for {src}")
    return str(dest)


def dialogue_of() -> dict:
    """Post-processing cluster metadata only (never a runner/gate input)."""
    P = json.loads((ROOT / POPULATION).read_text())
    return {r["utterance_id"]: r["dialogue_id"] for r in P["selected"]}


def q(xs: list, p: float) -> float | None:
    return float(np.quantile(np.asarray(xs, dtype=np.float64), p)) if xs else None


def job_a_metrics(cfg: dict, rows: list[dict], gate_rows: dict, runtime: dict) -> dict:
    dlg = dialogue_of()
    ok = {u: r for u, r in gate_rows.items() if r.get("status") == "ok"}
    dialogues = sorted(set(dlg.values()))
    complete_d = [d for d in dialogues if all(u in ok for u, dd in dlg.items() if dd == d)]
    inv_ok = sum(1 for r in ok.values() if [x["t"] for x in r["queries"]] == r["inventory"]
                 == S.inventory(r["baseline"]["content_ids"], r["baseline"]["terminated"]))
    struct = [(u, x) for u, r in ok.items() for x in r["queries"] if x["structural"]["eligible"]]
    nonzero = [(u, x) for u, x in struct if x["g"] > 0]
    complete_prefix = [x for r in ok.values() for x in r["queries"] if x["utf8_complete"]]
    finite_provider = [x for x in complete_prefix if x["gate"]["fallback_reason"] not in PROVIDER_FAILURES]
    comp = [x["compat"] for _, x in struct]
    probes = [p for r in ok.values() for p in r["probes"]]
    tv = [c["TV"] for c in comp]
    rel = [c["site_rel_L2"] for c in comp if c["site_rel_L2"] is not None]
    cosv = [c["site_cos"] for c in comp if c["site_cos"] is not None]
    metrics = {
        "rows_complete": len(ok), "dialogues_complete": len(complete_d),
        "inventory_complete_fraction": inv_ok / len(rows), "structural_queries": len(struct),
        "structural_dialogues": len({dlg[u] for u, _ in struct}), "nonzero_gates": len(nonzero),
        "nonzero_gate_dialogues": len({dlg[u] for u, _ in nonzero}),
        "finite_provider_fraction": len(finite_provider) / len(complete_prefix) if complete_prefix else None,
        "identity_fraction": (sum(1 for r in ok.values() if r["full_replay"]["input_ids_sha256"] == S.prefix_hash(S.CB, r["baseline"]["content_ids"])
                                  and r["baseline"]["lineage_ok"] and r["full_replay"]["shared_encoder"]
                                  and [x["query"] for x in r["queries"]] == [S.query_index(t) for t in r["inventory"]]) / len(rows)),
        "cached_baseline_identity_bitwise": bool(ok) and all(r["baseline"]["replay_bitwise"] and r["baseline"]["lineage_ok"] for r in ok.values()),
        "raw_TV_median": q(tv, .5), "raw_TV_p99": q(tv, .99), "raw_TV_max": max(tv) if tv else None,
        "PE_abs_diff_p99": q([c["PE_abs_diff"] for c in comp], .99), "PM_abs_diff_p99": q([c["PM_abs_diff"] for c in comp], .99),
        "processed_argmax_agreement": (sum(c["argmax_agree"] for c in comp) / len(comp)) if comp else None,
        "site_rel_L2_p99": q(rel, .99) if len(rel) == len(comp) else None,
        "site_rel_L2_max": max(rel) if rel and len(rel) == len(comp) else None,
        "site_cosine_min": min(cosv) if cosv and len(cosv) == len(comp) else None,
        "causal_probe_TV_max": max(p["TV"] for p in probes) if probes else None,
        "causal_probe_attention_L1_max": max(p["attention_L1"] for p in probes) if probes else None,
        "future_self_attention_mass_max": max([r["full_replay"]["future_mass_max"] for r in ok.values()]
                                              + [p["future_mass_max"] for p in probes]) if ok else None,
        "null_EN_abs_error": runtime.get("null", {}).get("EN_abs_error"),
        "null_ZH_abs_error": runtime.get("null", {}).get("ZH_abs_error")}
    return metrics


def predicate(rule: dict, metrics: dict) -> bool:
    import operator
    ops = {"==": operator.eq, "<=": operator.le, ">=": operator.ge, ">": operator.gt}
    if "all" in rule:
        return all(predicate(x, metrics) for x in rule["all"])
    if "any" in rule:
        return any(predicate(x, metrics) for x in rule["any"])
    v = metrics.get(rule["metric"])
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return False
    return bool(ops[rule["op"]](v, rule["value"]))


def cmd_seal_a(args) -> None:
    cfg = load_cfg()
    m = json.loads((ROOT / MANIFEST).read_text())
    rows = S.validate_runtime_panel(json.loads((ROOT / PANEL).read_text()))
    out = ROOT / GATE_SEAL
    if out.exists():
        raise FileExistsError("gate seal exists")
    gdir = run_dir() / "gate"
    runtime = json.loads((run_dir() / "capture_runtime.json").read_text())
    gate_rows, files, archive = {}, {}, {}
    for r in rows:
        uid = r["utterance_id"]
        p = gdir / f"{uid}.json"
        if not p.exists():
            continue
        g = json.loads(p.read_text())
        if g["identity"] != uid or g["manifest_hash"] != m["manifest_hash"]:
            raise ValueError(f"gate row identity {uid}")
        gate_rows[uid] = g
        files[f"gate/{uid}.json"] = file_hash(p)
        if g["status"] == "ok":
            npz = gdir / g["arrays"]["file"]
            h = file_hash(npz)
            if h != g["arrays"]["sha256"]:
                raise ValueError(f"gate arrays changed {uid}")
            files[f"gate/{uid}.npz"] = h
            archive[f"gate/{uid}.npz"] = archive_copy(npz, h)
    files["capture_runtime.json"] = file_hash(run_dir() / "capture_runtime.json")
    for extra in sorted(run_dir().glob("slurm-A-*")):
        files[extra.name] = file_hash(extra)
    metrics = job_a_metrics(cfg, rows, gate_rows, runtime)
    preds = {"job_A_coverage": predicate(cfg["gate_predicates"]["job_A_coverage"], metrics),
             "compatibility": predicate(cfg["gate_predicates"]["compatibility"], metrics)}
    critical = {"runtime_completed": runtime.get("status") == "completed", "weights_unchanged": runtime.get("weights_unchanged") is True,
                "no_parameter_grads": runtime.get("parameters_with_grad") == 0 and runtime.get("parameters_requiring_grad") == 0}
    seal = {"schema": SCHEMA, "phase": "A", "status": "SEALED", "manifest_hash": m["manifest_hash"],
            "config_sha256": CONFIG_SHA, "runtime_hash": m["runtime_hash"], "files": files,
            "archive": {"root": str(ARCHIVE), "content_addressed": archive},
            "counts": {"rows": len(gate_rows), "ok": sum(g["status"] == "ok" for g in gate_rows.values()),
                       "queries": sum(len(g.get("queries", [])) for g in gate_rows.values())},
            "metrics": metrics, "predicates": preds, "critical": critical,
            "git_head_at_seal": git("rev-parse", "HEAD"), "references_used": False, "created_unix": time.time()}
    seal["seal_hash"] = digest(seal)
    atomic_json(out, seal)
    print(json.dumps({"predicates": preds, "critical": critical, "metrics": metrics}, indent=1))


# ---- permutation / authorization ----------------------------------------------------------------------------

def sealed_gate_rows(cfg: dict) -> tuple[dict, list[dict]]:
    seal = json.loads((ROOT / GATE_SEAL).read_text())
    if not (committed(GATE_SEAL) and on_remote(GATE_SEAL)) or seal.get("status") != "SEALED":
        raise SystemExit("pushed gate seal required")
    rows = S.validate_runtime_panel(json.loads((ROOT / PANEL).read_text()))
    out = {}
    for r in rows:
        rel = f"gate/{r['utterance_id']}.json"
        p = run_dir() / rel
        if file_hash(p) != seal["files"][rel]:
            raise ValueError(f"sealed gate row changed {rel}")
        out[r["utterance_id"]] = json.loads(p.read_text())
    return out, rows


def cmd_permutation(args) -> None:
    cfg = load_cfg()
    if not (committed(AUDIT_A) and on_remote(AUDIT_A)) or json.loads((ROOT / AUDIT_A).read_text()).get("verdict") != "SRD2_G0_AUDIT_A: PASS":
        raise SystemExit("pushed SRD2_G0_AUDIT_A: PASS required before the permutation")
    if (ROOT / PERM).exists():
        raise FileExistsError("permutation exists")
    gate, rows = sealed_gate_rows(cfg)
    utts = []
    for r in rows:
        g = gate[r["utterance_id"]]
        gates = {x["t"]: float(x["g"]) for x in g["queries"] if x["structural"]["eligible"]}
        utts.append(S.permute(r["utterance_id"], gates))
    orig = [p["g"] for u in utts for p in u["pairs"]]
    shuf = [p["g_shuffled"] for u in utts for p in u["pairs"]]
    e = float(cfg["dose"]["e_star"])
    perm = {"schema": SCHEMA, "tag": S.PERMUTATION_TAG, "seed": S.SEED, "gate_seal_sha256": file_hash(ROOT / GATE_SEAL),
            "config_sha256": CONFIG_SHA, "utterances": utts,
            "multiset_original": S.bit_multiset(orig), "multiset_shuffled": S.bit_multiset(shuf),
            "planned_sq_B2": S.planned_sq_total([S.target_chord(e, x) for x in orig]),
            "planned_sq_B3": S.planned_sq_total([S.target_chord(e, x) for x in shuf]),
            "structural_queries": len(orig),
            "uninformative_queries": sum(u["n"] for u in utts if u["uninformative"]),
            "flags": {"singleton": sum(u["singleton"] for u in utts), "constant": sum(u["constant"] for u in utts),
                      "identity_values": sum(u["identity_values"] for u in utts), "empty": sum(u["n"] == 0 for u in utts)}}
    perm["multisets_equal"] = perm["multiset_original"] == perm["multiset_shuffled"]
    atomic_json(ROOT / PERM, perm)
    seal = {"schema": SCHEMA, "status": "SEALED", "permutation_sha256": file_hash(ROOT / PERM),
            "gate_seal_sha256": perm["gate_seal_sha256"], "audit_A_sha256": file_hash(ROOT / AUDIT_A),
            "multisets_equal": perm["multisets_equal"], "references_used": False, "created_unix": time.time()}
    atomic_json(ROOT / PERM_SEAL, seal)
    print(json.dumps({k: perm[k] for k in ("structural_queries", "uninformative_queries", "flags", "multisets_equal",
                                           "planned_sq_B2", "planned_sq_B3")}))


def resource_forecast(cfg: dict, gate: dict, runtime: dict) -> dict:
    c = cfg["compute"]
    ok = [g for g in gate.values() if g.get("status") == "ok"]
    n = sum(1 for g in ok for x in g["queries"] if x["structural"]["eligible"])
    cached = [s for g in ok for s in g["timing"]["cached_query_sec"]]
    enc = [g["encoder"]["encoder_sec"] for g in ok]
    wbytes = sum(g["arrays"]["bytes"] for g in ok)
    wsec = sum(g["arrays"]["write_sec"] for g in ok)
    io = sum(g["audio"]["io_sec"] + g["encoder"]["features_sec"] for g in ok)
    inputs = {"actual_structural_queries": n, "A_p95_cached_query_seconds": q(cached, .95),
              "A_durable_archive_bytes_per_second": (wbytes / wsec) if wsec > 0 else None,
              "A_model_load_seconds": runtime.get("model_load_sec"), "A_p95_original_encoder_seconds": q(enc, .95),
              "A_total_audio_IO_seconds": io}
    out = {"schema": SCHEMA, "inputs": inputs, "formula_forecast_seconds": c["job_B_cost_seconds_per_query"] * n + c["job_B_fixed_seconds"]}
    valid = all(isinstance(v, (int, float)) and v > 0 for v in inputs.values())
    if valid:
        per = max(.18, c["historical_D2_p95_sec"] + 3 * c["historical_pulse_p95_sec"] + 2 * inputs["A_p95_cached_query_seconds"]
                  + .04 + (4 * 51866 * 2 + 4 * 3 * 1280 * 2 + 32768) / inputs["A_durable_archive_bytes_per_second"])
        fixed = max(800, inputs["A_model_load_seconds"] + 400 * inputs["A_p95_original_encoder_seconds"]
                    + inputs["A_total_audio_IO_seconds"] + 60 + 120)
        out.update(observed_component_per_query=per, observed_component_fixed=fixed,
                   observed_component_forecast_seconds=n * per + fixed)
    else:
        out["observed_component_forecast_seconds"] = None
    out["metrics"] = {"formula_forecast_seconds": out["formula_forecast_seconds"],
                      "observed_component_forecast_seconds": out["observed_component_forecast_seconds"],
                      "scientific_jobs_planned": 2, "walltime_per_job_seconds": 10800}
    out["resource_pass"] = predicate(cfg["gate_predicates"]["resource"], out["metrics"])
    return out


def cmd_authorize_b(args) -> None:
    cfg = load_cfg()
    for x in (AUDIT_A, PERM, PERM_SEAL):
        if not (committed(x) and on_remote(x)):
            raise SystemExit(f"{x} must be committed and pushed")
    if (ROOT / AUTH).exists() or (ROOT / FORECAST).exists():
        raise FileExistsError("authorization exists")
    gate, _ = sealed_gate_rows(cfg)
    a = json.loads((ROOT / AUDIT_A).read_text())
    seal = json.loads((ROOT / GATE_SEAL).read_text())
    perm = json.loads((ROOT / PERM).read_text())
    runtime = json.loads((run_dir() / "capture_runtime.json").read_text())
    fc = resource_forecast(cfg, gate, runtime)
    atomic_json(ROOT / FORECAST, fc)
    authorized = bool(a.get("verdict") == "SRD2_G0_AUDIT_A: PASS" and seal["predicates"]["job_A_coverage"]
                      and seal["predicates"]["compatibility"] and all(seal["critical"].values())
                      and perm["multisets_equal"] and fc["resource_pass"])
    auth = {"schema": "srd2_g0_authorization_B_v1", "job_A_seal_hash": file_hash(ROOT / GATE_SEAL), "config_hash": CONFIG_SHA,
            "population_hash": cfg["population"]["selected_ids_hash"], "permutation_hash": file_hash(ROOT / PERM_SEAL),
            "audit_hash": file_hash(ROOT / AUDIT_A), "resource_forecast_hash": file_hash(ROOT / FORECAST),
            "authorized": authorized}
    atomic_json(ROOT / AUTH, auth)
    print(json.dumps({"authorized": authorized, "forecast": fc["metrics"], "resource_pass": fc["resource_pass"]}))


# ---- Job B: pulses ------------------------------------------------------------------------------------

class Engine:
    """Owner-scoped cached steps with the passive native observer, plus the one-solve pulse executor."""

    def __init__(self, ctx: Ctx):
        import experiments.inference_cf_p2r as p2r
        self.ctx, self.p2r = ctx, p2r

    def step(self, B, new, hook=None):
        from csasr.inference_cf.loc0_sites import Composite
        obs = S.NativeSite(self.ctx.bundle, self.ctx.layer)
        sync()
        t0 = time.perf_counter()
        logits, _, _ = B.step(new, capture_layer=None, attention=True, hook=Composite(hook, obs))
        sync()
        dt = time.perf_counter() - t0
        S.assert_no_hooks_anywhere(self.ctx.bundle)
        return logits, obs, dt

    def arm(self, B, L, new, t, query, v, target, site_obs) -> tuple[dict, object, object]:
        """One arm at absolute ``query``: zero target bypass, invalid direction, one solve, native preview
        guards, then the hook consuming the cached solve; FFN input must equal the preview bitwise."""
        import torch
        from csasr.inference_cf.core_p1 import processed_argmax
        from csasr.inference_cf.loc0_sites import cross_pulse_hook, pulse_action
        ctx, dose = self.ctx, self.ctx.cfg["dose"]
        rec = {"target": float(target), "status": None, "executed": False, "solver": None, "guards": None}
        if target == 0.0:
            rec["status"] = "zero_target"
            return rec, None, None
        if v is None:
            rec["status"] = "invalid_direction"
            return rec, None, None
        r_nat, q_nat, u_nat = site_obs.r[0], site_obs.q[0], site_obs.u[0]
        sol = self.p2r.solve_scale(r_nat, v, float(target))
        rec["solver"] = {k: sol.get(k) for k in ("status", "s", "phi", "theta", "edit_norm", "evals", "rel_sq_err")}
        if sol["status"] != "ok":
            rec["status"] = sol["status"]
            return rec, None, None
        if not sol["rel_sq_err"] <= dose["relative_squared_energy_error_max"]:
            rec["status"] = "solver_target_unattainable"
            return rec, None, None
        vv = v.detach().cpu().double()
        vv = vv / torch.linalg.vector_norm(vv)
        dirs = self.p2r.scaled_direction(vv, sol["s"], r_nat)
        prop, cons = S.native_preview(q_nat, u_nat, r_nat, dirs)
        guards = S.chord_guards(float(target), S.chord64(prop, r_nat), S.chord64(cons, r_nat), dose)
        rec["guards"] = guards
        if not guards["pass_all"]:
            rec["status"] = "native_consumption_unattainable"
            return rec, None, None
        B.crop(L)
        info: dict = {}
        solver = S.CachedSolver(r_nat, v, float(target), sol)
        action = pulse_action(query, lambda r: (v, "ok"), float(target), info,
                              dose["relative_squared_energy_error_max"], solver, self.p2r.scaled_direction)
        hook = cross_pulse_hook(ctx.bundle, ctx.layer, action, ctx.nfp)
        logits, obs, dt = self.step(B, new, hook=hook)
        audit = hook.records[-1].to_dict() if hook.records else None
        rec.update(executed=True, status="executed", pulse_sec=dt, solver_calls=solver.calls,
                   hook_steered=bool(audit and audit["steered"]), hook_edit_norm=audit["edit_norm"] if audit else None,
                   q_bitwise=bool(torch.equal(obs.q, site_obs.q)),
                   ffn_equals_preview_bitwise=bool(torch.equal(obs.ffn.reshape(-1), cons)),
                   proposal_equals_preview_bitwise=bool(torch.equal(info.get("_proposed", torch.empty(0)), prop.float().cpu())),
                   top1=processed_argmax(logits, t, ctx.suppress, ctx.begin),
                   consumed_chord_actual=S.chord64(obs.ffn.reshape(-1), r_nat))
        e = rec["hook_edit_norm"]
        rec["solver_matches_hook"] = bool(e is not None and abs(sol["edit_norm"] - e) <= dose["solver_vs_hook_abs_error_factor"] * max(1.0, e))
        rec["integrity_ok"] = bool(rec["solver_calls"] == 1 and rec["hook_steered"] and rec["q_bitwise"]
                                   and rec["ffn_equals_preview_bitwise"] and rec["solver_matches_hook"]
                                   and rec["proposal_equals_preview_bitwise"])
        return rec, logits, obs.ffn.reshape(-1)


def apparatus_old30(eng: Engine, cfg: dict) -> dict:
    """Historical engineering reproduction on the audited P2-DIR spot set (outside every new denominator)."""
    import torch
    import experiments.inference_cf_p2r as p2r
    from csasr.inference_cf.core_p1 import processed_argmax
    from csasr.inference_cf.directions import DirectionContext, ReadoutDirection
    ctx = eng.ctx
    ha = cfg["historical_apparatus"]
    sealed = json.loads((ROOT / APPARATUS["p2dir_sealed"]).read_text())
    by_uid = {}
    for rel, h in sealed["files"].items():
        if rel.endswith(".json"):
            p = ROOT / "results/inference_cf/p2dir/exp1_run1" / rel
            if file_hash(p) != h:
                raise ValueError(f"historical sealed row changed {rel}")
            by_uid[json.loads(p.read_text())["identity"]] = rel[:-5]
    bpanel = json.loads((ROOT / APPARATUS["baseline_panel"]).read_text())["rows"]
    bidx = {r["utterance_id"]: i for i, r in enumerate(bpanel)}
    audio = {r["utterance_id"]: r["audio_path"] for r in json.loads((ROOT / APPARATUS["audio_panel"]).read_text())["rows"]}
    want: dict = {}
    for x in ha["queries"]:
        want.setdefault(x["utterance_id"], set()).add(int(x["t"]))
    d2p = ReadoutDirection(ctx.bundle, layer=ctx.layer, suppress=ctx.suppress, begin=ctx.begin, partition=ctx.partition)
    res = []
    for uid in sorted(want):
        stem = ROOT / "results/inference_cf/p2dir/exp1_run1" / by_uid[uid]
        for suf in (".npz", "_logits.npz"):
            if file_hash(Path(str(stem) + suf)) != sealed["files"][by_uid[uid] + suf]:
                raise ValueError(f"historical sealed arrays changed {uid}{suf}")
        saved_row = json.loads(Path(str(stem) + ".json").read_text())["positions"]
        with np.load(str(stem) + ".npz") as z:
            vec = {k: z[k] for k in z.files}
        with np.load(str(stem) + "_logits.npz") as z:
            lg = {k: z[k] for k in z.files}
        toks = json.loads((ROOT / APPARATUS["baseline_rows"] / f"{bidx[uid]:03d}.json").read_text())["systems"]["B0M_L16"]["tokens"]
        enc, _ = encode(ctx.bundle, audio[uid])
        B = p2r.DiagBranch(ctx.bundle, enc, list(ctx.prompt), "B")
        new = list(ctx.prompt)
        for t in range(max(want[uid]) + 1):
            L = B.length
            d2 = None
            if t in want[uid]:
                d2 = d2p(DirectionContext(b_cache_pre_step=B.cache, encoded=enc, new_tokens=list(new), start=L, step=t))
            logits0, obs0, _ = eng.step(B, new)
            if t in want[uid]:
                pre = f"t{t}_"
                srow = saved_row[str(t)]
                r = {"utterance_id": uid, "t": t}
                r["clean_logits_bitwise"] = bool(np.array_equal(S.bf16_bits(logits0), lg[pre + "none"]))
                r["clean_site_bitwise"] = bool(np.array_equal(obs0.r[0].float().cpu().numpy(), vec[pre + "hb"]))
                r["scratch_identity"] = bool(torch.equal(d2.extras["logits"], logits0) and torch.equal(d2.extras["site"], obs0.r[0].float().cpu()))
                r["d2_status"] = d2.status
                r["d2_max_abs_error"] = float(np.abs(d2.direction.numpy().astype(np.float64) - vec[pre + "D2"].astype(np.float64)).max())
                r["J_abs_diff"] = abs(d2.provenance["J"] - srow["directions"]["D2"]["provenance"]["J"])
                arm, la, ffn = eng.arm(B, L, new, t, L + len(new) - 1, d2.direction, ctx.e_star, obs0)
                r["pulse"] = arm
                r["pulse_logits_bitwise"] = la is not None and bool(np.array_equal(S.bf16_bits(la), lg[pre + "D2"]))
                r["pulse_post_bitwise"] = ffn is not None and bool(np.array_equal(ffn.float().cpu().numpy(), vec[pre + "post_D2"]))
                r["solver_s_equal"] = arm["solver"] is not None and arm["solver"]["s"] == srow["arms"]["D2"]["solver"]["s"]
                # zero-dose: the DG-02 hook with a zero gain must be bitwise B0 (value-identical forward)
                B.crop(L)
                from csasr.inference_cf.loc0_sites import cross_pulse_hook
                zhook = cross_pulse_hook(ctx.bundle, ctx.layer, lambda q=None, u_source=None, r=None, abs_pos=None: (
                    torch.zeros((1, r.shape[1]), device=r.device, dtype=r.dtype), torch.zeros_like(r)), ctx.nfp)
                lz, oz, _ = eng.step(B, new, hook=zhook)
                r["zero_dose_bitwise"] = bool(torch.equal(lz, logits0) and torch.equal(oz.ffn, obs0.ffn))
                B.crop(L)
                l2, o2, _ = eng.step(B, new)
                r["restore_bitwise"] = bool(torch.equal(l2, logits0) and torch.equal(o2.r, obs0.r))
                r["argmax_matches_tokens"] = processed_argmax(logits0, t, ctx.suppress, ctx.begin) == (toks[t] if t < len(toks) else ctx.eos)
                r["ok"] = bool(r["clean_logits_bitwise"] and r["clean_site_bitwise"] and r["scratch_identity"] and r["d2_status"] == "ok"
                               and r["d2_max_abs_error"] <= ha["saved_D2_direction_max_abs_error"]
                               and r["J_abs_diff"] <= ha["objective_abs_tolerance"] and arm["executed"] and arm["integrity_ok"]
                               and r["pulse_logits_bitwise"] and r["pulse_post_bitwise"] and r["solver_s_equal"]
                               and r["zero_dose_bitwise"] and r["restore_bitwise"])
                res.append(r)
            if t < len(toks):
                new = [int(toks[t])]
    return {"n": len(res), "ok": len(res) == len(ha["queries"]) and all(r["ok"] for r in res), "positions": res,
            "outcome_denominator_inclusion": False}


def pulse_utterance(eng: Engine, row: dict, grow: dict, garr: dict, perm: dict, out_dir: Path, m: dict) -> dict:
    import torch
    import experiments.inference_cf_p2r as p2r
    from csasr.inference_cf.core_p1 import processed_argmax
    from csasr.inference_cf.directions import DirectionContext, ReadoutDirection
    ctx = eng.ctx
    uid = row["utterance_id"]
    if file_hash(row["audio_path"]) != row["audio_full_sha256"]:
        raise ValueError("audio bytes changed")
    OPENED.add(os.path.abspath(row["audio_path"]))
    enc, enc_t = encode(ctx.bundle, row["audio_path"])
    if enc_t["encoder_sha256"] != grow["encoder"]["encoder_sha256"]:
        raise RuntimeError("encoder output differs from Job A")
    content = grow["baseline"]["content_ids"]
    inv = grow["inventory"]
    shuffled = {p["recipient_t"]: p for p in perm["pairs"]}
    d2p = ReadoutDirection(ctx.bundle, layer=ctx.layer, suppress=ctx.suppress, begin=ctx.begin, partition=ctx.partition)
    B = p2r.DiagBranch(ctx.bundle, enc, list(ctx.prompt), "B")
    queries, dirs, grads, natq, natu, natr, ex_keys, ex_logits, ex_ffn = [], [], [], [], [], [], [], [], []
    for i, t in enumerate(inv):
        qa = grow["queries"][i]
        if qa["t"] != t:
            raise RuntimeError("Job A query order")
        new = S.feed_tokens(content, t, ctx.prompt)
        L = B.length
        query = L + len(new) - 1
        if query != qa["query"]:
            raise RuntimeError("absolute query mismatch")
        structural = qa["structural"]["eligible"]
        d2 = None
        if structural:
            d2 = d2p(DirectionContext(b_cache_pre_step=B.cache, encoded=enc, new_tokens=list(new), start=L, step=t))
        logits0, obs0, clean_sec = eng.step(B, new)
        top0 = processed_argmax(logits0, t, ctx.suppress, ctx.begin)
        rec = {"t": t, "query": query, "structural": structural,
               "clean": {"logits_bitwise_vs_A": bool(np.array_equal(S.bf16_bits(logits0), garr["cached_logits"][i])),
                         "site_bitwise_vs_A": bool(np.array_equal(S.bf16_bits(obs0.r[0].float()), garr["cached_site"][i])),
                         "top1": top0, "expected_action": qa["expected_action"], "sec": clean_sec},
               "arms": {"B0": {"status": "none", "executed": False, "top1": top0, "logits": "B0"}}}
        if top0 != qa["expected_action"]:
            raise RuntimeError("clean argmax differs from the sealed baseline action")
        if structural:
            pr = d2.provenance
            rec["d2"] = {"status": d2.status, "reason": d2.reason, "counters": d2.counters,
                         "runtime_sec": d2.extras["runtime_sec"],
                         **{k: pr.get(k) for k in ("J", "log_PE", "log_PM", "g_norm", "h_norm", "tangent_norm", "g_radial",
                                                   "unit_error", "radial_dot")},
                         "direction_sha256": None if d2.direction is None else S.array_sha(d2.direction.numpy()),
                         "scratch_logits_bitwise": bool(torch.equal(d2.extras["logits"], logits0)),
                         "scratch_site_bitwise": bool(torch.equal(d2.extras["site"], obs0.r[0].float().cpu()))}
            dirs.append(np.full(obs0.r.shape[-1], np.nan, np.float32) if d2.direction is None else d2.direction.numpy())
            g = d2.extras.get("gradient")
            grads.append(np.full(obs0.r.shape[-1], np.nan, np.float32) if g is None else g.numpy())
            natq.append(S.bf16_bits(obs0.q[0].float()))
            natu.append(S.bf16_bits(obs0.u[0].float()))
            natr.append(S.bf16_bits(obs0.r[0].float()))
            g2 = float(qa["g"])
            sp = shuffled[t]
            if sp["g_bits"] != qa["g_bits"]:
                raise RuntimeError("permutation original gate differs from the sealed gate")
            g3 = float(sp["g_shuffled"])
            targets = {"B1": ctx.e_star, "B2": S.target_chord(ctx.e_star, g2), "B3": S.target_chord(ctx.e_star, g3)}
            rec["gates"] = {"g": g2, "g_shuffled": g3, "donor_t": sp["donor_t"]}
            executed = False
            for a in S.EDIT_ARMS:
                ar, la, ffn = eng.arm(B, L, new, t, query, d2.direction, targets[a], obs0)
                if ar["executed"]:
                    executed = True
                    ex_keys.append([i, S.EDIT_ARMS.index(a)])
                    ex_logits.append(S.bf16_bits(la))
                    ex_ffn.append(S.bf16_bits(ffn.float()))
                    ar["logits"] = f"exec:{len(ex_keys) - 1}"
                else:
                    ar["top1"], ar["logits"] = top0, "B0"
                rec["arms"][a] = ar
            if executed:
                B.crop(L)
                l2, o2, _ = eng.step(B, new)
                rec["restore_bitwise"] = bool(torch.equal(l2, logits0) and torch.equal(o2.r, obs0.r) and torch.equal(o2.ffn, obs0.ffn))
            else:
                rec["restore_bitwise"] = None
        else:
            for a in S.EDIT_ARMS:
                rec["arms"][a] = {"target": 0.0, "status": "structurally_ineligible", "executed": False, "top1": top0, "logits": "B0"}
        queries.append(rec)
    D = int(ctx.bundle.d_model)
    V = int(garr["cached_logits"].shape[1])
    arrays = {"d2_direction": np.stack(dirs) if dirs else np.zeros((0, D), np.float32),
              "d2_gradient": np.stack(grads) if grads else np.zeros((0, D), np.float32),
              "native_q": np.stack(natq) if natq else np.zeros((0, D), np.int16),
              "native_u": np.stack(natu) if natu else np.zeros((0, D), np.int16),
              "native_r": np.stack(natr) if natr else np.zeros((0, D), np.int16),
              "exec_keys": np.asarray(ex_keys, dtype=np.int64).reshape(-1, 2),
              "exec_ffn": np.stack(ex_ffn) if ex_ffn else np.zeros((0, D), np.int16)}
    logit_arrays = {"exec_keys": arrays["exec_keys"],
                    "exec_logits": np.stack(ex_logits) if ex_logits else np.zeros((0, V), np.int16)}
    h1, b1, w1 = write_npz(out_dir / f"{uid}.npz", arrays)
    h2, b2, w2 = write_npz(out_dir / f"{uid}_logits.npz", logit_arrays)
    return {"schema": SCHEMA, "phase": "B", "identity": uid, "utterance_id": uid, "canonical_index": row["canonical_index"],
            "manifest_hash": m["manifest_hash"], "status": "ok", "encoder": enc_t, "inventory": inv,
            "structural_ts": [x["t"] for x in queries if x["structural"]], "queries": queries,
            "arrays": {"file": f"{uid}.npz", "sha256": h1, "bytes": b1, "write_sec": w1},
            "logits": {"file": f"{uid}_logits.npz", "sha256": h2, "bytes": b2, "write_sec": w2},
            "gate_row_digest": digest(grow)}


def cmd_pulses(args) -> None:
    import torch
    sys.addaudithook(_audit_open)
    m, cfg, rows = load_run("pulses")
    out = run_dir() / "pulses"
    if out.exists():
        raise FileExistsError("pulses directory exists; a pulse attempt is never overwritten")
    gate, _ = sealed_gate_rows(cfg)
    gseal = json.loads((ROOT / GATE_SEAL).read_text())
    perm = json.loads((ROOT / PERM).read_text())
    if file_hash(ROOT / PERM) != json.loads((ROOT / PERM_SEAL).read_text())["permutation_sha256"]:
        raise ValueError("permutation changed after its seal")
    perm_by = {u["utterance_id"]: u for u in perm["utterances"]}
    out.mkdir(parents=True)
    bundle, load_sec = load_model(cfg)
    bundle.model.requires_grad_(False)
    ctx = Ctx(bundle, cfg)
    eng = Engine(ctx)
    w0 = weights_digest(bundle)
    torch.cuda.reset_peak_memory_stats()
    rt = {"schema": SCHEMA, "phase": "B", "job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
          "gpu": torch.cuda.get_device_name(0), "manifest_hash": m["manifest_hash"], "authorization_sha256": file_hash(ROOT / AUTH),
          "start_unix": time.time(), "status": "running", "model_load_sec": load_sec, "weights_digest_start": w0}
    atomic_json(run_dir() / "pulse_runtime.json", rt)
    t0 = time.time()
    with torch.inference_mode():
        app = apparatus_old30(eng, cfg)
    app["elapsed_sec"] = time.time() - t0
    atomic_json(run_dir() / "apparatus_old30.json", app)
    rt["apparatus_ok"] = app["ok"]
    print(f"old30 apparatus ok={app['ok']} n={app['n']} {app['elapsed_sec']:.1f}s", flush=True)
    if not app["ok"]:
        rt.update(status="apparatus_failed", end_unix=time.time(), opened_paths=sorted(OPENED))
        atomic_json(run_dir() / "pulse_runtime.json", rt)
        raise SystemExit("historical apparatus reproduction failed: INVALID, no new pulses")
    failures = 0
    autograd = 0
    for row in rows:
        uid = row["utterance_id"]
        ts = time.time()
        try:
            gr = gate[uid]
            if gr["status"] != "ok":
                raise RuntimeError("Job A row not ok")
            gp = run_dir() / "gate" / gr["arrays"]["file"]
            if file_hash(gp) != gseal["files"][f"gate/{uid}.npz"]:
                raise ValueError("sealed gate arrays changed")
            with np.load(gp) as z:
                garr = {k: z[k] for k in ("cached_logits", "cached_site")}
            with torch.inference_mode():
                res = pulse_utterance(eng, row, gr, garr, perm_by[uid], out, m)
            autograd += sum(x.get("d2", {}).get("counters", {}).get("autograd_calls", 0) for x in res["queries"])
        except Exception as exc:
            import traceback
            failures += 1
            res = {"schema": SCHEMA, "phase": "B", "identity": uid, "utterance_id": uid, "canonical_index": row["canonical_index"],
                   "manifest_hash": m["manifest_hash"], "status": "failure", "reason": repr(exc), "traceback": traceback.format_exc()}
        res["elapsed_sec"] = time.time() - ts
        atomic_json(out / f"{uid}.json", res)
        print(f"B {row['canonical_index'] + 1}/{len(rows)} {uid} {res['status']} {res['elapsed_sec']:.1f}s", flush=True)
    S.assert_no_hooks_anywhere(bundle)
    rt.update(weights_digest_end=weights_digest(bundle), end_unix=time.time(), failures=failures, d2_autograd_calls=autograd,
              status="completed" if failures == 0 else "failed",
              peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(), peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(),
              parameters_with_grad=sum(p.grad is not None for p in bundle.model.parameters()),
              parameters_requiring_grad=sum(p.requires_grad for p in bundle.model.parameters()),
              opened_paths=sorted(OPENED))
    rt["elapsed_sec"] = rt["end_unix"] - rt["start_unix"]
    rt["weights_unchanged"] = rt["weights_digest_end"] == w0
    atomic_json(run_dir() / "pulse_runtime.json", rt)
    if failures:
        raise SystemExit(f"{failures} utterance failures")


def cmd_seal_b(args) -> None:
    m = json.loads((ROOT / MANIFEST).read_text())
    rows = S.validate_runtime_panel(json.loads((ROOT / PANEL).read_text()))
    out = ROOT / PULSE_SEAL
    if out.exists():
        raise FileExistsError("pulse seal exists")
    pdir = run_dir() / "pulses"
    files, archive = {}, {}
    n_ok = 0
    for r in rows:
        uid = r["utterance_id"]
        p = pdir / f"{uid}.json"
        if not p.exists():
            continue
        x = json.loads(p.read_text())
        if x["identity"] != uid or x["manifest_hash"] != m["manifest_hash"]:
            raise ValueError(f"pulse row identity {uid}")
        files[f"pulses/{uid}.json"] = file_hash(p)
        if x["status"] == "ok":
            n_ok += 1
            for key in ("arrays", "logits"):
                f = pdir / x[key]["file"]
                h = file_hash(f)
                if h != x[key]["sha256"]:
                    raise ValueError(f"pulse arrays changed {uid}")
                files[f"pulses/{x[key]['file']}"] = h
                archive[f"pulses/{x[key]['file']}"] = archive_copy(f, h)
    for name in ("pulse_runtime.json", "apparatus_old30.json"):
        files[name] = file_hash(run_dir() / name)
    for extra in sorted(run_dir().glob("slurm-B-*")):
        files[extra.name] = file_hash(extra)
    rt = json.loads((run_dir() / "pulse_runtime.json").read_text())
    seal = {"schema": SCHEMA, "phase": "B", "status": "SEALED", "manifest_hash": m["manifest_hash"], "config_sha256": CONFIG_SHA,
            "gate_seal_sha256": file_hash(ROOT / GATE_SEAL), "permutation_sha256": file_hash(ROOT / PERM),
            "authorization_sha256": file_hash(ROOT / AUTH), "files": files,
            "archive": {"root": str(ARCHIVE), "content_addressed": archive},
            "counts": {"rows": sum(1 for k in files if k.startswith("pulses/") and k.endswith(".json")), "ok": n_ok},
            "runtime_status": rt.get("status"), "git_head_at_seal": git("rev-parse", "HEAD"), "references_used": False,
            "created_unix": time.time()}
    seal["seal_hash"] = digest(seal)
    atomic_json(out, seal)
    print(json.dumps({"rows": seal["counts"], "runtime_status": seal["runtime_status"]}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prepare", "manifest", "capture", "seal-a", "permutation", "authorize-b", "pulses", "seal-b"))
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--out", default=RUN)
    args = ap.parse_args()
    if args.config != CONFIG or args.out.rstrip("/") != RUN:
        raise SystemExit(f"only the frozen config {CONFIG} and run root {RUN} are allowed")
    {"prepare": cmd_prepare, "manifest": cmd_manifest, "capture": cmd_capture, "seal-a": cmd_seal_a,
     "permutation": cmd_permutation, "authorize-b": cmd_authorize_b, "pulses": cmd_pulses, "seal-b": cmd_seal_b}[args.mode](args)


if __name__ == "__main__":
    main()
