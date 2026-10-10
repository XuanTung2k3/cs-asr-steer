#!/usr/bin/env python
"""DIR-SPRINT0 runner: three-family steering-direction discovery at L16 DG-02 (frozen config
``configs/inference_cf/dir_sprint0.json``; spec ``docs/inference_cf/DIR_SPRINT0_SPEC.md``).

Modes (each refuses an absent/failed audit, a changed source/config/model pin or a missing remote seal):

* ``prepare``      CPU  four-key runtime projection of the frozen 240 prospective + 300 calibration-bank utterances.
* ``manifest``     CPU  immutable source/config/population/model/provider/environment manifest.
* ``capture``      GPU  Job A: canonical cached forced-ZH B0, every query, R2 g_old (full replay), compatibility,
                        same-prefix E / translate / null-audio branches, D3 candidate decisions, v_AC regions and masked
                        states; calibration-bank B0 states and R2 windows; two CPU child processes compute the full-audio
                        phone posteriors. No pulse, no readout autograd, no reference. Ends with ``construct``.
* ``phones``       CPU  (Job A child) frozen phone-provider apparatus check, then full-audio posteriors.
* ``construct``    CPU  D5 evidence, calibration prototypes and every non-autograd direction (D0, D1, D4, D5, D5SH,
                        v_AC, random) from the sealed Job A arrays.
* ``seal-a``       CPU  Job A digest seal + durable content-addressed archive + reference-free predicates.
* ``authorize-b``  CPU  complete planned-pulse recost and the minimal global authorization.
* ``pulses``       GPU  Job B: old30 historical apparatus, then D2 and D3 autograd once per structural query and the
                        fifteen pristine single-query arms. No reference.
* ``seal-b``       CPU  Job B digest seal + durable archive.
* ``smoke``        CPU  engineering smoke on public engineering audio only (scratch directory; no barrier, no outcome).

No reference transcript, target set, stratum, CTC/MMS-FA timing, language label or evaluator quantity is read here.
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
from csasr.inference_cf import dir_sprint0 as X
from csasr.inference_cf import srd2_g0 as S

SCHEMA = "dir_sprint0_run_v1"
CONFIG = "configs/inference_cf/dir_sprint0.json"
CONFIG_SHA = "sha256:b264460beedd09ae4b0c005ef5f8178494c49a52aa9ba2bec4bc116c984d27db"
FREEZE = "docs/inference_cf/DIR_SPRINT0_FREEZE.json"
DESIGN_COMMIT = "c54d50271857fb802b9576bcda6a22b81cf85be8"
POPULATION = "docs/inference_cf/DIR_SPRINT0_POPULATION.json"
RUN = "results/inference_cf/dir_sprint0/run1"
REMOTE = "origin/cs-asr-steer-inf"
ARCHIVE = Path("/mnt/data/tungnx/cs-asr-steer/archives/dir_sprint0/run1")
PY = "/home/tungnx/miniconda3/envs/acl1/bin/python"
PROVIDER_MANIFEST = "results/inference_cf/dir_sprint0/d5_provider/provider_manifest.json"
FEATURE_TABLE = "results/inference_cf/dir_sprint0/d5_provider/feature_table.json"
CONCEPT_TABLE = "results/inference_cf/dir_sprint0/design/d5_concept_table.json"
PROVIDER_REF = "results/inference_cf/dir_sprint0/design/d5_provider_reference.json"
PROVIDER_REF_NPZ = "results/inference_cf/dir_sprint0/design/d5_provider_reference.npz"
OWN = ("src/csasr/inference_cf/dir_sprint0.py", "experiments/inference_cf_dir_sprint0.py", "slurm/inference_cf_dir_sprint0.sbatch")
NOT_EXECUTED = ("experiments/inference_cf_dir_sprint0_evaluate.py", "experiments/inference_cf_dir_sprint0_audit.py",
                "tests/test_dir_sprint0_impl.py", "tests/test_dir_sprint0_freeze.py")
REUSED = ("src/csasr/__init__.py", "src/csasr/inference_cf/__init__.py", "src/csasr/utils/__init__.py",
          "src/csasr/utils/config.py", "src/csasr/utils/hashing.py", "src/csasr/utils/logging.py",
          "src/csasr/data/__init__.py", "src/csasr/data/alignment.py", "src/csasr/models/__init__.py",
          "src/csasr/lss/__init__.py", ".gitignore", PROVIDER_REF, PROVIDER_REF_NPZ,
          "experiments/inference_cf_dir_sprint0_population.py", "experiments/inference_cf_dir_sprint0_d5_concepts.py")
APPARATUS = {"p2dir_sealed": "results/inference_cf/p2dir/exp1_run1/directions_sealed.json",
             "p2dir_rows": "results/inference_cf/p2dir/exp1_run1/rows",
             "baseline_panel": "results/inference_cf/p2_A_r1_L16/panel.json",
             "baseline_rows": "results/inference_cf/p2_A_r1_L16/rows",
             "audio_panel": "results/inference_cf/p0_r2/inference_panel.json"}
PANEL = f"{RUN}/runtime_panel.json"
MANIFEST = f"{RUN}/manifest.json"
PRE = f"{RUN}/audit_PRE.json"
A_SEAL = f"{RUN}/job_A_seal.json"
AUDIT_A = f"{RUN}/audit_A.json"
FORECAST = f"{RUN}/resource_forecast.json"
AUTH = f"{RUN}/authorization_B.json"
PULSE_SEAL = f"{RUN}/pulse_seal.json"
PROVIDER_FAILURES = ("localizer_fail", "local_support_fail", "baseline_provider_fail", "nonfinite_signal")
FAMILY_ARRAYS = ("D0", "D1", "D4", "D5", "D5SH", "VAC", "RND")


# ---- git / provenance ----------------------------------------------------------------------------------------------

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


def verify_freeze(cfg: dict) -> dict:
    fr = json.loads((ROOT / FREEZE).read_text())
    for rel, h in fr["files"].items():
        if file_hash(ROOT / rel) != h:
            raise ValueError(f"frozen design file changed: {rel}")
    pop = cfg["population"]
    if file_hash(ROOT / pop["manifest"]) != pop["file_sha256"]:
        raise ValueError("population file hash")
    P = json.loads((ROOT / pop["manifest"]).read_text())
    if digest({k: v for k, v in P.items() if k != "manifest_hash"}) != P["manifest_hash"] or P["manifest_hash"] != pop["manifest_hash"]:
        raise ValueError("population manifest hash")
    if digest(P["selected"]) != pop["selected_hash"] or digest([r["utterance_id"] for r in P["selected"]]) != pop["selected_ids_hash"]:
        raise ValueError("population selection hash")
    if digest(P["calibration_bank"]["rows"]) != pop["calibration_bank"]["rows_hash"]:
        raise ValueError("calibration bank hash")
    return P


def pinned_sources(cfg: dict) -> list[str]:
    return sorted(set(cfg["source_sha256"]) | set(OWN) | set(NOT_EXECUTED) | set(REUSED) | {CONFIG, FREEZE, PANEL, POPULATION,
                                                                                         CONCEPT_TABLE})


def environment() -> dict:
    import platform
    import torch
    import transformers
    return {"python": platform.python_version(), "interpreter": sys.executable, "torch": torch.__version__,
            "transformers": transformers.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
            "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}


class Paths:
    """Run-directory layout (the frozen run root, or a scratch root for the engineering smoke)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.capture = self.root / "capture"
        self.bank = self.root / "bank"
        self.phones = self.root / "phones"
        self.construct = self.root / "construct"
        self.pulses = self.root / "pulses"


def run_paths() -> Paths:
    return Paths(ROOT / RUN)


# ---- prepare / manifest --------------------------------------------------------------------------------------------

def cmd_prepare(args) -> None:
    cfg = load_cfg()
    P = verify_freeze(cfg)
    out = ROOT / PANEL
    if out.exists():
        raise FileExistsError("runtime panel exists; never overwrite")
    panel = X.runtime_projection(P)
    if len(panel["rows"]) != cfg["population"]["utterances"] or len(panel["bank_rows"]) != 300:
        raise ValueError("selected / bank count")
    atomic_json(out, panel)
    print(json.dumps({"rows": len(panel["rows"]), "bank_rows": len(panel["bank_rows"]), "runtime_hash": panel["runtime_hash"]}))


def cmd_manifest(args) -> None:
    import inspect
    from transformers import WhisperTokenizer
    from transformers.models.whisper.modeling_whisper import WhisperDecoderLayer
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.inference_cf.phone_provider import load_manifest
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
    for name, br in cfg["branches"].items():
        if isinstance(br, dict) and "tokens" in br and tok.convert_ids_to_tokens(br["prompt"]) != br["tokens"]:
            raise ValueError(f"branch {name} prompt tokens differ")
    if gen["task_to_id"] != {"transcribe": 50360, "translate": 50359}:
        raise ValueError("task ids differ")
    legal = X.legal_static(tok, gen["suppress_tokens"])
    if len(legal) != cfg["families"]["D3"]["legal_static"]["count"] or digest(legal) != cfg["families"]["D3"]["legal_static"]["hash"]:
        raise ValueError("D3 legal static set differs from the freeze")
    pm = load_manifest(ROOT / PROVIDER_MANIFEST)
    if pm["manifest_digest"] != cfg["families"]["D5"]["provider_manifest_digest"]:
        raise ValueError("provider manifest digest")
    for name, h in pm["model"]["files"].items():
        if file_hash(Path(pm["model"]["dir"]) / name) != h:
            raise ValueError(f"provider model file changed: {name}")
    panel = json.loads((ROOT / PANEL).read_text())
    X.validate_runtime_panel(panel)
    env = environment()
    if env["torch"] != cfg["environment"]["torch"] or env["transformers"] != cfg["environment"]["transformers"]:
        raise ValueError("library versions differ from the freeze")
    if not (len(DESIGN_COMMIT) == 40 and subprocess.run(["git", "merge-base", "--is-ancestor", DESIGN_COMMIT, "HEAD"], cwd=ROOT).returncode == 0):
        raise ValueError("DESIGN_COMMIT must name the pushed design-freeze commit")
    man = {"schema": SCHEMA, "design_freeze_commit": DESIGN_COMMIT,
           "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"), "config": CONFIG,
           "config_sha256": CONFIG_SHA, "config_hash": digest(cfg),
           "freeze_index_hash": json.loads((ROOT / FREEZE).read_text())["freeze_index_hash"],
           "population": {k: cfg["population"][k] for k in ("file_sha256", "manifest_hash", "selected_hash", "selected_ids_hash",
                                                            "roster_hash")},
           "bank_rows_hash": cfg["population"]["calibration_bank"]["rows_hash"],
           "runtime_panel": PANEL, "runtime_hash": panel["runtime_hash"], "environment": env,
           "model": {"dir": str(mdir), "files": model_files, "precision": "bfloat16", "attention": "eager"},
           "transformers_forward_sha256": cfg["environment"]["transformers_forward_sha256"],
           "decoder_layer_forward_source_sha256": "sha256:" + hashlib.sha256(inspect.getsource(WhisperDecoderLayer.forward).encode()).hexdigest(),
           "partition_hash": part["hash"], "partition_counts": counts, "suppression_hash": supp,
           "alignment_heads": cfg["gate"]["alignment_heads"],
           "prompts": {k: v["prompt"] for k, v in cfg["branches"].items() if isinstance(v, dict)},
           "D3_legal_static_hash": digest(legal), "provider_manifest_digest": pm["manifest_digest"],
           "provider_model_files": pm["model"]["files"],
           "sources": {s: file_hash(ROOT / s) for s in srcs},
           "apparatus_inputs": {"p2dir_sealed": file_hash(ROOT / APPARATUS["p2dir_sealed"]),
                                "baseline_panel": file_hash(ROOT / APPARATUS["baseline_panel"]),
                                "audio_panel": file_hash(ROOT / APPARATUS["audio_panel"])},
           "archive_root": str(ARCHIVE),
           "provenance": {"code_config_snapshot_hash": code_config_snapshot_hash(ROOT), "test_snapshot_hash": test_snapshot_hash(ROOT)},
           "reference_access_boundary": "runner phases read the four-key runtime panel, audio, Whisper, the frozen phone provider, "
                                        "the pinned D1 folds / provider tables and (Job B only) the hash-pinned old30 apparatus "
                                        "artifacts; references only in the separate evaluator after the pushed pulse seal and "
                                        "DIR_SPRINT0_AUDIT_PRIMARY: PASS",
           "trainable_parameters": 0, "optimizer": None, "references_used": False, "status": "FROZEN", "created_unix": time.time()}
    man["manifest_hash"] = digest(man)
    atomic_json(out, man)
    print(json.dumps({"manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"], "sources": len(man["sources"])}))


# ---- GPU-phase guards ------------------------------------------------------------------------------------------------

OPENED: set = set()


def _audit_open(event, args) -> None:
    """Runtime file-open log (firewall evidence): every path opened outside the Python installation."""
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.path.abspath(os.fsdecode(args[0]))
        if not p.startswith((sys.prefix, sys.base_prefix, "/proc", "/sys", "/dev", "/usr", "/lib", "/etc",
                             "/tmp", "/run", "/var")) and "/.git/" not in p and "__pycache__" not in p:
            OPENED.add(p)


def load_run(stage: str) -> tuple[dict, dict, list[dict], list[dict]]:
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
    rows, bank = X.validate_runtime_panel(panel)
    if panel["runtime_hash"] != m["runtime_hash"]:
        raise ValueError("runtime panel hash")
    if not (committed(MANIFEST) and on_remote(MANIFEST)):
        raise SystemExit("the manifest must be committed and pushed")
    pre = json.loads((ROOT / PRE).read_text())
    if not (committed(PRE) and on_remote(PRE) and pre.get("verdict") == "PASS_TO_DIR_SPRINT0"
            and pre.get("manifest_hash") == m["manifest_hash"]):
        raise SystemExit("independent PASS_TO_DIR_SPRINT0 (committed + pushed, same manifest) required")
    if stage == "pulses":
        need = (A_SEAL, AUDIT_A, FORECAST, AUTH)
        if not all(committed(x) and on_remote(x) for x in need):
            raise SystemExit("Job B requires the pushed Job A seal, audit A, forecast and authorization")
        auth = json.loads((ROOT / AUTH).read_text())
        if list(auth) != cfg["firewall"]["authorization_B_fields"] or auth["authorized"] is not True:
            raise SystemExit("authorization B absent, malformed or not authorized")
        expect = {"job_A_seal_hash": file_hash(ROOT / A_SEAL), "config_hash": CONFIG_SHA,
                  "population_hash": cfg["population"]["selected_ids_hash"], "audit_hash": file_hash(ROOT / AUDIT_A),
                  "resource_forecast_hash": file_hash(ROOT / FORECAST)}
        if any(auth[k] != v for k, v in expect.items()):
            raise SystemExit("authorization B hashes do not match the sealed artifacts")
        if json.loads((ROOT / AUDIT_A).read_text()).get("verdict") != "DIR_SPRINT0_AUDIT_A: PASS":
            raise SystemExit("DIR_SPRINT0_AUDIT_A: PASS required")
    return m, cfg, rows, bank


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
        if self.nfp != 4 or self.eos != X.EOS:
            raise ValueError("forced prefix / EOS mismatch")
        native = gen.lang_to_id
        self.language_ids = tuple(sorted(set(int(x) for x in native.values())))
        if len(self.language_ids) != 100:
            raise ValueError("unexpected language-token vocabulary")
        self.en_id, self.zh_id = int(native["<|en|>"]), int(native["<|zh|>"])
        if dict(gen.task_to_id) != {"transcribe": 50360, "translate": 50359}:
            raise ValueError("task ids changed")
        self.byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
        self.e_star = float(cfg["dose"]["e_star"])
        self.prompt = list(X.CB)
        branches = {k: v for k, v in cfg["branches"].items() if isinstance(v, dict)}
        self.prompts = {k: list(v["prompt"]) for k, v in branches.items()}
        for k, v in branches.items():
            if "tokens" in v and self.tok.convert_ids_to_tokens(v["prompt"]) != v["tokens"]:
                raise ValueError(f"branch {k} prompt tokens changed")
        if self.prompts["B0"] != list(X.CB) or self.prompts["E"] != list(X.CE) or self.prompts["TL"] != list(X.CTL) \
                or self.prompts["NULL"] != list(X.CB):
            raise ValueError("branch prompts differ from the mechanics module")
        self.max_new = int(cfg["baseline"]["max_new_tokens"])
        self.layer = int(cfg["model"]["layer_zero_based"])
        legal = X.legal_static(self.tok, self.suppress)
        if digest(legal) != cfg["families"]["D3"]["legal_static"]["hash"]:
            raise ValueError("D3 legal static set changed")
        self.legal_mask = np.zeros(int(bundle.model.config.vocab_size), dtype=bool)
        self.legal_mask[legal] = True
        self.null_enc = None


def load_model(cfg: dict, smoke: bool = False):
    import torch
    from csasr.models.whisper import load_whisper
    from csasr.utils.config import load_config
    torch.manual_seed(X.SEED)
    torch.set_num_threads(1 if not smoke else max(1, int(os.environ.get("SMOKE_THREADS", "16"))))
    t0 = time.perf_counter()
    mc = load_config(ROOT / "configs/model/whisper_large_v3.yaml")
    if smoke:
        mc["model"]["device"] = "cpu"
    bundle = load_whisper(mc)
    if not smoke and (bundle.device != "cuda" or bundle.dtype != torch.bfloat16):
        raise ValueError("requires CUDA bf16")
    if bundle.dtype != torch.bfloat16:
        raise ValueError("bf16 required")
    if getattr(bundle.model.config, "_attn_implementation", "eager") != "eager":
        raise ValueError("attention implementation must be eager")
    bundle.model.eval()
    bundle.model.requires_grad_(False)
    sync()
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


def encode_path(bundle, audio_path: str):
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
                                                   "encoder_sha256": S.tensor_sha(h),
                                                   "features_sha256": S.tensor_sha(inputs["input_features"]),
                                                   "attention_mask_sha256": S.tensor_sha(inputs["attention_mask"])}


def encode_wave(bundle, x: np.ndarray):
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.lexical_compatibility import waveform_model_inputs
    feats = waveform_model_inputs(bundle, x)
    with torch.inference_mode():
        h = bundle.model.model.encoder(input_features=feats["input_features"], attention_mask=feats["attention_mask"],
                                       return_dict=True).last_hidden_state
    return BaseModelOutput(last_hidden_state=h), {"encoder_sha256": S.tensor_sha(h),
                                                   "features_sha256": S.tensor_sha(feats["input_features"]),
                                                   "attention_mask_sha256": S.tensor_sha(feats["attention_mask"])}


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


def stack_bits(xs: list, width: int) -> np.ndarray:
    return np.stack([S.bf16_bits(x) for x in xs]) if xs else np.zeros((0, width), np.int16)


# ---- Job A: capture --------------------------------------------------------------------------------------------------

def cached_baseline(ctx: Ctx, enc, *, teacher: list[int] | None = None, keep_heads: bool = False) -> dict:
    """Canonical pristine cached B0 (explicit lowest-ID processed argmax); with ``teacher`` the same cached path is
    replayed on fixed content (bitwise identity evidence)."""
    import experiments.inference_cf_p2r as p2r
    from csasr.inference_cf.core_p1 import processed_argmax
    B = p2r.DiagBranch(ctx.bundle, enc, list(ctx.prompt), "B")
    new, content, terminated = list(ctx.prompt), [], "cap"
    logits, sites, heads, qsec, argmax = [], [], [], [], []
    for t in range(ctx.max_new):
        site = S.NativeSite(ctx.bundle, ctx.layer)
        sync()
        t0 = time.perf_counter()
        lg, hd, _ = B.step(new, capture_layer=None, attention=True, hook=site)
        sync()
        qsec.append(time.perf_counter() - t0)
        S.assert_no_hooks_anywhere(ctx.bundle)
        logits.append(lg)
        sites.append(site.r[0].float().cpu())
        if keep_heads:
            heads.append(hd)
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
    return {"content": content, "terminated": terminated, "logits": logits, "sites": sites, "heads": heads,
            "query_sec": qsec, "argmax": argmax, "lineage_ok": bool(lineage)}


def branch_replay(ctx: Ctx, enc, prompt: list[int], content: list[int], inv: list[int], name: str) -> dict:
    """Same-prefix cached branch with its own cold cache: at each inventory t feed prompt (t=0) or content[t-1];
    record the native L16 DG-02 site and float32 logits at absolute query 4+t-1."""
    import experiments.inference_cf_p2r as p2r
    B = p2r.DiagBranch(ctx.bundle, enc, list(prompt), name)
    logits, sites, ident, qsec = [], [], [], []
    for t in inv:
        new = S.feed_tokens(content, t, prompt)
        site = S.NativeSite(ctx.bundle, ctx.layer)
        sync()
        t0 = time.perf_counter()
        lg, _, _ = B.step(new, capture_layer=None, attention=True, hook=site)
        sync()
        qsec.append(time.perf_counter() - t0)
        S.assert_no_hooks_anywhere(ctx.bundle)
        logits.append(lg)
        sites.append(site.r[0].float().cpu())
        ident.append(B.fed == list(prompt) + [int(x) for x in content[:t]] and B.length - 1 == S.query_index(t, len(prompt))
                     and B.positions == list(range(B.length)))
    return {"logits": logits, "sites": sites, "identity": ident, "query_sec": qsec}


def capture_utterance(ctx: Ctx, row: dict, null_probs: dict, primitives: dict, out_dir: Path, m: dict) -> dict:
    import torch
    import experiments.inference_cf_p0_r2 as r2
    import experiments.inference_cf_p2r as p2r
    from csasr.inference_cf.core_p1 import processed_argmax
    from csasr.inference_cf.core_r2 import conflict_from_logits, prefix_utf8_complete
    from csasr.inference_cf.lexical_compatibility import hard_mask
    from csasr.lss.sites import DecoderPostCrossAttnRecorder
    from csasr.models.whisper import load_audio
    cfg = ctx.cfg
    uid = row["utterance_id"]
    t_io = time.perf_counter()
    if file_hash(row["audio_path"]) != row["audio_full_sha256"]:
        raise ValueError("audio bytes changed after the population freeze")
    waveform = load_audio(row["audio_path"], ctx.bundle.sample_rate)
    io_sec = time.perf_counter() - t_io
    OPENED.add(os.path.abspath(row["audio_path"]))
    heard = min(len(waveform), 480000)
    enc, enc_t = encode_path(ctx.bundle, row["audio_path"])
    base = cached_baseline(ctx, enc, keep_heads=True)
    content, terminated = base["content"], base["terminated"]
    inv = S.inventory(content, terminated)
    if len(base["logits"]) != len(inv):
        raise RuntimeError("inventory/cached query count mismatch")
    replay = cached_baseline(ctx, enc, teacher=content)
    replay_bitwise = (replay["terminated"] == terminated and replay["content"] == content and len(replay["logits"]) == len(inv)
                      and all(torch.equal(a, b) for a, b in zip(replay["logits"], base["logits"]))
                      and all(torch.equal(a, b) for a, b in zip(replay["sites"], base["sites"])))
    # ---- original R2 full causal replay (no cache) on the exact cached-B0 content ----------------------------------
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

    # ---- same-prefix branches (own cold caches) -----------------------------------------------------------------------
    t_br = time.perf_counter()
    brE = branch_replay(ctx, enc, ctx.prompts["E"], content, inv, "E")
    brT = branch_replay(ctx, enc, ctx.prompts["TL"], content, inv, "TL")
    brN = branch_replay(ctx, ctx.null_enc, ctx.prompts["NULL"], content, inv, "NULL")
    branch_sec = time.perf_counter() - t_br
    # ---- v_AC: R0 region track (fixed LID window grid) ----------------------------------------------------------------
    t_r0 = time.perf_counter()
    vac_cfg = cfg["families"]["VAC"]
    track = X.vac_track(waveform, lid, ctx.language_ids, vac_cfg["regions"])
    r0_sec = time.perf_counter() - t_r0
    queries, structural_ts = [], []
    vac_targets: dict = {}
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
        b_t = S.expected_action(content, t, ctx.eos)
        qr = {"t": t, "query": q, "prefix_sha256": S.prefix_hash(ctx.prompt, prefix), "expected_action": b_t,
              "is_eos_query": t == len(content), "remaining_content_tokens": len(content) - t, "utf8_complete": bool(utf8),
              "structural": {"eligible": ok, "reasons": why}, "cached_argmax": base["argmax"][i], "gate": gate,
              "g": float(gate["g"]), "g_bits": S.float_bits(gate["g"]),
              "branches": {"E": {"identity": brE["identity"][i], "site_finite": bool(torch.isfinite(brE["sites"][i]).all())},
                           "TL": {"identity": brT["identity"][i], "site_finite": bool(torch.isfinite(brT["sites"][i]).all()),
                                  "logits_finite": bool(torch.isfinite(brT["logits"][i]).all())},
                           "NULL": {"identity": brN["identity"][i], "site_finite": bool(torch.isfinite(brN["sites"][i]).all()),
                                    "logits_finite": bool(torch.isfinite(brN["logits"][i]).all())}}}
        if base["argmax"][i] != b_t:
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
            # descriptive task-language entanglement of the translate branch at the same query
            ct = conflict_from_logits(brT["logits"][i], ctx.partition) if qr["branches"]["TL"]["logits_finite"] else None
            qr["branches"]["TL"].update(top1=processed_argmax(brT["logits"][i], t, ctx.suppress, ctx.begin),
                                        P_E=None if ct is None else ct["P_E"], P_M=None if ct is None else ct["P_M"])
            qr["branches"]["NULL"]["top1"] = processed_argmax(brN["logits"][i], t, ctx.suppress, ctx.begin)
            # D3 frozen candidate decision (B0 cached raw logits vs null-audio raw logits)
            if qr["branches"]["NULL"]["logits_finite"] and qr["branches"]["NULL"]["identity"]:
                qr["D3"] = X.d3_candidate(lc.numpy(), brN["logits"][i].numpy(), ctx.legal_mask, ctx.suppress, b_t)
            else:
                qr["D3"] = {"status": "nonfinite_branch", "c_AP": None, "baseline_token": b_t}
            qr["D3CD_top1"] = qr["D3"]["c_AP"] if qr["D3"]["status"] == "candidate" else b_t
            # v_AC S1 association on the cached B0 query attention
            tgt = X.vac_target(base["heads"][i].float().numpy(), track["track_heard_intervals"], waveform, heard,
                               vac_cfg["s1_regions"], vac_cfg["heard_mass_min"])
            qr["VAC_target"] = tgt
            if tgt["status"] == "OK":
                vac_targets.setdefault(tuple(int(b) for b in tgt["bounds"]), []).append(i)
        queries.append(qr)
    # ---- v_AC masked states (one cold forced-ZH replay per distinct target mask) --------------------------------------
    t_m = time.perf_counter()
    mask_records, mask_sites, mask_keys = [], [], []
    zero_change = None
    if vac_targets:
        _, zc = encode_wave(ctx.bundle, waveform)
        zero_change = bool(zc["features_sha256"] == enc_t["features_sha256"] and zc["attention_mask_sha256"] == enc_t["attention_mask_sha256"]
                           and zc["encoder_sha256"] == enc_t["encoder_sha256"])
        for (a, b), members in sorted(vac_targets.items()):
            if not zero_change:
                mask_records.append({"bounds": [a, b], "status": "zero_change_identity_failed", "members": [inv[j] for j in members]})
                continue
            xm = hard_mask(waveform.astype(np.float32), a, b)
            menc, mlin = encode_wave(ctx.bundle, xm)
            want = {inv[j]: j for j in members}
            B = p2r.DiagBranch(ctx.bundle, menc, list(ctx.prompt), f"MASK_{a}_{b}")
            got = {}
            for t in range(max(want) + 1):
                site = S.NativeSite(ctx.bundle, ctx.layer)
                B.step(S.feed_tokens(content, t, ctx.prompt), capture_layer=None, attention=True, hook=site)
                S.assert_no_hooks_anywhere(ctx.bundle)
                if t in want:
                    got[t] = site.r[0].float().cpu()
            ident = B.fed == list(ctx.prompt) + [int(x) for x in content[:max(want)]]
            for t, j in sorted(want.items()):
                mask_keys.append([j, a, b])
                mask_sites.append(got[t])
            mask_records.append({"bounds": [a, b], "status": "ok" if ident else "prefix_identity_failed",
                                 "members": sorted(want), "masked_encoder": mlin,
                                 "masked_waveform_sha256": S.array_sha(xm), "outside_identical": bool(
                                     np.array_equal(xm[:a], waveform[:a]) and np.array_equal(xm[b:], waveform[b:])),
                                 "inside_zero": bool(np.all(xm[a:b] == 0))})
    mask_sec = time.perf_counter() - t_m
    # ---- fixed causal probes: prefix-only vs full replay (first / median / last structural query) --------------------
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
    V, D = int(full_logits.shape[1]), int(ctx.bundle.d_model)
    arrays = {"cached_logits": stack_bits(base["logits"], V),
              "full_logits": S.bf16_bits(full_logits[qidx]),
              "heads": S.bf16_bits(heads[:, qidx].permute(1, 0, 2).contiguous()),
              "cached_heads": np.stack([S.bf16_bits(h) for h in base["heads"]]),
              "cached_site": stack_bits(base["sites"], D),
              "full_site": S.bf16_bits(full_site[qidx]),
              "E_site": stack_bits(brE["sites"], D), "TL_site": stack_bits(brT["sites"], D),
              "NULL_site": stack_bits(brN["sites"], D), "NULL_logits": stack_bits(brN["logits"], V),
              "mask_keys": np.asarray(mask_keys, dtype=np.int64).reshape(-1, 3), "mask_site": stack_bits(mask_sites, D),
              "r0_window_probs": track["probs"].astype(np.float32),
              "r0_grid": np.asarray(track["grid"], dtype=np.int64).reshape(-1, 2),
              "r0_track_heard_intervals": np.asarray(track["track_heard_intervals"], dtype=np.int64).reshape(-1, 3),
              "lid_keys": np.asarray(lid_order, dtype=np.int64).reshape(-1, 2),
              "lid_probs": np.asarray([[lid_cache[k][i] for i in ctx.language_ids] for k in lid_order],
                                      dtype=np.float64).reshape(-1, len(ctx.language_ids)),
              "probe_logits": np.stack(probe_logits) if probe_logits else np.zeros((0, V), np.int16),
              "probe_heads": np.stack(probe_heads) if probe_heads else np.zeros((0, heads.shape[0], heads.shape[2]), np.int16)}
    npz_sha, nbytes, wsec = write_npz(out_dir / f"{uid}.npz", arrays)
    return {"schema": SCHEMA, "phase": "A", "identity": uid, "utterance_id": uid, "canonical_index": row["canonical_index"],
            "manifest_hash": m["manifest_hash"], "status": "ok",
            "audio": {"full_sha256": row["audio_full_sha256"], "samples": int(len(waveform)), "heard": heard, "io_sec": io_sec},
            "encoder": enc_t, "baseline": {"content_ids": content, "terminated": terminated, "T": len(content),
                                           "lineage_ok": base["lineage_ok"], "replay_bitwise": bool(replay_bitwise)},
            "inventory": inv, "structural_ts": structural_ts, "queries": queries,
            "full_replay": {"sec": full_sec, "future_mass_max": fm.max_mass, "self_attn_calls": fm.calls,
                            "input_ids_sha256": S.prefix_hash(ctx.prompt, content), "shared_encoder": True},
            "probes": probes, "lid": {**lid_counts, "language_ids_sha256": digest(list(ctx.language_ids)),
                                      "language_order": list(ctx.language_ids)},
            "r0": {"windows": track["windows"], "heard": track["heard"], "sec": r0_sec},
            "vac_masks": {"zero_change_identity": zero_change, "masks": mask_records, "sec": mask_sec},
            "timing": {"cached_query_sec": base["query_sec"], "replay_query_sec": replay["query_sec"], "branch_sec": branch_sec,
                       "branch_query_sec": {"E": brE["query_sec"], "TL": brT["query_sec"], "NULL": brN["query_sec"]}},
            "arrays": {"file": f"{uid}.npz", "sha256": npz_sha, "bytes": nbytes, "write_sec": wsec,
                       "keys": {k: [str(v.dtype), list(v.shape)] for k, v in arrays.items()}}}


def capture_bank(ctx: Ctx, row: dict, out_dir: Path, m: dict) -> dict:
    """Calibration bank: canonical cached B0 L16 DG-02 states at every query and the R2 window (full-replay heads)."""
    import torch
    import experiments.inference_cf_p0_r2 as r2
    from csasr.inference_cf.core_r2 import max_attention_window, prefix_utf8_complete
    from csasr.models.whisper import load_audio
    uid = row["utterance_id"]
    if file_hash(row["audio_path"]) != row["audio_full_sha256"]:
        raise ValueError("bank audio bytes changed after the population freeze")
    waveform = load_audio(row["audio_path"], ctx.bundle.sample_rate)
    OPENED.add(os.path.abspath(row["audio_path"]))
    enc, enc_t = encode_path(ctx.bundle, row["audio_path"])
    base = cached_baseline(ctx, enc)
    content, terminated = base["content"], base["terminated"]
    inv = S.inventory(content, terminated)
    full_logits, heads = r2.full_replay(ctx.bundle, enc, list(ctx.prompt), content, attention=True)
    S.assert_no_hooks_anywhere(ctx.bundle)
    queries = []
    for i, t in enumerate(inv):
        q = S.query_index(t, len(ctx.prompt))
        prefix = content[:t]
        utf8 = prefix_utf8_complete(ctx.tok, prefix, ctx.byte_decoder)
        ok, why = S.structural(t, prefix, utf8, bool(torch.isfinite(base["sites"][i]).all()),
                               bool(torch.isfinite(base["logits"][i]).all()), ctx.eos)
        try:
            w = max_attention_window(heads[:, q].numpy(), len(waveform)).record()
            wstat = "ok"
        except Exception as exc:
            w, wstat = None, str(exc)
        b_t = S.expected_action(content, t, ctx.eos)
        queries.append({"t": t, "query": q, "structural": {"eligible": ok, "reasons": why}, "window": w,
                        "window_status": wstat, "b_t": b_t,
                        "b_t_class": "EOS" if b_t == ctx.eos else ("LATIN" if b_t in ctx.partition_sets["embedded"] else
                                                                 ("HAN" if b_t in ctx.partition_sets["matrix"] else "OTHER"))})
    D = int(ctx.bundle.d_model)
    arrays = {"cached_site": stack_bits(base["sites"], D)}
    npz_sha, nbytes, wsec = write_npz(out_dir / f"{uid}.npz", arrays)
    return {"schema": SCHEMA, "phase": "A_bank", "identity": uid, "utterance_id": uid, "bank_index": row["bank_index"],
            "manifest_hash": m["manifest_hash"], "status": "ok",
            "audio": {"full_sha256": row["audio_full_sha256"], "samples": int(len(waveform))}, "encoder": enc_t,
            "baseline": {"content_ids": content, "terminated": terminated, "T": len(content), "lineage_ok": base["lineage_ok"]},
            "inventory": inv, "queries": queries, "timing": {"cached_query_sec": base["query_sec"]},
            "arrays": {"file": f"{uid}.npz", "sha256": npz_sha, "bytes": nbytes, "write_sec": wsec}}


def spawn_phones(paths: Paths, which: str, smoke: bool) -> subprocess.Popen:
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=f"{ROOT / 'src'}:{ROOT}")
    cmd = [PY, str(ROOT / "experiments/inference_cf_dir_sprint0.py"), "phones", "--which", which, "--root", str(paths.root)]
    if smoke:
        cmd.append("--smoke")
    log = open(paths.root / f"phones_{which}.log", "w")
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def cmd_capture(args, smoke_rows: tuple | None = None, smoke_paths: Paths | None = None) -> None:
    import torch
    import experiments.inference_cf_p0_r2 as r2
    from csasr.inference_cf.core_r2 import conflict_from_logits, gate_values, local_support, max_attention_window
    smoke = smoke_rows is not None
    sys.addaudithook(_audit_open)
    if smoke:
        m, cfg, rows, bank = {"manifest_hash": "smoke"}, load_cfg(), smoke_rows[0], smoke_rows[1]
        paths = smoke_paths
    else:
        m, cfg, rows, bank = load_run("capture")
        paths = run_paths()
    for d in (paths.capture, paths.bank, paths.phones):
        if d.exists():
            raise FileExistsError(f"{d} exists; a capture attempt is never overwritten")
        d.mkdir(parents=True)
    children = {w: spawn_phones(paths, w, smoke) for w in ("prospective", "bank")}
    bundle, load_sec = load_model(cfg, smoke=smoke)
    ctx = Ctx(bundle, cfg)
    ctx.partition_sets = {"embedded": set(ctx.partition["embedded_ids"]), "matrix": set(ctx.partition["matrix_ids"])}
    w0 = weights_digest(bundle)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rt = {"schema": SCHEMA, "phase": "A", "job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-smoke", "manifest_hash": m["manifest_hash"],
          "start_unix": time.time(), "status": "running", "model_load_sec": load_sec, "weights_digest_start": w0,
          "phone_children": {w: c.pid for w, c in children.items()}}
    atomic_json(paths.root / "capture_runtime.json", rt)
    zeros = np.zeros(30 * bundle.sample_rate, dtype=np.float32)
    null = r2.native_lid(bundle, zeros, ctx.language_ids)
    rt["null"] = {"EN": null[ctx.en_id], "ZH": null[ctx.zh_id],
                  "EN_abs_error": abs(null[ctx.en_id] - cfg["gate"]["null_EN_historical"]),
                  "ZH_abs_error": abs(null[ctx.zh_id] - cfg["gate"]["null_ZH_historical"]),
                  "probs_sha256": digest([null[i] for i in ctx.language_ids])}
    ctx.null_enc, nlin = encode_wave(bundle, zeros)
    rt["null_encoder"] = {**nlin, "waveform_sha256": S.array_sha(zeros)}
    primitives = {"max_attention_window": max_attention_window, "conflict_from_logits": conflict_from_logits,
                  "local_support": local_support, "gate_values": gate_values, "reason": r2._reason}
    failures = 0
    for row in rows:
        t0 = time.time()
        try:
            res = capture_utterance(ctx, row, null, primitives, paths.capture, m)
        except Exception as exc:
            import traceback
            failures += 1
            res = {"schema": SCHEMA, "phase": "A", "identity": row["utterance_id"], "utterance_id": row["utterance_id"],
                   "canonical_index": row["canonical_index"], "manifest_hash": m["manifest_hash"], "status": "failure",
                   "reason": repr(exc), "traceback": traceback.format_exc()}
        res["elapsed_sec"] = time.time() - t0
        atomic_json(paths.capture / f"{row['utterance_id']}.json", res)
        print(f"A {row['canonical_index'] + 1}/{len(rows)} {row['utterance_id']} {res['status']} "
              f"q={len(res.get('inventory', []))} {res['elapsed_sec']:.1f}s", flush=True)
    bank_failures = 0
    for row in bank:
        t0 = time.time()
        try:
            res = capture_bank(ctx, row, paths.bank, m)
        except Exception as exc:
            import traceback
            bank_failures += 1
            res = {"schema": SCHEMA, "phase": "A_bank", "identity": row["utterance_id"], "utterance_id": row["utterance_id"],
                   "bank_index": row["bank_index"], "manifest_hash": m["manifest_hash"], "status": "failure",
                   "reason": repr(exc), "traceback": traceback.format_exc()}
        res["elapsed_sec"] = time.time() - t0
        atomic_json(paths.bank / f"{row['utterance_id']}.json", res)
        print(f"BANK {row['bank_index'] + 1}/{len(bank)} {row['utterance_id']} {res['status']} {res['elapsed_sec']:.1f}s", flush=True)
    S.assert_no_hooks_anywhere(bundle)
    gpu_end = time.time()
    child_rc = {w: c.wait() for w, c in children.items()}
    rt.update(weights_digest_end=weights_digest(bundle), end_unix=time.time(), gpu_loop_end_unix=gpu_end, failures=failures,
              bank_failures=bank_failures, phone_children_returncodes=child_rc,
              status="completed" if failures == 0 else "failed",
              peak_vram_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
              peak_vram_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
              parameters_with_grad=sum(p.grad is not None for p in bundle.model.parameters()),
              parameters_requiring_grad=sum(p.requires_grad for p in bundle.model.parameters()),
              opened_paths=sorted(OPENED))
    rt["elapsed_sec"] = rt["end_unix"] - rt["start_unix"]
    rt["weights_unchanged"] = rt["weights_digest_end"] == w0
    atomic_json(paths.root / "capture_runtime.json", rt)
    del bundle
    try:
        construct(paths, cfg, smoke=smoke)
    except Exception as exc:
        import traceback
        atomic_json(paths.root / "construct_error.json", {"reason": repr(exc), "traceback": traceback.format_exc()})
        print(f"construct failed (rerun 'construct' on CPU): {exc!r}", flush=True)
    if failures:
        raise SystemExit(f"{failures} utterance failures")


# ---- Job A child: phone provider ----------------------------------------------------------------------------------------

def cmd_phones(args) -> None:
    """Frozen phone-provider apparatus check, then full-audio posteriors for one row list (CPU, float32, 4 threads)."""
    import torch
    import soundfile as sf
    from csasr.inference_cf.phone_provider import PhoneProvider
    sys.addaudithook(_audit_open)
    torch.set_num_threads(4)
    paths = Paths(Path(args.root))
    cfg = load_cfg()
    panel = json.loads((paths.root / "runtime_panel.json").read_text()) if args.smoke else json.loads((ROOT / PANEL).read_text())
    rows, bank = X.validate_runtime_panel(panel)
    todo = rows if args.which == "prospective" else bank
    rt = {"schema": SCHEMA, "phase": "A_phones", "which": args.which, "start_unix": time.time(), "status": "running",
          "threads": torch.get_num_threads(), "node": socket.gethostname()}
    d5 = cfg["families"]["D5"]
    prov = PhoneProvider(ROOT / PROVIDER_MANIFEST, ROOT / FEATURE_TABLE, device="cpu")
    if prov.manifest["manifest_digest"] != d5["provider_manifest_digest"] or prov.table.digest != d5["feature_table_digest"]:
        raise ValueError("provider manifest / feature table differ from the freeze")
    ref = json.loads((ROOT / PROVIDER_REF).read_text())
    if file_hash(ROOT / PROVIDER_REF_NPZ) != ref["npz_sha256"]:
        raise ValueError("provider reference arrays changed")
    app = {}
    with np.load(ROOT / PROVIDER_REF_NPZ) as z:
        for clip in d5["provider_apparatus_clips"]:
            meta = prov.manifest["engineering_audio"]["files"][clip]
            p = Path(prov.manifest["engineering_audio"]["dir"]) / clip
            if file_hash(p) != meta["sha256"]:
                raise ValueError(f"engineering clip changed: {clip}")
            x, sr = sf.read(str(p), dtype="float32")
            r1, r2_ = prov.posteriors(x, sr), prov.posteriors(x, sr)
            refa = z[ref["npz_keys"][clip]]
            same_shape = r1["log_probs"].shape == refa.shape
            diff = float(np.abs(r1["log_probs"].astype(np.float64) - refa).max()) if same_shape else None
            agree = float((r1["log_probs"].argmax(1) == refa.argmax(1)).mean()) if same_shape else None
            tol = d5["provider_apparatus_tolerance"]
            app[clip] = {"repeat_bitwise": bool(np.array_equal(r1["log_probs"], r2_["log_probs"])),
                         "sealed_hash_equal": r1["log_probs_sha256"] == ref["clips"][clip]["log_probs_sha256"],
                         "max_abs_logprob_diff": diff, "argmax_agreement": agree}
            app[clip]["pass"] = bool(app[clip]["repeat_bitwise"] and (app[clip]["sealed_hash_equal"] or (
                diff is not None and diff <= tol["max_abs_logprob"] and agree is not None and agree >= tol["argmax_agreement"])))
    rt["apparatus"] = app
    rt["apparatus_pass"] = all(v["pass"] for v in app.values())
    atomic_json(paths.root / f"phones_runtime_{args.which}.json", rt)
    if not rt["apparatus_pass"]:
        rt.update(status="apparatus_failed", end_unix=time.time(), opened_paths=sorted(OPENED))
        atomic_json(paths.root / f"phones_runtime_{args.which}.json", rt)
        raise SystemExit("phone provider apparatus failed: D5 PROVIDER_OR_CONSTRUCTION_BLOCKED")
    failures = 0
    for r in todo:
        uid = r["utterance_id"]
        t0 = time.time()
        rec = {"schema": SCHEMA, "phase": "A_phones", "identity": uid, "utterance_id": uid, "which": args.which}
        try:
            if file_hash(r["audio_path"]) != r["audio_full_sha256"]:
                raise ValueError("audio bytes changed")
            x, sr = sf.read(r["audio_path"], dtype="float32", always_2d=False)
            out = prov.posteriors(x, sr)
            rec.update(status=out["status"], reason=out["reason"], n_samples=out["n_samples"], n_frames=out["n_frames"],
                       input_sha256=out["input_sha256"], log_probs_sha256=out["log_probs_sha256"], provenance=out["provenance"])
            if out["status"] == "ok":
                h, nb, ws = write_npz(paths.phones / f"{uid}.npz", {"log_probs": out["log_probs"]})
                rec["arrays"] = {"file": f"{uid}.npz", "sha256": h, "bytes": nb}
        except Exception as exc:
            import traceback
            failures += 1
            rec.update(status="failure", reason=repr(exc), traceback=traceback.format_exc())
        rec["elapsed_sec"] = time.time() - t0
        atomic_json(paths.phones / f"{uid}.json", rec)
    rt.update(status="completed" if failures == 0 else "failed", failures=failures, end_unix=time.time(),
              opened_paths=sorted(OPENED))
    rt["elapsed_sec"] = rt["end_unix"] - rt["start_unix"]
    atomic_json(paths.root / f"phones_runtime_{args.which}.json", rt)


# ---- construct (CPU): D5 evidence / prototypes and every non-autograd direction ---------------------------------------

def construct(paths: Paths, cfg: dict, smoke: bool = False) -> dict:
    import torch
    from csasr.inference_cf.phone_provider import FeatureTable
    from csasr.inference_cf.readout import tangent_unit
    if smoke:   # two-clip engineering bank: exercise the code path only (scratch root; never the frozen thresholds' run)
        X.DIALOGUES_MIN, X.SIDE_TOTAL_MIN, X.DIALOGUE_SIDE_MIN = 1, 1, 1
    out = paths.construct
    if out.exists():
        raise FileExistsError("construct directory exists; never overwritten")
    out.mkdir(parents=True)
    t_start = time.time()
    P = json.loads((ROOT / POPULATION).read_text())
    if smoke:
        P = json.loads((paths.root / "smoke_population.json").read_text())
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in P["selected"]}
    bdlg = {r["utterance_id"]: r["dialogue_id"] for r in P["calibration_bank"]["rows"]}
    d5 = cfg["families"]["D5"]
    table = FeatureTable(ROOT / FEATURE_TABLE, expected_digest=d5["feature_table_digest"])
    ct = json.loads((ROOT / CONCEPT_TABLE).read_text())
    if ct["concept_table_digest"] != d5["concept_table_digest"] or file_hash(ROOT / CONCEPT_TABLE) != d5["concept_table_sha256"]:
        raise ValueError("concept table differs from the freeze")
    C, classes = X.concept_matrix(ct, len(table))
    valid = table.valid.copy()
    excluded = table.status == "excluded"
    tone_bearing = valid & (table.tone > 0)
    phones_rt = {w: json.loads((paths.root / f"phones_runtime_{w}.json").read_text()) for w in ("prospective", "bank")}
    # family-wide: apparatus passed and each child finished its loop; a single utterance's provider failure only
    # abstains that utterance's evidence (provider_failure)
    provider_ok = all(r.get("apparatus_pass") is True and r.get("status") in ("completed", "failed") for r in phones_rt.values())

    def phones(uid: str):
        pj = paths.phones / f"{uid}.json"
        if not pj.exists():
            return None, "provider_failure"
        rec = json.loads(pj.read_text())
        if rec.get("status") != "ok":
            return None, "provider_failure"
        f = paths.phones / rec["arrays"]["file"]
        if file_hash(f) != rec["arrays"]["sha256"]:
            raise ValueError(f"phone arrays changed {uid}")
        with np.load(f) as z:
            lp = z["log_probs"]
        return lp, None

    def evidence(lp, why, window) -> dict:
        if window is None:
            return {"status": "no_window"}
        if lp is None:
            return {"status": why}
        w = X.window_weights(window["start_sample"], window["end_sample"], lp.shape[0])
        ev = X.concept_evidence(lp, w, valid, excluded, C)
        if ev["status"] == "ok":
            p = np.exp(lp.astype(np.float64))
            s = p[:, valid].sum(axis=1)
            we = w * (s >= X.EMIT_MIN)
            ev["tone_bearing_share"] = float((we @ p[:, tone_bearing]).sum() / ev["segmental_mass"])
        ev["window"] = [int(window["start_sample"]), int(window["end_sample"])]
        return ev

    # ---- calibration bank evidence ----
    bank_rows = sorted((json.loads(p.read_text()) for p in paths.bank.glob("*.json")), key=lambda r: r["bank_index"])
    bank_ok = [b for b in bank_rows if b.get("status") == "ok"]
    BQ, BH, BD, BT, BC, BREL, bank_ev = [], [], [], [], [], [], {}
    for b in bank_ok:
        uid = b["utterance_id"]
        f = paths.bank / b["arrays"]["file"]
        if file_hash(f) != b["arrays"]["sha256"]:
            raise ValueError(f"bank arrays changed {uid}")
        with np.load(f) as z:
            sites = S.from_bf16_bits(z["cached_site"])
        lp, why = phones(uid)
        recs = []
        for i, q in enumerate(b["queries"]):
            if not q["structural"]["eligible"]:
                continue
            ev = evidence(lp, why, q["window"])
            recs.append({"t": q["t"], **{k: v for k, v in ev.items() if k not in ("valid_phone_mass",)}})
            if ev["status"] == "ok":
                BQ.append(ev["q"])
                BH.append(sites[i])
                BD.append(bdlg[uid])
                BT.append(q["t"])
                BREL.append(q["t"] / max(1, b["baseline"]["T"]))
                BC.append(q["b_t_class"])
        bank_ev[uid] = recs
    atomic_json(out / "bank_evidence.json", {"schema": SCHEMA, "rows": bank_ev})
    proto = {"status": "not_built"}
    blocked = []
    if not provider_ok:
        blocked.append("provider_apparatus_or_child_failure")
    if len(bank_ok) != len(P["calibration_bank"]["rows"]) and not smoke:
        blocked.append("bank_capture_incomplete")
    if BQ:
        proto = X.build_prototypes(np.asarray(BQ), np.asarray(BH), BD)
        if proto["status"] != "ok":
            blocked.append("prototype_construction_failed")
    else:
        blocked.append("no_valid_bank_evidence")
    diag = {}
    if BQ:
        Qa, Ua = np.asarray(BQ), X.unit_rows(np.asarray(BH))
        from scipy.stats import spearmanr
        cls = np.asarray(BC)
        dl = np.asarray(BD)
        diag["position_spearman_t"] = [float(spearmanr(Qa[:, k], BT).statistic) for k in range(Qa.shape[1])]
        diag["position_spearman_relative"] = [float(spearmanr(Qa[:, k], BREL).statistic) for k in range(Qa.shape[1])]
        diag["baseline_token_class_mean_q"] = {c: Qa[cls == c].mean(axis=0).tolist() for c in sorted(set(cls.tolist())) if (cls == c).sum()}
        diag["baseline_token_class_counts"] = {c: int((cls == c).sum()) for c in sorted(set(cls.tolist()))}
        tot = Qa.var(axis=0)
        between = np.zeros(Qa.shape[1])
        for d in sorted(set(dl.tolist())):
            mk = dl == d
            between += mk.mean() * (Qa[mk].mean(axis=0) - Qa.mean(axis=0)) ** 2
        diag["dialogue_eta2"] = (between / np.where(tot > 0, tot, np.nan)).tolist()
        if proto.get("status") == "ok" and (cls == "LATIN").sum() and (cls == "HAN").sum():
            vs = Ua[cls == "LATIN"].mean(axis=0) - Ua[cls == "HAN"].mean(axis=0)
            vs /= np.linalg.norm(vs)
            diag["cos_prototype_vs_baseline_script_axis"] = (proto["V"] @ vs).tolist()
        diag["n_valid_bank_queries"] = int(len(BQ))
    pr = {k: v for k, v in proto.items() if k not in ("V", "mu", "sd")}
    if proto.get("status") == "ok":
        np.savez(out / "prototypes.npz", V=proto["V"], mu=proto["mu"], sd=proto["sd"])
        pr["arrays_sha256"] = file_hash(out / "prototypes.npz")
        pr["mu"], pr["sd"] = proto["mu"].tolist(), proto["sd"].tolist()
    pr["diagnostics"] = diag
    pr["blocked_reasons"] = blocked
    pr["bank_rows_ok"] = len(bank_ok)
    pr["bank_structural_queries"] = sum(len(v) for v in bank_ev.values())
    atomic_json(out / "prototypes.json", pr)
    D5_ok = not blocked
    V = proto["V"] if D5_ok else None
    # ---- D1 fold vectors (hash-verified) ----
    d1 = cfg["families"]["D1"]
    folds = json.loads((ROOT / d1["folds"]).read_text())
    if file_hash(ROOT / d1["folds"]) != d1["folds_sha256"]:
        raise ValueError("D1 folds file changed")
    fold_vec = {}
    for d, f in folds["folds"].items():
        vp = ROOT / f["vector_path"]
        if file_hash(vp) != d1["vector_file_sha256"][d] or f["status"] != "ok":
            raise ValueError(f"D1 fold vector changed {d}")
        v = np.load(vp).astype(np.float32)
        from csasr.inference_cf.unique import array_hash
        if array_hash(v) != d1["vector_sha256"][d]:
            raise ValueError(f"D1 vector hash {d}")
        fold_vec[d] = v
    # ---- prospective directions ----
    cap_rows = sorted((json.loads(p.read_text()) for p in paths.capture.glob("*.json")), key=lambda r: r["canonical_index"])
    summary = {f: {"valid": 0, "invalid": {}} for f in FAMILY_ARRAYS}
    summary["D3"] = {"candidate": 0, "baseline_is_candidate": 0, "no_plausible_legal_candidate": 0, "nonfinite_branch": 0}
    shuffle_log, ev_all = [], {}
    n_struct = 0
    for r in cap_rows:
        if r.get("status") != "ok":
            continue
        uid = r["utterance_id"]
        f = paths.capture / r["arrays"]["file"]
        if file_hash(f) != r["arrays"]["sha256"]:
            raise ValueError(f"capture arrays changed {uid}")
        with np.load(f) as z:
            hB = S.from_bf16_bits(z["cached_site"])
            hE = S.from_bf16_bits(z["E_site"])
            hT = S.from_bf16_bits(z["TL_site"])
            mkeys = z["mask_keys"]
            msite = S.from_bf16_bits(z["mask_site"]) if len(mkeys) else np.zeros((0, X.DIM), np.float32)
        lp, why = phones(uid)
        masks_ok = {tuple(mr["bounds"]): mr["status"] == "ok" for mr in r["vac_masks"]["masks"]}
        mask_of = {int(j): (int(a), int(b), k) for k, (j, a, b) in enumerate(mkeys.tolist())}
        sts = [i for i, q in enumerate(r["queries"]) if q["structural"]["eligible"]]
        n_struct += len(sts)
        arr = {fam: np.full((len(sts), X.DIM), np.nan, np.float32) for fam in FAMILY_ARRAYS}
        stat = []
        evs = {}
        zrec = {}
        for k, i in enumerate(sts):
            q = r["queries"][i]
            t = q["t"]
            h = hB[i]
            st = {"t": t}
            r0 = X.d0_direction(hE[i], h)
            if r0["d"] is not None:
                arr["D0"][k] = r0["d"].numpy()
            st["D0"] = {"status": r0["status"], "norm": r0["norm"] if math.isfinite(r0["norm"]) else None}
            arr["D1"][k] = fold_vec[dlg[uid]]
            st["D1"] = {"status": "ok", "fold": dlg[uid]}
            r4 = X.d4_direction(h, hT[i])
            if r4["status"] == "ok":
                arr["D4"][k] = r4["direction"].numpy()
            st["D4"] = {k2: (None if isinstance(r4.get(k2), float) and not math.isfinite(r4[k2]) else r4.get(k2))
                        for k2 in ("status", "reason", "raw_norm", "tangent_norm", "h_norm", "g_radial", "unit_error", "radial_dot")}
            arr["RND"][k] = X.random_direction(uid, t)
            st["RND"] = {"status": "ok"}
            tg = q.get("VAC_target", {})
            if tg.get("status") == "OK":
                key = (int(tg["bounds"][0]), int(tg["bounds"][1]))
                if i in mask_of and masks_ok.get(key) and mask_of[i][:2] == key:
                    rv = X.vac_direction(h, msite[mask_of[i][2]])
                    if rv["status"] == "OK":
                        arr["VAC"][k] = rv["vector"]
                    st["VAC"] = {"status": rv["status"], "raw_norm": rv.get("raw_norm")}
                else:
                    st["VAC"] = {"status": "mask_unavailable"}
            else:
                st["VAC"] = {"status": f"target_{tg.get('status')}"}
            ev = evidence(lp, why, q["gate"].get("window"))
            evs[t] = ev
            if D5_ok and ev["status"] == "ok":
                r5 = X.d5_direction(ev["q"], proto["mu"], proto["sd"], V, h)
                st["D5"] = {"status": r5["status"], "reason": r5["reason"], "z_norm": r5.get("z_norm"), "u_norm": r5.get("u_norm"),
                            "tangent_norm": r5.get("tangent_norm")}
                if r5["status"] == "ok":
                    arr["D5"][k] = r5["direction"].numpy()
                    zrec[t] = np.asarray(r5["z"], dtype=np.float64)
            else:
                st["D5"] = {"status": "blocked" if not D5_ok else f"evidence_{ev['status']}"}
            st["D3"] = {"status": q["D3"]["status"], "c_AP": q["D3"].get("c_AP")}
            summary["D3"][q["D3"]["status"]] = summary["D3"].get(q["D3"]["status"], 0) + 1
            stat.append(st)
        # D5 shuffled evidence (within-utterance derangement over D5-valid queries)
        sh = X.d5_shuffle(uid, sorted(zrec))
        shuffle_log.append({**{k2: v for k2, v in sh.items() if k2 != "donor"},
                            "donor": {str(a): b for a, b in sh["donor"].items()}})
        for k, i in enumerate(sts):
            t = r["queries"][i]["t"]
            if t in sh["donor"]:
                dz = zrec[sh["donor"][t]]
                rs = tangent_unit(torch.from_numpy(dz @ V), torch.as_tensor(hB[i]))
                stat[k]["D5SH"] = {"status": rs["status"], "reason": rs["reason"], "donor_t": sh["donor"][t],
                                   "singleton": sh["singleton"], "z_identical": bool(np.array_equal(dz, zrec[t]))}
                if rs["status"] == "ok":
                    arr["D5SH"][k] = rs["direction"].numpy()
                    if not np.isnan(arr["D5"][k]).any():
                        stat[k]["D5SH"]["cos_vs_D5"] = float(np.dot(arr["D5"][k].astype(np.float64), arr["D5SH"][k].astype(np.float64)))
            else:
                stat[k]["D5SH"] = {"status": "not_in_D5_pool"}
        for k in range(len(sts)):
            for fam in FAMILY_ARRAYS:
                s_ = stat[k][fam]["status"]
                good = s_ in ("ok", "OK")
                if good:
                    summary[fam]["valid"] += 1
                else:
                    summary[fam]["invalid"][s_] = summary[fam]["invalid"].get(s_, 0) + 1
        h_, nb, _ = write_npz(out / f"{uid}.npz", {**arr, "structural_ts": np.asarray([r["queries"][i]["t"] for i in sts], np.int64)})
        rec = {"schema": SCHEMA, "phase": "A_construct", "identity": uid, "utterance_id": uid, "structural_ts": [r["queries"][i]["t"] for i in sts],
               "status": stat, "arrays": {"file": f"{uid}.npz", "sha256": h_, "bytes": nb}}
        atomic_json(out / f"{uid}.json", rec)
        ev_all[uid] = {str(t): {k2: v for k2, v in e.items() if k2 != "valid_phone_mass"} for t, e in evs.items()}
    atomic_json(out / "prospective_evidence.json", {"schema": SCHEMA, "rows": ev_all})
    atomic_json(out / "shuffle.json", {"schema": SCHEMA, "tag": X.SHUFFLE_TAG, "seed": X.SEED, "utterances": shuffle_log,
                                       "singleton_utterances": sum(s["singleton"] for s in shuffle_log),
                                       "singleton_queries": sum(s["n"] for s in shuffle_log if s["singleton"]),
                                       "pool_queries": sum(s["n"] for s in shuffle_log)})
    summ = {"schema": SCHEMA, "D5_blocked_reasons": blocked, "D5_constructible": D5_ok, "structural_queries": n_struct,
            "families": summary, "elapsed_sec": time.time() - t_start, "provider_runtime": {w: {k: v for k, v in r.items() if k != "opened_paths"}
                                                                                         for w, r in phones_rt.items()}}
    atomic_json(out / "construct_summary.json", summ)
    print(json.dumps({"construct": {f: summary[f] for f in summary}, "D5_blocked": blocked}, indent=None), flush=True)
    return summ


def cmd_construct(args) -> None:
    """CPU rerun of the deterministic construct stage (only if the in-job construct raised; failure preserved)."""
    cfg = load_cfg()
    paths = run_paths()
    if not (paths.root / "construct_error.json").exists():
        raise SystemExit("construct already ran inside Job A; nothing to rerun")
    if paths.construct.exists():
        shutil.move(str(paths.construct), str(paths.root / f"construct_failed_{int(time.time())}"))
    construct(paths, cfg)


# ---- seal / archive helpers -------------------------------------------------------------------------------------------

def archive_copy(src: Path, sha: str) -> str:
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
    P = json.loads((ROOT / POPULATION).read_text())
    return {r["utterance_id"]: r["dialogue_id"] for r in P["selected"]}


def q(xs: list, p: float) -> float | None:
    return float(np.quantile(np.asarray(xs, dtype=np.float64), p)) if xs else None


def predicate_all(checks: dict) -> bool:
    return all(bool(v) for v in checks.values())


def job_a_metrics(cfg: dict, rows: list[dict], cap: dict, runtime: dict) -> dict:
    dlg = dialogue_of()
    ok = {u: r for u, r in cap.items() if r.get("status") == "ok"}
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
    allq = [x for r in ok.values() for x in r["queries"]]
    br_ident = [x["branches"][b]["identity"] for x in allq for b in ("E", "TL", "NULL")]
    br_fin = [x["branches"][b]["site_finite"] for x in allq for b in ("E", "TL", "NULL")]
    null_fin = [x["branches"]["NULL"]["logits_finite"] for _, x in struct]
    return {
        "rows_complete": len(ok), "dialogues_complete": len(complete_d),
        "inventory_complete_fraction": inv_ok / len(rows), "structural_queries": len(struct),
        "structural_dialogues": len({dlg[u] for u, _ in struct}), "nonzero_gates": len(nonzero),
        "nonzero_gate_dialogues": len({dlg[u] for u, _ in nonzero}),
        "finite_provider_fraction": len(finite_provider) / len(complete_prefix) if complete_prefix else None,
        "identity_fraction": (sum(1 for r in ok.values() if r["full_replay"]["input_ids_sha256"] == S.prefix_hash(X.CB, r["baseline"]["content_ids"])
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
        "null_ZH_abs_error": runtime.get("null", {}).get("ZH_abs_error"),
        "branch_identity_fraction": (sum(br_ident) / len(br_ident)) if br_ident else None,
        "branch_finite_fraction": (sum(br_fin) / len(br_fin)) if br_fin else None,
        "D3_null_finite_fraction": (sum(null_fin) / len(null_fin)) if null_fin else None}


def job_a_predicates(cfg: dict, M: dict) -> dict:
    g, c = cfg["job_A_gate"], cfg["compatibility"]

    def ge(k, v):
        return M.get(k) is not None and M[k] >= v

    def le(k, v):
        return M.get(k) is not None and M[k] <= v
    cov = {"rows_complete": M["rows_complete"] == g["rows_complete"], "dialogues_complete": M["dialogues_complete"] == g["dialogues_complete"],
           "inventory_complete_fraction": M["inventory_complete_fraction"] == g["inventory_complete_fraction"],
           "structural_queries": ge("structural_queries", g["pulse_structural_queries_min"]),
           "structural_dialogues": ge("structural_dialogues", g["pulse_structural_dialogues_min"]),
           "nonzero_gates": ge("nonzero_gates", g["nonzero_old_gate_queries_min"]),
           "nonzero_gate_dialogues": ge("nonzero_gate_dialogues", g["nonzero_old_gate_dialogues_min"]),
           "finite_provider_fraction": ge("finite_provider_fraction", g["finite_provider_fraction_on_complete_prefix_min"]),
           "branch_identity_fraction": M["branch_identity_fraction"] == g["branch_identity_fraction"],
           "branch_finite_fraction": M["branch_finite_fraction"] == g["branch_finite_fraction"]}
    comp = {"identity_fraction": M["identity_fraction"] == c["input_and_query_identity_fraction"],
            "cached_baseline_identity_bitwise": M["cached_baseline_identity_bitwise"] is True,
            "raw_TV_median": le("raw_TV_median", c["full_vs_cached_raw_TV_median_max"]),
            "raw_TV_p99": le("raw_TV_p99", c["full_vs_cached_raw_TV_p99_max"]),
            "raw_TV_max": le("raw_TV_max", c["full_vs_cached_raw_TV_max"]),
            "PE_abs_diff_p99": le("PE_abs_diff_p99", c["full_vs_cached_raw_script_mass_abs_diff_p99_max"]),
            "PM_abs_diff_p99": le("PM_abs_diff_p99", c["full_vs_cached_raw_script_mass_abs_diff_p99_max"]),
            "processed_argmax_agreement": ge("processed_argmax_agreement", c["full_vs_cached_processed_argmax_agreement_min"]),
            "site_rel_L2_p99": le("site_rel_L2_p99", c["full_vs_cached_site_relative_L2_p99_max"]),
            "site_rel_L2_max": le("site_rel_L2_max", c["full_vs_cached_site_relative_L2_max"]),
            "site_cosine_min": ge("site_cosine_min", c["full_vs_cached_site_cosine_min"]),
            "causal_probe_TV_max": le("causal_probe_TV_max", c["prefix_only_vs_full_TV_max"]),
            "causal_probe_attention_L1_max": le("causal_probe_attention_L1_max", c["prefix_only_vs_full_attention_L1_max"]),
            "future_self_attention_mass_max": le("future_self_attention_mass_max", c["decoder_self_attention_future_mass_max"]),
            "null_EN_abs_error": le("null_EN_abs_error", cfg["gate"]["null_probability_abs_tolerance"]),
            "null_ZH_abs_error": le("null_ZH_abs_error", cfg["gate"]["null_probability_abs_tolerance"])}
    return {"job_A_coverage": cov, "compatibility": comp, "job_A_coverage_pass": predicate_all(cov),
            "compatibility_pass": predicate_all(comp)}


def family_construction(cfg: dict, M: dict, summ: dict | None) -> dict:
    """Reference-free family-wide construction status (PROVIDER_OR_CONSTRUCTION_BLOCKED or not)."""
    out = {}
    out["D3"] = {"blocked": not (M.get("D3_null_finite_fraction") is not None and M["D3_null_finite_fraction"] >= cfg["job_A_gate"]["D3_null_finite_fraction_min"]),
                 "reasons": []}
    if out["D3"]["blocked"]:
        out["D3"]["reasons"].append("null_branch_finite_fraction")
    out["D4"] = {"blocked": False, "reasons": []}
    if summ is None:
        out["D5"] = {"blocked": True, "reasons": ["construct_missing"]}
    else:
        out["D5"] = {"blocked": not summ["D5_constructible"], "reasons": list(summ["D5_blocked_reasons"])}
    return out


def cmd_seal_a(args) -> None:
    cfg = load_cfg()
    m = json.loads((ROOT / MANIFEST).read_text())
    rows, bank = X.validate_runtime_panel(json.loads((ROOT / PANEL).read_text()))
    out = ROOT / A_SEAL
    if out.exists():
        raise FileExistsError("Job A seal exists")
    paths = run_paths()
    runtime = json.loads((paths.root / "capture_runtime.json").read_text())
    cap, files, archive = {}, {}, {}

    def add(rel: str, big: bool = False) -> str:
        p = paths.root / rel
        h = file_hash(p)
        files[rel] = h
        if big:
            archive[rel] = archive_copy(p, h)
        return h
    for r in rows:
        uid = r["utterance_id"]
        p = paths.capture / f"{uid}.json"
        if not p.exists():
            continue
        g = json.loads(p.read_text())
        if g["identity"] != uid or g["manifest_hash"] != m["manifest_hash"]:
            raise ValueError(f"capture row identity {uid}")
        cap[uid] = g
        add(f"capture/{uid}.json")
        if g["status"] == "ok":
            if add(f"capture/{uid}.npz", True) != g["arrays"]["sha256"]:
                raise ValueError(f"capture arrays changed {uid}")
    for r in bank:
        uid = r["utterance_id"]
        if (paths.bank / f"{uid}.json").exists():
            b = json.loads((paths.bank / f"{uid}.json").read_text())
            add(f"bank/{uid}.json")
            if b["status"] == "ok" and add(f"bank/{uid}.npz", True) != b["arrays"]["sha256"]:
                raise ValueError(f"bank arrays changed {uid}")
    for r in list(rows) + list(bank):
        uid = r["utterance_id"]
        if (paths.phones / f"{uid}.json").exists():
            ph = json.loads((paths.phones / f"{uid}.json").read_text())
            add(f"phones/{uid}.json")
            if ph.get("status") == "ok" and add(f"phones/{uid}.npz", True) != ph["arrays"]["sha256"]:
                raise ValueError(f"phone arrays changed {uid}")
    for p in sorted(paths.construct.glob("*")):
        add(f"construct/{p.name}", p.suffix == ".npz")
    for name in ("capture_runtime.json", "phones_runtime_prospective.json", "phones_runtime_bank.json",
                 "phones_prospective.log", "phones_bank.log"):
        if (paths.root / name).exists():
            add(name)
    for extra in sorted(paths.root.glob("slurm-A-*")):
        add(extra.name)
    for extra in sorted(paths.root.glob("construct_*")):
        if extra.is_file():
            add(extra.name)
    M = job_a_metrics(cfg, rows, cap, runtime)
    preds = job_a_predicates(cfg, M)
    summ = json.loads((paths.construct / "construct_summary.json").read_text()) if (paths.construct / "construct_summary.json").exists() else None
    fam = family_construction(cfg, M, summ)
    critical = {"runtime_completed": runtime.get("status") == "completed", "weights_unchanged": runtime.get("weights_unchanged") is True,
                "no_parameter_grads": runtime.get("parameters_with_grad") == 0 and runtime.get("parameters_requiring_grad") == 0,
                "construct_completed": summ is not None, "phone_children_exited": all(v == 0 for v in runtime.get("phone_children_returncodes", {}).values())
                or fam["D5"]["blocked"]}
    seal = {"schema": SCHEMA, "phase": "A", "status": "SEALED", "manifest_hash": m["manifest_hash"], "config_sha256": CONFIG_SHA,
            "runtime_hash": m["runtime_hash"], "files": files, "archive": {"root": str(ARCHIVE), "content_addressed": archive},
            "counts": {"rows": len(cap), "ok": sum(g["status"] == "ok" for g in cap.values()),
                       "queries": sum(len(g.get("queries", [])) for g in cap.values())},
            "metrics": M, "predicates": preds, "family_construction": fam, "construct_summary": summ, "critical": critical,
            "git_head_at_seal": git("rev-parse", "HEAD"), "references_used": False, "created_unix": time.time()}
    seal["seal_hash"] = digest(seal)
    atomic_json(out, seal)
    print(json.dumps({"predicates": {k: v for k, v in preds.items() if k.endswith("_pass")}, "critical": critical,
                      "family_construction": fam, "metrics": M}, indent=1))


# ---- authorization ------------------------------------------------------------------------------------------------

def sealed_rows(seal: dict, sub: str, ids: list[str]) -> dict:
    out = {}
    for uid in ids:
        rel = f"{sub}/{uid}.json"
        if rel not in seal["files"]:
            continue
        p = ROOT / RUN / rel
        if file_hash(p) != seal["files"][rel]:
            raise ValueError(f"sealed row changed {rel}")
        out[uid] = json.loads(p.read_text())
    return out


def planned_pulses(cfg: dict, cap: dict, cons: dict, fam: dict) -> tuple[int, int, int]:
    """Reference-free planned matrix: (structural queries, planned nonzero pulses, planned autograd calls)."""
    n_struct = n_pulse = n_auto = 0
    for uid, r in cap.items():
        if r.get("status") != "ok":
            continue
        c = cons[uid]
        st = {s["t"]: s for s in c["status"]}
        for qd in r["queries"]:
            if not qd["structural"]["eligible"]:
                continue
            n_struct += 1
            s = st[qd["t"]]
            g = float(qd["g"])
            d3c = qd["D3"]["status"] == "candidate" and not fam["D3"]["blocked"]
            n_auto += 1 + int(d3c)
            valid = {"D0": s["D0"]["status"] == "ok", "D1": True, "D2": True, "VAC": s["VAC"]["status"] == "OK",
                     "RND": True, "D3": d3c, "D4": s["D4"]["status"] == "ok" and not fam["D4"]["blocked"],
                     "D5": s["D5"]["status"] == "ok" and not fam["D5"]["blocked"],
                     "D5SH": s["D5SH"]["status"] == "ok" and not fam["D5"]["blocked"]}
            for a in X.PULSE_ARMS:
                base = X.GATED_BASE.get(a, a)
                if valid[base] and X.arm_target(a, 1.0, g) > 0:
                    n_pulse += 1
    return n_struct, n_pulse, n_auto


def resource_forecast(cfg: dict, cap: dict, cons: dict, fam: dict, runtime: dict) -> dict:
    c = cfg["compute"]
    ok = [g for g in cap.values() if g.get("status") == "ok"]
    n, npulse, nauto = planned_pulses(cfg, cap, cons, fam)
    cached = [s for g in ok for s in g["timing"]["cached_query_sec"]]
    enc = [g["encoder"]["encoder_sec"] for g in ok]
    wbytes = sum(g["arrays"]["bytes"] for g in ok)
    wsec = sum(g["arrays"]["write_sec"] for g in ok)
    io = sum(g["audio"]["io_sec"] + g["encoder"]["features_sec"] for g in ok)
    inputs = {"structural_queries": n, "planned_nonzero_pulses": npulse, "planned_autograd_calls": nauto,
              "A_p95_cached_query_seconds": q(cached, .95),
              "A_durable_archive_bytes_per_second": (wbytes / wsec) if wsec > 0 else None,
              "A_model_load_seconds": runtime.get("model_load_sec"), "A_p95_original_encoder_seconds": q(enc, .95),
              "A_total_audio_IO_seconds": io}
    out = {"schema": SCHEMA, "inputs": inputs, "formula_forecast_seconds": 0.18 * n + 0.05 * npulse + 800}
    valid = all(isinstance(v, (int, float)) and v > 0 for v in inputs.values())
    if valid:
        cq = inputs["A_p95_cached_query_seconds"]
        v = inputs["A_durable_archive_bytes_per_second"]
        per_pulse = c["historical_pulse_p95_sec"] + c["per_pulse_solver_allowance_sec"] + c["per_pulse_bytes"] / v
        variable = n * 2 * cq + nauto * c["historical_D2_p95_sec"] + npulse * per_pulse
        fixed = max(800, inputs["A_model_load_seconds"] + 240 * inputs["A_p95_original_encoder_seconds"]
                    + inputs["A_total_audio_IO_seconds"] + 180)
        out.update(observed_component_variable=variable, observed_component_fixed=fixed,
                   observed_component_forecast_seconds=variable + fixed)
    else:
        out["observed_component_forecast_seconds"] = None
    lim = c["pre_job_B_forecast_max_seconds"]
    out["metrics"] = {"formula_forecast_seconds": out["formula_forecast_seconds"],
                      "observed_component_forecast_seconds": out["observed_component_forecast_seconds"],
                      "scientific_jobs_planned": 2, "walltime_per_job_seconds": c["seconds_max_each"]}
    out["resource_pass"] = bool(out["formula_forecast_seconds"] <= lim and out["observed_component_forecast_seconds"] is not None
                                and out["observed_component_forecast_seconds"] <= lim)
    return out


def cmd_authorize_b(args) -> None:
    cfg = load_cfg()
    for x in (A_SEAL, AUDIT_A):
        if not (committed(x) and on_remote(x)):
            raise SystemExit(f"{x} must be committed and pushed")
    if (ROOT / AUTH).exists() or (ROOT / FORECAST).exists():
        raise FileExistsError("authorization exists")
    seal = json.loads((ROOT / A_SEAL).read_text())
    rows, _ = X.validate_runtime_panel(json.loads((ROOT / PANEL).read_text()))
    ids = [r["utterance_id"] for r in rows]
    cap = sealed_rows(seal, "capture", ids)
    cons = sealed_rows(seal, "construct", ids)
    a = json.loads((ROOT / AUDIT_A).read_text())
    runtime = json.loads((ROOT / RUN / "capture_runtime.json").read_text())
    fc = resource_forecast(cfg, cap, cons, seal["family_construction"], runtime)
    atomic_json(ROOT / FORECAST, fc)
    fam = seal["family_construction"]
    authorized = bool(a.get("verdict") == "DIR_SPRINT0_AUDIT_A: PASS" and seal["predicates"]["job_A_coverage_pass"]
                      and seal["predicates"]["compatibility_pass"] and all(seal["critical"].values())
                      and not all(fam[f]["blocked"] for f in X.NEW_FAMILIES) and fc["resource_pass"])
    auth = {"schema": "dir_sprint0_authorization_B_v1", "job_A_seal_hash": file_hash(ROOT / A_SEAL),
            "construct_seal_hash": seal["seal_hash"], "config_hash": CONFIG_SHA,
            "population_hash": cfg["population"]["selected_ids_hash"], "audit_hash": file_hash(ROOT / AUDIT_A),
            "resource_forecast_hash": file_hash(ROOT / FORECAST), "authorized": authorized}
    atomic_json(ROOT / AUTH, auth)
    print(json.dumps({"authorized": authorized, "forecast": fc["metrics"], "resource_pass": fc["resource_pass"],
                      "inputs": fc["inputs"]}))


# ---- Job B: pulses ----------------------------------------------------------------------------------------------------

class Engine:
    """Owner-scoped cached steps with the passive native observer, plus the one-solve pulse executor (SRD2-G0 Engine,
    unchanged arithmetic)."""

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
    """Historical engineering reproduction on the audited P2-DIR spot set (outside every new denominator); the SRD2-G0
    apparatus check unchanged."""
    import torch
    import experiments.inference_cf_p2r as p2r
    from csasr.inference_cf.core_p1 import processed_argmax
    from csasr.inference_cf.directions import DirectionContext, ReadoutDirection
    from csasr.inference_cf.loc0_sites import cross_pulse_hook
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
        enc, _ = encode_path(ctx.bundle, audio[uid])
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
                B.crop(L)
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


def pulse_utterance(eng: Engine, row: dict, crow: dict, carr: dict, cons: dict, carrs: dict, out_dir: Path, m: dict,
                    blocked: dict) -> dict:
    import torch
    import experiments.inference_cf_p2r as p2r
    from csasr.inference_cf.core_p1 import processed_argmax
    from csasr.inference_cf.directions import DirectionContext, ReadoutDirection
    ctx = eng.ctx
    uid = row["utterance_id"]
    if file_hash(row["audio_path"]) != row["audio_full_sha256"]:
        raise ValueError("audio bytes changed")
    OPENED.add(os.path.abspath(row["audio_path"]))
    enc, enc_t = encode_path(ctx.bundle, row["audio_path"])
    if enc_t["encoder_sha256"] != crow["encoder"]["encoder_sha256"]:
        raise RuntimeError("encoder output differs from Job A")
    content = crow["baseline"]["content_ids"]
    inv = crow["inventory"]
    struct_ts = cons["structural_ts"]
    if struct_ts != crow["structural_ts"]:
        raise RuntimeError("construct structural queries differ from capture")
    kidx = {t: k for k, t in enumerate(struct_ts)}
    d2p = ReadoutDirection(ctx.bundle, layer=ctx.layer, suppress=ctx.suppress, begin=ctx.begin, partition=ctx.partition)
    B = p2r.DiagBranch(ctx.bundle, enc, list(ctx.prompt), "B")
    queries, d2dirs, d2grads, d3dirs, d3grads, natq, natu, natr = [], [], [], [], [], [], [], []
    ex_keys, ex_logits, ex_ffn = [], [], []
    D = int(ctx.bundle.d_model)
    nanv = np.full(D, np.nan, np.float32)
    for i, t in enumerate(inv):
        qa = crow["queries"][i]
        if qa["t"] != t:
            raise RuntimeError("Job A query order")
        new = S.feed_tokens(content, t, ctx.prompt)
        L = B.length
        query = L + len(new) - 1
        if query != qa["query"]:
            raise RuntimeError("absolute query mismatch")
        structural = qa["structural"]["eligible"]
        d2 = d3 = None
        d3_wanted = structural and qa["D3"]["status"] == "candidate" and not blocked["D3"]
        if structural:
            d2 = d2p(DirectionContext(b_cache_pre_step=B.cache, encoded=enc, new_tokens=list(new), start=L, step=t))
            if d3_wanted:
                d3 = X.token_margin_direction(ctx.bundle, layer=ctx.layer, cache=B.cache, encoded=enc, new_tokens=list(new),
                                              start=L, c_token=int(qa["D3"]["c_AP"]), b_token=int(qa["expected_action"]))
        logits0, obs0, clean_sec = eng.step(B, new)
        top0 = processed_argmax(logits0, t, ctx.suppress, ctx.begin)
        rec = {"t": t, "query": query, "structural": structural,
               "clean": {"logits_bitwise_vs_A": bool(np.array_equal(S.bf16_bits(logits0), carr["cached_logits"][i])),
                         "site_bitwise_vs_A": bool(np.array_equal(S.bf16_bits(obs0.r[0].float()), carr["cached_site"][i])),
                         "top1": top0, "expected_action": qa["expected_action"], "sec": clean_sec},
               "arms": {"B0": {"status": "none", "executed": False, "top1": top0, "logits": "B0"}}}
        if top0 != qa["expected_action"]:
            raise RuntimeError("clean argmax differs from the sealed baseline action")
        if structural:
            pr = d2.provenance
            rec["d2"] = {"status": d2.status, "reason": d2.reason, "counters": d2.counters, "runtime_sec": d2.extras["runtime_sec"],
                         **{k: pr.get(k) for k in ("J", "log_PE", "log_PM", "g_norm", "h_norm", "tangent_norm", "g_radial",
                                                   "unit_error", "radial_dot")},
                         "direction_sha256": None if d2.direction is None else S.array_sha(d2.direction.numpy()),
                         "scratch_logits_bitwise": bool(torch.equal(d2.extras["logits"], logits0)),
                         "scratch_site_bitwise": bool(torch.equal(d2.extras["site"], obs0.r[0].float().cpu()))}
            d2dirs.append(nanv if d2.direction is None else d2.direction.numpy())
            g = d2.extras.get("gradient")
            d2grads.append(nanv if g is None else g.numpy())
            if d3 is not None:
                j3 = d3.get("J")
                rec["d3"] = {"status": d3["status"], "reason": d3["reason"], "J": j3 if j3 is not None and math.isfinite(j3) else None,
                             "c_token": d3["c_token"],
                             "b_token": d3["b_token"], "counters": d3["counters"], "runtime_sec": d3["runtime_sec"],
                             **{k: d3.get(k) for k in ("g_norm", "h_norm", "tangent_norm", "g_radial", "unit_error", "radial_dot")},
                             "direction_sha256": None if d3["direction"] is None else S.array_sha(d3["direction"].numpy()),
                             "scratch_logits_bitwise": bool(torch.equal(d3["logits"], logits0)),
                             "scratch_site_bitwise": bool(torch.equal(d3["site"], obs0.r[0].float().cpu()))}
                d3dirs.append(nanv if d3["direction"] is None else d3["direction"].numpy())
                d3grads.append(nanv if d3.get("gradient") is None else d3["gradient"].numpy())
            else:
                rec["d3"] = {"status": "no_edit:" + (qa["D3"]["status"] if not blocked["D3"] else "family_blocked")}
                d3dirs.append(nanv)
                d3grads.append(nanv)
            natq.append(S.bf16_bits(obs0.q[0].float()))
            natu.append(S.bf16_bits(obs0.u[0].float()))
            natr.append(S.bf16_bits(obs0.r[0].float()))
            k = kidx[t]
            g_old = float(qa["g"])

            def vec(fam: str):
                if fam == "D2":
                    return d2.direction
                if fam == "D3":
                    return None if d3 is None else d3["direction"]
                if fam in ("D4",) and blocked["D4"]:
                    return None
                if fam in ("D5", "D5SH") and blocked["D5"]:
                    return None
                a = carrs[fam][k]
                return None if np.isnan(a).any() else torch.from_numpy(a.astype(np.float32))
            executed = False
            for a in X.PULSE_ARMS:
                fam = X.GATED_BASE.get(a, a)
                target = X.arm_target(a, ctx.e_star, g_old)
                ar, la, ffn = eng.arm(B, L, new, t, query, vec(fam), target, obs0)
                if ar["executed"]:
                    executed = True
                    ex_keys.append([i, X.PULSE_ARMS.index(a)])
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
            rec["g_old"] = g_old
        else:
            for a in X.PULSE_ARMS:
                rec["arms"][a] = {"target": 0.0, "status": "structurally_ineligible", "executed": False, "top1": top0, "logits": "B0"}
        queries.append(rec)
    V = int(carr["cached_logits"].shape[1])
    arrays = {"d2_direction": np.stack(d2dirs) if d2dirs else np.zeros((0, D), np.float32),
              "d2_gradient": np.stack(d2grads) if d2grads else np.zeros((0, D), np.float32),
              "d3_direction": np.stack(d3dirs) if d3dirs else np.zeros((0, D), np.float32),
              "d3_gradient": np.stack(d3grads) if d3grads else np.zeros((0, D), np.float32),
              "native_q": np.stack(natq) if natq else np.zeros((0, D), np.int16),
              "native_u": np.stack(natu) if natu else np.zeros((0, D), np.int16),
              "native_r": np.stack(natr) if natr else np.zeros((0, D), np.int16),
              "exec_keys": np.asarray(ex_keys, dtype=np.int64).reshape(-1, 2),
              "exec_ffn": np.stack(ex_ffn) if ex_ffn else np.zeros((0, D), np.int16)}
    logit_arrays = {"exec_keys": arrays["exec_keys"], "exec_logits": np.stack(ex_logits) if ex_logits else np.zeros((0, V), np.int16)}
    h1, b1, w1 = write_npz(out_dir / f"{uid}.npz", arrays)
    h2, b2, w2 = write_npz(out_dir / f"{uid}_logits.npz", logit_arrays)
    return {"schema": SCHEMA, "phase": "B", "identity": uid, "utterance_id": uid, "canonical_index": row["canonical_index"],
            "manifest_hash": m["manifest_hash"], "status": "ok", "encoder": enc_t, "inventory": inv,
            "structural_ts": [x["t"] for x in queries if x["structural"]], "arm_order": list(X.PULSE_ARMS), "queries": queries,
            "arrays": {"file": f"{uid}.npz", "sha256": h1, "bytes": b1, "write_sec": w1},
            "logits": {"file": f"{uid}_logits.npz", "sha256": h2, "bytes": b2, "write_sec": w2},
            "capture_row_digest": digest(crow), "construct_row_digest": digest(cons)}


def cmd_pulses(args, smoke: tuple | None = None) -> None:
    import torch
    sys.addaudithook(_audit_open)
    if smoke is None:
        m, cfg, rows, _ = load_run("pulses")
        paths = run_paths()
        seal = json.loads((ROOT / A_SEAL).read_text())
        ids = [r["utterance_id"] for r in rows]
        cap = sealed_rows(seal, "capture", ids)
        cons = sealed_rows(seal, "construct", ids)
        fam = seal["family_construction"]
        do_apparatus = True
    else:
        m, cfg, rows, paths, cap, cons, fam = smoke
        do_apparatus = False
    blocked = {f: bool(fam[f]["blocked"]) for f in X.NEW_FAMILIES}
    out = paths.pulses
    if out.exists():
        raise FileExistsError("pulses directory exists; a pulse attempt is never overwritten")
    out.mkdir(parents=True)
    bundle, load_sec = load_model(cfg, smoke=smoke is not None)
    ctx = Ctx(bundle, cfg)
    eng = Engine(ctx)
    w0 = weights_digest(bundle)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rt = {"schema": SCHEMA, "phase": "B", "job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-smoke", "manifest_hash": m["manifest_hash"],
          "authorization_sha256": file_hash(ROOT / AUTH) if smoke is None else None, "start_unix": time.time(), "status": "running",
          "model_load_sec": load_sec, "weights_digest_start": w0, "families_blocked": blocked}
    atomic_json(paths.root / "pulse_runtime.json", rt)
    if do_apparatus:
        t0 = time.time()
        with torch.inference_mode():
            app = apparatus_old30(eng, cfg)
        app["elapsed_sec"] = time.time() - t0
        atomic_json(paths.root / "apparatus_old30.json", app)
        rt["apparatus_ok"] = app["ok"]
        print(f"old30 apparatus ok={app['ok']} n={app['n']} {app['elapsed_sec']:.1f}s", flush=True)
        if not app["ok"]:
            rt.update(status="apparatus_failed", end_unix=time.time(), opened_paths=sorted(OPENED))
            atomic_json(paths.root / "pulse_runtime.json", rt)
            raise SystemExit("historical apparatus reproduction failed: INVALID, no new pulses")
    failures = 0
    autograd = {"D2": 0, "D3": 0}
    for row in rows:
        uid = row["utterance_id"]
        ts = time.time()
        try:
            cr = cap[uid]
            if cr["status"] != "ok":
                raise RuntimeError("Job A row not ok")
            cp = paths.capture / cr["arrays"]["file"]
            if file_hash(cp) != cr["arrays"]["sha256"]:
                raise ValueError("sealed capture arrays changed")
            with np.load(cp) as z:
                carr = {k: z[k] for k in ("cached_logits", "cached_site")}
            co = cons[uid]
            dp = paths.construct / co["arrays"]["file"]
            if file_hash(dp) != co["arrays"]["sha256"]:
                raise ValueError("sealed construct arrays changed")
            with np.load(dp) as z:
                carrs = {k: z[k] for k in FAMILY_ARRAYS}
            with torch.inference_mode():
                res = pulse_utterance(eng, row, cr, carr, co, carrs, out, m, blocked)
            for x in res["queries"]:
                autograd["D2"] += x.get("d2", {}).get("counters", {}).get("autograd_calls", 0)
                autograd["D3"] += x.get("d3", {}).get("counters", {}).get("autograd_calls", 0)
        except Exception as exc:
            import traceback
            failures += 1
            res = {"schema": SCHEMA, "phase": "B", "identity": uid, "utterance_id": uid, "canonical_index": row["canonical_index"],
                   "manifest_hash": m["manifest_hash"], "status": "failure", "reason": repr(exc), "traceback": traceback.format_exc()}
        res["elapsed_sec"] = time.time() - ts
        atomic_json(out / f"{uid}.json", res)
        print(f"B {row['canonical_index'] + 1}/{len(rows)} {uid} {res['status']} {res['elapsed_sec']:.1f}s", flush=True)
    S.assert_no_hooks_anywhere(bundle)
    rt.update(weights_digest_end=weights_digest(bundle), end_unix=time.time(), failures=failures, autograd_calls=autograd,
              status="completed" if failures == 0 else "failed",
              peak_vram_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
              peak_vram_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
              parameters_with_grad=sum(p.grad is not None for p in bundle.model.parameters()),
              parameters_requiring_grad=sum(p.requires_grad for p in bundle.model.parameters()),
              opened_paths=sorted(OPENED))
    rt["elapsed_sec"] = rt["end_unix"] - rt["start_unix"]
    rt["weights_unchanged"] = rt["weights_digest_end"] == w0
    atomic_json(paths.root / "pulse_runtime.json", rt)
    if failures:
        raise SystemExit(f"{failures} utterance failures")


def cmd_seal_b(args) -> None:
    m = json.loads((ROOT / MANIFEST).read_text())
    rows, _ = X.validate_runtime_panel(json.loads((ROOT / PANEL).read_text()))
    out = ROOT / PULSE_SEAL
    if out.exists():
        raise FileExistsError("pulse seal exists")
    paths = run_paths()
    files, archive = {}, {}
    n_ok = 0
    for r in rows:
        uid = r["utterance_id"]
        p = paths.pulses / f"{uid}.json"
        if not p.exists():
            continue
        x = json.loads(p.read_text())
        if x["identity"] != uid or x["manifest_hash"] != m["manifest_hash"]:
            raise ValueError(f"pulse row identity {uid}")
        files[f"pulses/{uid}.json"] = file_hash(p)
        if x["status"] == "ok":
            n_ok += 1
            for key in ("arrays", "logits"):
                f = paths.pulses / x[key]["file"]
                h = file_hash(f)
                if h != x[key]["sha256"]:
                    raise ValueError(f"pulse arrays changed {uid}")
                files[f"pulses/{x[key]['file']}"] = h
                archive[f"pulses/{x[key]['file']}"] = archive_copy(f, h)
    for name in ("pulse_runtime.json", "apparatus_old30.json"):
        files[name] = file_hash(paths.root / name)
    for extra in sorted(paths.root.glob("slurm-B-*")):
        files[extra.name] = file_hash(extra)
    rt = json.loads((paths.root / "pulse_runtime.json").read_text())
    seal = {"schema": SCHEMA, "phase": "B", "status": "SEALED", "manifest_hash": m["manifest_hash"], "config_sha256": CONFIG_SHA,
            "job_A_seal_sha256": file_hash(ROOT / A_SEAL), "authorization_sha256": file_hash(ROOT / AUTH), "files": files,
            "archive": {"root": str(ARCHIVE), "content_addressed": archive},
            "counts": {"rows": sum(1 for k in files if k.startswith("pulses/") and k.endswith(".json")), "ok": n_ok},
            "runtime_status": rt.get("status"), "git_head_at_seal": git("rev-parse", "HEAD"), "references_used": False,
            "created_unix": time.time()}
    seal["seal_hash"] = digest(seal)
    atomic_json(out, seal)
    print(json.dumps({"rows": seal["counts"], "runtime_status": seal["runtime_status"]}))


# ---- CPU engineering smoke (public engineering audio only; scratch root) ------------------------------------------------

def cmd_smoke(args) -> None:
    """End-to-end engineering smoke on CPU (BF16) with public engineering clips posing as prospective / bank rows.
    Writes only under ``--root``; no barrier, no D-dev-select audio, no outcome."""
    import soundfile as sf
    cfg = load_cfg()
    root = Path(args.root)
    if root.exists():
        raise FileExistsError("smoke root exists")
    root.mkdir(parents=True)
    pm = json.loads((ROOT / PROVIDER_MANIFEST).read_text())
    adir = Path(pm["engineering_audio"]["dir"])
    clips = [c for c in ("aishell_example_mandarin.wav", "sample1.flac") if pm["engineering_audio"]["files"][c]["sample_rate"] == 16000]
    # sample1 is ~11 s; crop it to 4 s into the scratch root to bound CPU time
    x, sr = sf.read(str(adir / "sample1.flac"), dtype="float32")
    crop = root / "sample1_4s.wav"
    sf.write(str(crop), x[:64000], sr, subtype="FLOAT")
    paths_ = [str(adir / clips[0]), str(crop)]
    rows = [{"canonical_index": i, "utterance_id": f"SMOKE_P{i}", "audio_path": p, "audio_full_sha256": file_hash(p)} for i, p in enumerate(paths_)]
    bank = [{"bank_index": i, "utterance_id": f"SMOKE_B{i}", "audio_path": p, "audio_full_sha256": file_hash(p)} for i, p in enumerate(paths_)]
    panel = {"schema": X.PANEL_SCHEMA, "rows": rows, "bank_rows": bank,
             "population_selected_ids_hash": digest([r["utterance_id"] for r in rows]),
             "bank_ids_hash": digest(sorted(r["utterance_id"] for r in bank))}
    panel["runtime_hash"] = digest(panel)
    X.validate_runtime_panel(panel)
    atomic_json(root / "runtime_panel.json", panel)
    pop = {"selected": [{"utterance_id": r["utterance_id"], "dialogue_id": "CSD0006"} for r in rows],
           "calibration_bank": {"rows": [{"utterance_id": r["utterance_id"], "dialogue_id": "CSD0006"} for r in bank]}}
    atomic_json(root / "smoke_population.json", pop)
    paths = Paths(root)
    t0 = time.time()
    cmd_capture(args, smoke_rows=(rows, bank), smoke_paths=paths)
    print(f"smoke capture {time.time() - t0:.0f}s", flush=True)
    cap = {r["utterance_id"]: json.loads((paths.capture / f"{r['utterance_id']}.json").read_text()) for r in rows}
    if not (paths.construct / "construct_summary.json").exists():
        raise SystemExit("smoke construct failed: " + (root / "construct_error.json").read_text())
    cons = {r["utterance_id"]: json.loads((paths.construct / f"{r['utterance_id']}.json").read_text()) for r in rows}
    fam = {"D3": {"blocked": False}, "D4": {"blocked": False}, "D5": {"blocked": False}}
    t0 = time.time()
    cmd_pulses(args, smoke=({"manifest_hash": "smoke"}, cfg, rows, paths, cap, cons, fam))
    print(f"smoke pulses {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prepare", "manifest", "capture", "phones", "construct", "seal-a", "authorize-b", "pulses",
                                     "seal-b", "smoke"))
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--out", default=RUN)
    ap.add_argument("--which", choices=("prospective", "bank"))
    ap.add_argument("--root", default=str(ROOT / RUN))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.config != CONFIG or args.out.rstrip("/") != RUN:
        raise SystemExit(f"only the frozen config {CONFIG} and run root {RUN} are allowed")
    if args.mode == "smoke" and Path(args.root).resolve() == (ROOT / RUN).resolve():
        raise SystemExit("smoke must use a scratch --root, never the frozen run root")
    {"prepare": cmd_prepare, "manifest": cmd_manifest, "capture": cmd_capture, "phones": cmd_phones, "construct": cmd_construct,
     "seal-a": cmd_seal_a, "authorize-b": cmd_authorize_b, "pulses": cmd_pulses, "seal-b": cmd_seal_b,
     "smoke": cmd_smoke}[args.mode](args)


if __name__ == "__main__":
    main()
