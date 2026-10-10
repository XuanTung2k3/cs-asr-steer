#!/usr/bin/env python
"""ST-PROMPT-R1-A runner — decoder layer x relative dose x sign single-pulse causal screen
(docs/inference_cf/ST_PROMPT_R1_SPEC.md, _CODEX_DESIGN.md, _FIREWALL.md, _PANEL.json, configs/inference_cf/st_prompt_r1.json;
freeze e965a7f).

prepare   (CPU) freeze / panel / source / model / tokenizer / audio / historical-comparator pins; plan_sealed.json holds the
          RUNTIME projection only (runtime_queries + utterances), the 36-arm table and sealed apparatus vectors.
manifest  (CPU) resolved manifest after a committed PASS_TO_ST_PROMPT_R1_A.
capture   (GPU) passive paired forced-ZH / forced-EN replay of the exact 180 B0M_L16 prefixes; DG-02 states at L3/8/16/24;
          L16 states + NONE logits bitwise vs the sealed ST-LOC0 calibration; v_prompt = core_p1.direction(h_E, h_M) per
          query/layer (L16 hash-identical to historical D0); CPU geometry ledger for all 36 dose cells/query (no logits).
pulses    (GPU) requires the separate-process geometry gate PASS. Barrier pass on all 180: zero dose at all four layers,
          ST-LOC0 L16 v_prompt +/- and D2 at the historical absolute e* -> bitwise vs sealed ST-LOC0 logits / consumed
          states, solver scale <= 1e-9; any mismatch aborts before new pulses. Main pass: 24 primary + 12 random single
          relative-dose pulses per query, each from the cropped pristine pre-query cache.
seal      (CPU) immutable reference-free output seal.
The runner never reads strata, reference targets, competitors or evaluator outputs.
"""
from __future__ import annotations

import argparse
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

SCHEMA = "st_prompt_r1_v1"
CONFIG = "configs/inference_cf/st_prompt_r1.json"
PANEL = "docs/inference_cf/ST_PROMPT_R1_PANEL.json"
FROZEN = (CONFIG, PANEL, "docs/inference_cf/ST_PROMPT_R1_SPEC.md", "docs/inference_cf/ST_PROMPT_R1_CODEX_DESIGN.md",
          "docs/inference_cf/ST_PROMPT_R1_FIREWALL.md", "docs/inference_cf/ST_PROMPT_R1_REACHABILITY.json",
          "tests/test_st_prompt_r1_contract.py", "experiments/inference_cf_st_prompt_r1_preflight.py")
BASE = "results/inference_cf/st_prompt_r1"
PLAN = f"{BASE}/plan_sealed.json"
LOC0 = "results/inference_cf/st_loc0/run1"
LOC0_SEAL = "results/inference_cf/st_loc0/output_seal.json"
EXP1 = "results/inference_cf/p2dir/exp1_run1"
LAYERS = (3, 8, 16, 24)
CB = [50258, 50260, 50360, 50364]
CE = [50258, 50259, 50360, 50364]
# historical ST-LOC0 cells reproduced in the barrier (arm id -> (ST-LOC0 pass dir, ST-LOC0 arm id, family, sign))
BARRIER = {"hist_prompt_L16_plus": ("barrier", "v_prompt_L16_POST_CROSS_ATTENTION_PRE_FFN_plus", "prompt", 1),
           "hist_prompt_L16_minus": ("pulses", "v_prompt_L16_POST_CROSS_ATTENTION_PRE_FFN_minus", "prompt", -1),
           "hist_D2_L16": ("barrier", "D2_L16_POST_CROSS_ATTENTION_PRE_FFN", "D2", 1)}


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def committed(rel: str) -> bool:
    import hashlib
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return git("ls-files", rel) == rel and "sha256:" + hashlib.sha256(blob).hexdigest() == file_hash(ROOT / rel)


def cmd_prepare(args) -> None:
    import hashlib
    import inspect
    import transformers
    from transformers import GenerationConfig, WhisperProcessor
    from transformers.models.whisper.modeling_whisper import WhisperDecoderLayer
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.inference_cf.prompt_r1 import arm_table, random_vectors, runtime_projection
    import experiments.inference_cf_p2tta0 as t0run
    cfg = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    mdir = cfg["conditions"]["model"]["dir"]
    checks = {"frozen_committed": all(committed(p) for p in FROZEN),
              "panel_identity": digest({k: v for k, v in P.items() if k != "identity_hash"}) == P["identity_hash"] == cfg["panel_identity_hash"],
              "sources": all(file_hash(ROOT / p) == h for p, h in cfg["source_sha256"].items()),
              "model_files": all(file_hash(Path(mdir) / n) == h for n, h in cfg["conditions"]["model"]["files"].items())}
    tok = WhisperProcessor.from_pretrained(mdir, local_files_only=True).tokenizer
    gen = GenerationConfig.from_pretrained(mdir, local_files_only=True)
    checks["partition"] = tokenizer_partition(tok)["hash"] == cfg["conditions"]["partition_hash"]
    checks["suppression"] = digest({"suppress": list(gen.suppress_tokens or []), "begin": list(gen.begin_suppress_tokens or [])}) == cfg["conditions"]["suppression_hash"]
    rp = runtime_projection(P)
    by_u = {u["utterance_id"]: u for u in rp["utterances"]}
    ok = len(rp["runtime_queries"]) == 180 and len(rp["utterances"]) == 80
    for q in rp["runtime_queries"]:
        u = by_u[q["utterance_id"]]
        b = json.loads((ROOT / u["baseline_row"]).read_text())["systems"][cfg["conditions"]["baseline_system"]]
        prefix = b["tokens"][:q["t"]]
        ok &= (u["baseline_content_tokens"] == b["tokens"] and q["absolute_query"] == 4 + q["t"] - 1 and q["t"] >= 1
               and digest(prefix) == q["content_prefix_sha256"] and digest(CB + prefix) == q["forced_zh_query_input_sha256"]
               and digest(CE + prefix) == q["forced_en_query_input_sha256"])
    checks["runtime_queries"] = bool(ok)
    checks["audio"] = all(t0run.audio_fingerprint_64k(u["audio_path"]) == u["audio_sha256"] or t0run.audio_full_sha256(u["audio_path"]) == u["audio_sha256"]
                          for u in rp["utterances"])
    checks["random_vectors"] = len(random_vectors(cfg)) == 12
    arms = arm_table(cfg)
    checks["arm_table_36"] = len(arms) == 36 and len({a["id"] for a in arms}) == 36
    checks["site_forward"] = (transformers.__version__ == cfg["site_forward"]["transformers_version"]
                              and "sha256:" + hashlib.sha256(inspect.getsource(WhisperDecoderLayer.forward).encode()).hexdigest() == cfg["site_forward"]["forward_source_sha256"]
                              and file_hash(inspect.getfile(WhisperDecoderLayer)) == cfg["site_forward"]["module_file_sha256"])
    seal = json.loads((ROOT / LOC0_SEAL).read_text())
    need = [f"{LOC0}/calibration/states_L16_CROSS.npz", f"{LOC0}/calibration/baseline_logits.npz", f"{LOC0}/calibration/v1_records.json",
            f"{LOC0}/calibration/d2_vectors.npz"] + [f"{LOC0}/{d}/{i:03d}{s}" for d in ("barrier", "pulses") for i in range(80) for s in (".json", "_logits.npz", "_arrays.npz")]
    checks["st_loc0_sealed_comparators"] = all(seal["files"].get(p) == file_hash(ROOT / p) for p in need)
    sealed = json.loads((ROOT / EXP1 / "directions_sealed.json").read_text())
    checks["exp1_sealed"] = sealed["status"] == "SEALED" and all(file_hash(ROOT / EXP1 / rel) == h for rel, h in sealed["files"].items())
    plan = {"schema": SCHEMA + "_plan", "config_sha256": file_hash(ROOT / CONFIG), "config_hash": digest(cfg), "panel_identity_hash": P["identity_hash"],
            "checks": checks, "runtime": rp, "arms": arms, "barrier": {k: list(v) for k, v in BARRIER.items()}, "layers": list(LAYERS),
            "comparators": {"st_loc0_run": LOC0, "st_loc0_seal": LOC0_SEAL, "exp1": EXP1}, "references_used": False, "created_unix": time.time()}
    plan["plan_hash"] = digest(plan)
    out = ROOT / PLAN
    if out.exists():
        raise FileExistsError("plan exists; never overwrite")
    if not all(checks.values()):
        raise SystemExit("BLOCK: " + json.dumps([k for k, v in checks.items() if not v]))
    atomic_json(out, plan)
    print(json.dumps({"plan_hash": plan["plan_hash"], "checks": checks}))


SOURCES = FROZEN + (PLAN, "src/csasr/inference_cf/prompt_r1.py", "src/csasr/inference_cf/loc0_sites.py", "src/csasr/inference_cf/core_p1.py",
                    "src/csasr/lss/sites.py", "src/csasr/models/hooks.py", "src/csasr/models/whisper.py", "experiments/inference_cf_p2r.py",
                    "experiments/inference_cf_cached.py", "experiments/inference_cf_p2dir.py", "experiments/inference_cf_st_prompt_r1.py",
                    "experiments/inference_cf_st_prompt_r1_geometry.py", "experiments/inference_cf_st_prompt_r1_analyze.py",
                    "experiments/inference_cf_st_prompt_r1_audit.py", "slurm/inference_cf_st_prompt_r1.sbatch", "tests/test_st_prompt_r1_impl.py")


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    pre = f"{BASE}/prerun_audit_A.json"
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, pre)
    if dirty:
        raise ValueError("commit sources before the manifest:\n" + dirty)
    cfg = json.loads((ROOT / CONFIG).read_text())
    if json.loads((ROOT / pre).read_text())["verdict"] != cfg["audit"]["pre_A"]:
        raise ValueError("pre-run audit did not pass")
    plan = json.loads((ROOT / PLAN).read_text())
    man = {"schema": SCHEMA, "stage": "A_run1", "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "config_sha256": file_hash(ROOT / CONFIG), "config_hash": digest(cfg), "panel_identity_hash": plan["panel_identity_hash"],
           "plan_hash": plan["plan_hash"], "output_root": args.out, "energy": cfg["energy"], "bootstrap": cfg["bootstrap"],
           "reference_access_boundary": "no evaluator targets/competitors/strata in runner; evaluator only after pushed seal + primary audit PASS",
           "environment": prep.environment(), "model": {"dir": str(prep.MODEL), "files": prep.model_hashes()},
           "sources": {p: file_hash(ROOT / p) for p in SOURCES + (pre,)}, "trainable_parameters": 0, "references_used": False,
           "status": "RUNNING", "created_unix": time.time()}
    man["manifest_hash"] = digest(man)
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("manifest exists; never overwrite")
    atomic_json(out / "manifest.json", man)
    print(json.dumps({"manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}))


# ---------------------------------------------------------------------------------------------------------------------

def load_run(args):
    m = json.loads((ROOT / args.out / "manifest.json").read_text())
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
    return m, plan, cfg


def load_model(plan, cfg):
    import torch
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.lss.sites import num_forced_prefix_from
    from csasr.models.whisper import load_whisper
    from csasr.utils.config import load_config
    torch.manual_seed(240924)
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    bundle.model.eval()
    bundle.model.requires_grad_(False)
    if tokenizer_partition(bundle.processor.tokenizer)["hash"] != cfg["conditions"]["partition_hash"]:
        raise ValueError("partition")
    gen = bundle.model.generation_config
    if digest({"suppress": list(gen.suppress_tokens or []), "begin": list(gen.begin_suppress_tokens or [])}) != cfg["conditions"]["suppression_hash"]:
        raise ValueError("suppression")
    if num_forced_prefix_from(bundle.processor, language="zh", task="transcribe") != 4:
        raise ValueError("forced prefix")
    return bundle


class Engine:
    """Model-owned branches, passive 4-layer recorder, pulses (reference-free)."""

    def __init__(self, bundle, counters):
        import experiments.inference_cf_p2r as p2r
        self.p2r, self.bundle, self.c = p2r, bundle, counters
        gen = bundle.model.generation_config
        self.suppress, self.begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])

    def step(self, br, new, *, attention, hook=None):
        from csasr.inference_cf.loc0_sites import Composite, assert_no_any_site_hooks
        from csasr.lss.sites import DecoderPostCrossAttnRecorder
        rec = DecoderPostCrossAttnRecorder(self.bundle, list(LAYERS), keep_last_only=True)
        logits, _, _ = br.step(new, capture_layer=None, attention=attention, hook=Composite(hook, rec))
        self.c["forwards"] += 1
        assert_no_any_site_hooks(self.bundle)
        return logits, {l: rec.states[l][0, -1].float().cpu() for l in LAYERS}

    def pulse(self, br, L, new, layer, action):
        from csasr.inference_cf.loc0_sites import cross_pulse_hook
        br.crop(L)
        hook = cross_pulse_hook(self.bundle, layer, action, 4)
        logits, st = self.step(br, new, attention=True, hook=hook)
        self.c["pulse_calls"] += 1
        rec = hook.records[-1].to_dict() if hook.records else None
        return logits, st, rec


def encode(bundle, path, counters):
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.models.whisper import batch_model_inputs
    inputs = batch_model_inputs(bundle, [path])
    with torch.inference_mode():
        h = bundle.model.model.encoder(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state
    counters["encoder_passes"] += 1
    return BaseModelOutput(last_hidden_state=h)


def cmd_capture(args) -> None:
    """Passive paired E/M extraction at the four layers + CPU geometry ledger (no pulses, no outcomes)."""
    import torch
    import experiments.inference_cf_p2dir as p2dir
    from csasr.inference_cf.core_p1 import direction, processed_argmax
    from csasr.inference_cf.prompt_r1 import geometry_cell, random_vectors
    from csasr.inference_cf.unique import array_hash
    m, plan, cfg = load_run(args)
    out = ROOT / args.out
    cap = out / "capture"
    cap.mkdir(parents=True, exist_ok=True)
    if (cap / "capture.json").exists():
        raise FileExistsError("capture exists")
    t0 = time.time()
    bundle = load_model(plan, cfg)
    counters = {"forwards": 0, "pulse_calls": 0, "encoder_passes": 0, "autograd_calls": 0, "LID_calls": 0}
    eng = Engine(bundle, counters)
    Q, Ut = plan["runtime"]["runtime_queries"], plan["runtime"]["utterances"]
    by_u: dict = {}
    for j, q in enumerate(Q):
        by_u.setdefault(q["utterance_id"], []).append((j, q))
    SB = {l: np.zeros((180, 1280), np.float32) for l in LAYERS}
    SE = {l: np.zeros((180, 1280), np.float32) for l in LAYERS}
    base, argmax, fails = {}, {}, []
    torch.cuda.reset_peak_memory_stats()
    for u in Ut:
        enc = encode(bundle, u["audio_path"], counters)
        toks = u["baseline_content_tokens"]
        jt = {q["t"]: j for j, q in by_u[u["utterance_id"]]}
        B, E = eng.p2r.DiagBranch(bundle, enc, CB, "B"), eng.p2r.DiagBranch(bundle, enc, CE, "E")
        new, new_e = list(CB), list(CE)
        for t in range(max(jt) + 1):
            lb, sb = eng.step(B, new, attention=True)
            _, se = eng.step(E, new_e, attention=False)
            if t in jt:
                j = jt[t]
                for l in LAYERS:
                    SB[l][j], SE[l][j] = sb[l].numpy(), se[l].numpy()
                base[j] = lb
                argmax[j] = processed_argmax(lb, t, eng.suppress, eng.begin)
            if t < len(toks):
                new, new_e = [toks[t]], [toks[t]]
    # ---- L16 historical identity (bitwise) ----
    with np.load(ROOT / LOC0 / "calibration/states_L16_CROSS.npz") as z:
        hb, he = z["H_B"], z["H_E"]
    with np.load(ROOT / LOC0 / "calibration/baseline_logits.npz") as z:
        hl = {k: z[k] for k in z.files}
    ident = {"L16_B_bitwise": bool(np.array_equal(SB[16], hb)), "L16_E_bitwise": bool(np.array_equal(SE[16], he)),
             "NONE_logits_bitwise": all(np.array_equal(p2dir.pack_bf16(base[j]), hl[f"q{j:03d}"]) for j in range(180)),
             "finite": all(bool(torch.isfinite(base[j]).all()) for j in range(180)) and all(np.isfinite(SB[l]).all() and np.isfinite(SE[l]).all() for l in LAYERS)}
    # ---- v_prompt per query/layer; L16 == historical D0 ----
    V, vrec = {}, {}
    for j in range(180):
        for l in LAYERS:
            d = direction(torch.from_numpy(SE[l][j].copy()), torch.from_numpy(SB[l][j].copy()))
            vrec[f"q{j:03d}_L{l}"] = {"status": d["status"], "norm": d["norm"], "sha256": None if d["d"] is None else array_hash(d["d"].numpy())}
            if d["d"] is not None:
                V[f"q{j:03d}_L{l}"] = d["d"].numpy()
    v1 = json.loads((ROOT / LOC0 / "calibration/v1_records.json").read_text())
    ident["L16_D0_vectors_identical"] = all(vrec[f"q{j:03d}_L16"]["sha256"] == v1[f"q{j:03d}_L16_CROSS"]["sha256"] for j in range(180))
    # ---- CPU geometry ledger for all 36 cells/query (no logits, no outcomes) ----
    rv = random_vectors(cfg)
    ledger = {}
    for j in range(180):
        for a in plan["arms"]:
            l = a["layer"]
            r = torch.from_numpy(SB[l][j].copy()).to(torch.bfloat16)
            v = rv[a["id"]] if a["family"] == "random" else (None if f"q{j:03d}_L{l}" not in V else a["sign"] * V[f"q{j:03d}_L{l}"])
            ledger[f"q{j:03d}|{a['id']}"] = geometry_cell(r, v, a["eta"], cfg["energy"], eng.p2r.solve_scale)
    for l in LAYERS:
        np.savez(cap / f"states_L{l:02d}.npz", H_M=SB[l], H_E=SE[l])
    np.savez(cap / "v_prompt.npz", **V)
    np.savez_compressed(cap / "none_logits.npz", **{f"q{j:03d}": p2dir.pack_bf16(base[j]) for j in range(180)})
    atomic_json(cap / "v_prompt_records.json", vrec)
    atomic_json(cap / "geometry_ledger.json", ledger)
    doc = {"schema": SCHEMA + "_capture", "manifest_hash": m["manifest_hash"], "identity": ident, "identity_ok": all(ident.values()),
           "argmax_none": {f"q{j:03d}": argmax[j] for j in range(180)}, "counters": counters, "elapsed_sec": time.time() - t0,
           "peak_alloc": int(torch.cuda.max_memory_allocated()), "job_id": os.environ.get("SLURM_JOB_ID"),
           "files": {p.name: file_hash(p) for p in sorted(cap.iterdir()) if p.is_file()}}
    atomic_json(cap / "capture.json", doc)
    print(json.dumps({"identity": ident, "elapsed": doc["elapsed_sec"]}), flush=True)
    if not doc["identity_ok"]:
        raise SystemExit("ST_PROMPT_R1_A_INVALID: capture historical identity failed " + json.dumps(ident))


def cmd_pulses(args) -> None:
    import torch
    import experiments.inference_cf_p2dir as p2dir
    from csasr.inference_cf.core_p1 import processed_argmax
    from csasr.inference_cf.loc0_sites import assert_no_any_site_hooks, cache_fingerprint, pulse_action
    from csasr.inference_cf.prompt_r1 import cell_checks, random_vectors, relative_action
    from csasr.inference_cf.unique import array_hash
    m, plan, cfg = load_run(args)
    out = ROOT / args.out
    cap = out / "capture"
    capd = json.loads((cap / "capture.json").read_text())
    if not capd["identity_ok"] or any(file_hash(cap / n) != h for n, h in capd["files"].items()):
        raise ValueError("capture invalid/changed")
    gate = json.loads((out / "geometry_gate.json").read_text())
    if gate.get("verdict") != "GEOMETRY_COVERAGE_PASS":
        raise SystemExit("geometry gate did not pass; no pulses (explicit human dose revision required)")
    for d in ("barrier", "pulses"):
        (out / d).mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    bundle = load_model(plan, cfg)
    w0 = {k: v.detach().clone() for k, v in bundle.model.state_dict().items() if k.endswith("final_layer_norm.weight")}
    counters = {"forwards": 0, "pulse_calls": 0, "zero_dose_checks": 0, "restore_checks": 0, "cache_fingerprints": 0, "encoder_passes": 0,
                "solver_evals": 0, "autograd_calls": 0, "LID_calls": 0}
    eng = Engine(bundle, counters)
    ecfg = cfg["energy"]
    e_star = float(cfg["controls"]["D2_energy"])
    Q, Ut = plan["runtime"]["runtime_queries"], plan["runtime"]["utterances"]
    by_u: dict = {}
    for j, q in enumerate(Q):
        by_u.setdefault(q["utterance_id"], []).append((j, q))
    with np.load(cap / "v_prompt.npz") as z:
        V = {k: z[k] for k in z.files}
    S = {}
    for l in LAYERS:
        with np.load(cap / f"states_L{l:02d}.npz") as z:
            S[l] = z["H_M"]
    with np.load(cap / "none_logits.npz") as z:
        none_lg = {k: z[k] for k in z.files}
    with np.load(ROOT / LOC0 / "calibration/d2_vectors.npz") as z:
        D2 = {k: z[k] for k in z.files}
    rv = random_vectors(cfg)
    arms = {a["id"]: a for a in plan["arms"]}
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "start_unix": t_start,
               "status": "running", "counters": counters}
    atomic_json(out / "runtime.json", runtime)
    torch.cuda.reset_peak_memory_stats()
    failures: list = []
    encs = {}

    def v_fn(fam, j, layer, sign, aid):
        if fam == "prompt":
            v = V.get(f"q{j:03d}_L{layer}")
            return (lambda r: (None, "tiny_or_nonfinite_direction")) if v is None else (lambda r, v=v: (torch.from_numpy((sign * v).astype(np.float32)), "ok"))
        if fam == "random":
            return lambda r: (torch.from_numpy(rv[aid].copy()), "ok")
        if fam == "D2":
            return lambda r: (torch.from_numpy(D2[f"q{j:03d}"].astype(np.float32)), "ok")
        raise ValueError(fam)

    def run_pass(name, barrier):
        for i, u in enumerate(Ut):
            uid = u["utterance_id"]
            if uid not in encs:
                encs[uid] = encode(bundle, u["audio_path"], counters)
            toks = u["baseline_content_tokens"]
            jt = {q["t"]: (j, q) for j, q in by_u[uid]}
            B = eng.p2r.DiagBranch(bundle, encs[uid], CB, "B")
            new = list(CB)
            rows, logits_out, arrays_out = {}, {}, {}
            if barrier:
                hist = {}
                for d in ("barrier", "pulses"):
                    with np.load(ROOT / LOC0 / d / f"{i:03d}_logits.npz") as z:
                        hist[d + "_lg"] = {k: z[k] for k in z.files}
                    with np.load(ROOT / LOC0 / d / f"{i:03d}_arrays.npz") as z:
                        hist[d + "_ar"] = {k: z[k] for k in z.files}
                    hist[d] = json.loads((ROOT / LOC0 / d / f"{i:03d}.json").read_text())
            for t in range(max(jt) + 1):
                L = B.length
                if t in jt:
                    j, q = jt[t]
                    query = q["absolute_query"]
                    fp0 = cache_fingerprint(B.cache, L)
                    counters["cache_fingerprints"] += 1
                    lb, sb = eng.step(B, new, attention=True)
                    if L + len(new) - 1 != query:
                        raise RuntimeError("query index mismatch")
                    clean = bool(np.array_equal(p2dir.pack_bf16(lb), none_lg[f"q{j:03d}"])) and all(np.array_equal(sb[l].numpy(), S[l][j]) for l in LAYERS)
                    prow = {"q": j, "t": t, "query": query, "clean_replay_bitwise": clean, "cells": {}}
                    if not clean:
                        failures.append(f"{name}:{uid}:t{t}:clean_replay")
                    if barrier:
                        prow["zero_dose"] = {}
                        for l in LAYERS:
                            info: dict = {}
                            la, _, _ = eng.pulse(B, L, new, l, pulse_action(query, lambda r: (None, "zero_dose"), 1.0, info, 1.0, eng.p2r.solve_scale, eng.p2r.scaled_direction))
                            counters["zero_dose_checks"] += 1
                            prow["zero_dose"][str(l)] = bool(torch.equal(la, lb))
                            if not torch.equal(la, lb):
                                failures.append(f"{name}:{uid}:t{t}:zero:L{l}")
                        for bid, (hd, haid, fam, sign) in BARRIER.items():
                            info = {}
                            act = pulse_action(query, v_fn(fam, j, 16, sign, bid), e_star, info, ecfg["max_relative_squared_error"], eng.p2r.solve_scale, eng.p2r.scaled_direction)
                            la, sa, rec = eng.pulse(B, L, new, 16, act)
                            counters["solver_evals"] += int(info.get("evals", 0) or 0)
                            hrec = hist[hd]["positions"][str(t)]["cells"][haid]
                            bar = {"logits_bitwise": bool(np.array_equal(p2dir.pack_bf16(la), hist[hd + "_lg"][f"q{j:03d}_{haid}"])),
                                   "consumed_bitwise": bool(np.array_equal(sa[16].numpy(), hist[hd + "_ar"][f"q{j:03d}_{haid}_consumed"])) if hrec["steered"] else None,
                                   "solver_s": abs(float(info.get("s", 0.0)) - float(hrec["solver"].get("s", 0.0))) <= cfg["historical_reproduction"]["solver_scale_D0_abs"] * max(1.0, abs(float(hrec["solver"].get("s", 0.0)))),
                                   "edit_norm_equal": (rec["edit_norm"] if rec and rec["steered"] else 0.0) == hrec["edit_norm"]}
                            prow["cells"][bid] = {"barrier": bar, "steered": bool(rec and rec["steered"])}
                            logits_out[f"q{j:03d}_{bid}"] = p2dir.pack_bf16(la)
                            if rec and rec["steered"]:
                                arrays_out[f"q{j:03d}_{bid}_consumed"] = sa[16].numpy()
                            if not all(v for v in bar.values() if v is not None):
                                failures.append(f"{name}:{uid}:t{t}:{bid}:{bar}")
                            B.crop(L)
                    else:
                        for aid, arm in arms.items():
                            info = {}
                            act = relative_action(query, v_fn(arm["family"], j, arm["layer"], arm["sign"], aid), arm["eta"], info,
                                                  ecfg["max_relative_squared_error"], ecfg["minimum_state_norm"], eng.p2r.solve_scale, eng.p2r.scaled_direction)
                            la, sa, rec = eng.pulse(B, L, new, arm["layer"], act)
                            counters["solver_evals"] += int(info.get("evals", 0) or 0)
                            l = arm["layer"]
                            steered = bool(rec and rec["steered"])
                            c = {"arm": aid, "layer": l, "eta": arm["eta"], "sign": arm["sign"], "family": arm["family"], "steered": steered,
                                 "solver": {k: v for k, v in info.items() if k not in ("v", "_proposed")}, "integrity_failures": [],
                                 "argmax": processed_argmax(la, t, eng.suppress, eng.begin)}
                            if steered:
                                ch = cell_checks(arm["eta"], info, rec, sb[l], sa[l], ecfg)
                                c.update({k: v for k, v in ch.items() if k != "checks"}, checks=ch["checks"], valid=all(ch["checks"].values()))
                                c["integrity_failures"] += [k for k, v in ch["checks"].items() if not v]
                                c["solver_v_sha256"] = array_hash(info["v"].numpy())
                                arrays_out[f"q{j:03d}_{aid}_proposed"] = info["_proposed"].numpy()
                                arrays_out[f"q{j:03d}_{aid}_consumed"] = sa[l].numpy()
                            else:
                                c["valid"] = False
                                c["no_edit_reason"] = info.get("no_edit_reason")
                                if c["no_edit_reason"] not in cfg["eligibility"]["no_edit_reasons"]:
                                    c["integrity_failures"].append(f"unexpected_no_edit:{c['no_edit_reason']}")
                                if not torch.equal(la, lb):
                                    c["integrity_failures"].append("no_edit_logits_not_baseline")
                            if not bool(torch.isfinite(la).all()):
                                c["integrity_failures"].append("nonfinite_logits")
                            B.crop(L)
                            c["cache_prefix_unchanged"] = cache_fingerprint(B.cache, L) == fp0
                            c["cache_positions_ok"] = B.positions == list(range(L)) and B.cache.get_seq_length() == L
                            counters["cache_fingerprints"] += 1
                            if not (c["cache_prefix_unchanged"] and c["cache_positions_ok"]):
                                c["integrity_failures"].append("cache_lineage")
                            if c["integrity_failures"]:
                                failures.append(f"{name}:{uid}:t{t}:{aid}:{c['integrity_failures']}")
                            prow["cells"][aid] = c
                            logits_out[f"q{j:03d}_{aid}"] = p2dir.pack_bf16(la)
                        # pairwise squared-energy match among same-target (layer, eta) valid cells
                        prow["pairwise"] = {}
                        for l in LAYERS:
                            for eta in ecfg["eta"]:
                                en = [c["realized_edit_norm"] ** 2 for c in prow["cells"].values() if c.get("layer") == l and c.get("eta") == eta and c.get("valid")]
                                okp = (not en) or max(en) / min(en) - 1 <= ecfg["max_pairwise_relative_squared_error"]
                                prow["pairwise"][f"L{l}_eta{eta}"] = okp
                                if not okp:
                                    failures.append(f"{name}:{uid}:t{t}:pairwise:L{l}:{eta}")
                    B.crop(L)
                    fp1 = cache_fingerprint(B.cache, L)
                    lb2, sb2 = eng.step(B, new, attention=True)
                    counters["restore_checks"] += 1
                    prow["restore_bitwise"] = bool(torch.equal(lb2, lb)) and all(torch.equal(sb2[l], sb[l]) for l in LAYERS) and fp1 == fp0
                    if not prow["restore_bitwise"]:
                        failures.append(f"{name}:{uid}:t{t}:restore")
                    rows[str(t)] = prow
                else:
                    eng.step(B, new, attention=True)
                if t < len(toks):
                    new = [toks[t]]
            if logits_out:
                np.savez_compressed(out / name / f"{i:03d}_logits.npz", **logits_out)
                np.savez_compressed(out / name / f"{i:03d}_arrays.npz", **arrays_out)
            atomic_json(out / name / f"{i:03d}.json", {"identity": uid, "utterance_index": i, "manifest_hash": m["manifest_hash"], "positions": rows,
                                                     "logits_sha256": file_hash(out / name / f"{i:03d}_logits.npz") if logits_out else None,
                                                     "arrays_sha256": file_hash(out / name / f"{i:03d}_arrays.npz") if logits_out else None})
            print(f"R1-A {name} {i + 1}/80 {uid} failures {len(failures)}", flush=True)

    try:
        t_b = time.time()
        run_pass("barrier", True)
        runtime["barrier"] = {"pass": not failures, "failures": failures[:50], "sec": time.time() - t_b}
        atomic_json(out / "runtime.json", runtime)
        if failures:
            raise RuntimeError("ST_PROMPT_R1_A_INVALID: historical barrier failed before new pulses: " + "; ".join(failures[:10]))
        t_p = time.time()
        run_pass("pulses", False)
        runtime["pulses_sec"] = time.time() - t_p
        status = "completed" if not failures else "failed"
        if failures:
            runtime["failure"] = {"reason": "integrity: " + "; ".join(failures[:20])}
    except Exception as exc:
        import traceback
        status = "failed"
        runtime["failure"] = {"reason": repr(exc), "traceback": traceback.format_exc()}
    assert_no_any_site_hooks(bundle)
    w1 = {k: v for k, v in bundle.model.state_dict().items() if k in w0}
    runtime.update(status=status, failures=failures[:200], n_failures=len(failures), end_unix=time.time(),
                   peak_alloc=int(torch.cuda.max_memory_allocated()), peak_reserved=int(torch.cuda.max_memory_reserved()),
                   model_grads_none=all(p.grad is None for p in bundle.model.parameters()), requires_grad_any=any(p.requires_grad for p in bundle.model.parameters()),
                   weights_unchanged_sample=all(torch.equal(w0[k], w1[k]) for k in w0))
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)
    if status != "completed":
        raise SystemExit("R1-A run failed: " + runtime["failure"]["reason"])


def cmd_seal(args) -> None:
    run = ROOT / args.run
    files = sorted(p for p in run.rglob("*") if p.is_file()) + [ROOT / BASE / "primary_analysis_A.json"]
    man = json.loads((run / "manifest.json").read_text())
    prim = json.loads((ROOT / BASE / "primary_analysis_A.json").read_text())
    doc = {"schema": SCHEMA + "_A_output_seal", "run": args.run, "files": {str(p.relative_to(ROOT)): file_hash(p) for p in files},
           "manifest_hash": man["manifest_hash"], "source_commit": man["git_commit"], "config_sha256": file_hash(ROOT / CONFIG),
           "panel_identity_hash": man["panel_identity_hash"], "plan_hash": man["plan_hash"], "primary_valid": prim["valid"],
           "references_used": False, "status": "COMPLETE" if prim["valid"] else "INVALID", "created_unix": time.time()}
    doc["seal_hash"] = digest(doc)
    out = ROOT / BASE / "output_seal_A.json"
    if out.exists():
        raise FileExistsError("output seal exists")
    atomic_json(out, doc)
    print(json.dumps({"seal_hash": doc["seal_hash"], "files": len(doc["files"]), "primary_valid": doc["primary_valid"]}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    for nm in ("manifest", "capture", "pulses"):
        sub.add_parser(nm).add_argument("--out", required=True)
    sub.add_parser("seal").add_argument("--run", required=True)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "manifest": cmd_manifest, "capture": cmd_capture, "pulses": cmd_pulses, "seal": cmd_seal}[args.cmd](args)


if __name__ == "__main__":
    main()
