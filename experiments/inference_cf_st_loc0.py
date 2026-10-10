#!/usr/bin/env python
"""ST-LOC0 runner — frozen-direction multi-site causal feasibility (docs/inference_cf/ST_LOC0_SPEC.md,
ST_LOC0_CODEX_DESIGN.md, ST_LOC0_PANEL.json, configs/inference_cf/st_loc0.json; freeze 39ff5e9).

prepare   (CPU) verify config/panel/source/model identities, historical artifacts (extract states, folds, exp1 rows, sealed
          D0/D1/D2 hashes, P2-DIR independent D2 spot audit), random-control hashes; write plan_sealed.json with the RUNTIME
          projection only (runtime_queries + utterances + arm/site/control definitions + historical comparator paths).
manifest  (CPU) resolved manifest after committed PASS_TO_ST_LOC0.
run       (GPU) ONE allocation:
          A  passive paired B/E replay of B0M_L16 prefixes; 4-site unedited states (L16/L24 x self/cross); historical L16
             DG-02 B/E states and NONE logits must reproduce bitwise; V1 per query per site (core_p1.direction).
          B  offline calibration builder: 80 site/fold V2 fits (L16 cross == historical folds); random + D2 vectors;
             calibration seal written, then the job WAITS for GO.json (written only after the seal is pushed to remote).
          C1 historical barrier at L16 DG-02 on all 180: NONE / D0 (= v_prompt L16 cross +) / D1 (= v_unq L16 cross +) /
             D2 READOUT, bitwise vs historical logits + consumed states; zero-dose at all four sites; restore + cache
             fingerprints. Any mismatch aborts before new-site pulses.
          C2 remaining 14 primary arms + 4 random controls, single matched-energy pulses from the identical pre-query cache.
seal      (CPU) immutable reference-free output seal.
The pulse runner reads only the runtime projection and sealed vectors; never stratum labels, references or evaluator data.
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

SCHEMA = "st_loc0_v1"
CONFIG = "configs/inference_cf/st_loc0.json"
PANEL = "docs/inference_cf/ST_LOC0_PANEL.json"
SPEC = "docs/inference_cf/ST_LOC0_SPEC.md"
DESIGN = "docs/inference_cf/ST_LOC0_CODEX_DESIGN.md"
EXTRACT = "results/inference_cf/p2dir/extract_run1"
EXP1 = "results/inference_cf/p2dir/exp1_run1"
FOLDS = "results/inference_cf/p2dir/folds_run1/folds.json"
BASE = "results/inference_cf/st_loc0"
PLAN = f"{BASE}/plan_sealed.json"
CB = [50258, 50260, 50360, 50364]
CE = [50258, 50259, 50360, 50364]
LAYERS = (16, 24)
SITES = ("POST_SELF_ATTENTION", "POST_CROSS_ATTENTION_PRE_FFN")
SITE_KEYS = {(L, s): f"L{L}_{'SELF' if s == SITES[0] else 'CROSS'}" for L in LAYERS for s in SITES}
BARRIER_ARMS = ("v_prompt_L16_POST_CROSS_ATTENTION_PRE_FFN_plus", "v_unq_L16_POST_CROSS_ATTENTION_PRE_FFN_plus", "D2_L16_POST_CROSS_ATTENTION_PRE_FFN")
HIST_NAME = {BARRIER_ARMS[0]: "D0", BARRIER_ARMS[1]: "D1", BARRIER_ARMS[2]: "D2"}
NO_EDIT_REASONS = ("tiny_or_nonfinite_V1_delta", "invalid_new_site_D1_fold", "energy_unreachable",
                   "solver_target_unattainable_within_8_evaluations")
GO_TIMEOUT_SEC = 3600


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def committed(rel: str) -> bool:
    import hashlib
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return git("ls-files", rel) == rel and "sha256:" + hashlib.sha256(blob).hexdigest() == file_hash(ROOT / rel)


def runtime_projection(panel: dict) -> dict:
    """The ONLY panel content the pulse runner may consume (spec section 2 firewall)."""
    return {"runtime_queries": panel["runtime_queries"], "utterances": panel["utterances"]}


def arm_table(cfg: dict) -> list[dict]:
    arms = [dict(a) for a in cfg["primary_arms"]]
    for c in cfg["controls"]["random"]:
        arms.append({"id": c["id"], "family": "random", "layer": c["layer"], "site": c["site"], "sign": 1})
    arms.append({"id": BARRIER_ARMS[2], "family": "D2", "layer": 16, "site": SITES[1], "sign": 1})
    return arms


def random_vectors(cfg: dict) -> dict:
    from csasr.inference_cf.unique import array_hash
    out = {}
    for c in cfg["controls"]["random"]:
        v = np.random.default_rng(int(c["seed"])).standard_normal(1280)
        v = (v / np.linalg.norm(v)).astype(np.float32)
        if array_hash(v) != c["array_sha256"]:
            raise ValueError(f"random control hash mismatch {c['id']}")
        out[c["id"]] = v
    return out


def cmd_prepare(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    import experiments.inference_cf_p2tta0 as t0run
    cfg = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    checks = {"frozen_committed": all(committed(p) for p in (CONFIG, PANEL, SPEC, DESIGN)),
              "panel_identity": digest({k: v for k, v in P.items() if k != "identity_hash"}) == P["identity_hash"] == cfg["panel_identity_hash"],
              "sources": all(file_hash(ROOT / p) == h for p, h in cfg["source_sha256"].items()),
              "panel_fingerprints": all(file_hash(ROOT / p) == h for p, h in P["fingerprints"].items()),
              "model_files": prep.model_hashes() == cfg["conditions"]["model"]["files"]}
    tok = WhisperProcessor.from_pretrained(prep.MODEL, local_files_only=True).tokenizer
    gen = GenerationConfig.from_pretrained(prep.MODEL, local_files_only=True)
    part = tokenizer_partition(tok)
    checks["partition"] = part["hash"] == cfg["conditions"]["partition_hash"]
    checks["suppression"] = digest({"suppress": list(gen.suppress_tokens or []), "begin": list(gen.begin_suppress_tokens or [])}) == cfg["conditions"]["suppression_hash"]
    rp = runtime_projection(P)
    by_u = {u["utterance_id"]: u for u in rp["utterances"]}
    ok = True
    for q in rp["runtime_queries"]:
        u = by_u[q["utterance_id"]]
        b = json.loads((ROOT / u["baseline_row"]).read_text())["systems"][cfg["conditions"]["baseline_system"]]
        prefix = b["tokens"][:q["t"]]
        ok &= (u["baseline_content_tokens"] == b["tokens"] and q["absolute_query"] == 4 + q["t"] - 1 and digest(prefix) == q["content_prefix_sha256"]
               and digest(CB + prefix) == q["forced_zh_query_input_sha256"] and digest(CE + prefix) == q["forced_en_query_input_sha256"])
    checks["runtime_queries"] = bool(ok) and len(rp["runtime_queries"]) == 180 and len(rp["utterances"]) == 80
    audio_ok = all(t0run.audio_fingerprint_64k(u["audio_path"]) == u["audio_sha256"] or t0run.audio_full_sha256(u["audio_path"]) == u["audio_sha256"]
                   for u in rp["utterances"])
    checks["audio"] = bool(audio_ok)
    # historical sealed comparators (reference-free): exp1 rows, D0/D1/D2 hashes, P2-DIR spot + final audits
    sealed = json.loads((ROOT / EXP1 / "directions_sealed.json").read_text())
    checks["exp1_sealed"] = sealed["status"] == "SEALED" and all(file_hash(ROOT / EXP1 / rel) == h for rel, h in sealed["files"].items())
    spot = json.loads((ROOT / EXP1 / "audit_spot.json").read_text())
    final = json.loads((ROOT / "results/inference_cf/p2dir/final_audit.json").read_text())
    checks["historical_D2_independent_audit"] = spot["ok"] is True and final["verdict"] == "P2_DIR_AUDIT: PASS"
    rv = random_vectors(cfg)
    checks["random_controls"] = len(rv) == 4
    arms = arm_table(cfg)
    checks["arm_matrix"] = len(arms) == 21 and len({a["id"] for a in arms}) == 21 and all(a["id"] in {x["id"] for x in arms} for a in cfg["primary_arms"])
    import transformers, inspect, hashlib
    from transformers.models.whisper.modeling_whisper import WhisperDecoderLayer
    checks["site_forward"] = (transformers.__version__ == cfg["site_forward"]["transformers_version"]
                              and "sha256:" + hashlib.sha256(inspect.getsource(WhisperDecoderLayer.forward).encode()).hexdigest() == cfg["site_forward"]["forward_source_sha256"]
                              and file_hash(inspect.getfile(WhisperDecoderLayer)) == cfg["site_forward"]["module_file_sha256"])
    plan = {"schema": SCHEMA + "_plan", "config_sha256": file_hash(ROOT / CONFIG), "config_hash": digest(cfg), "panel_identity_hash": P["identity_hash"],
            "checks": checks, "runtime": rp, "arms": arms, "barrier_arms": list(BARRIER_ARMS), "sites": [list(k) + [v] for k, v in SITE_KEYS.items()],
            "partition_hash": part["hash"], "suppression_hash": cfg["conditions"]["suppression_hash"],
            "historical": {"extract": EXTRACT, "exp1": EXP1, "folds": FOLDS}, "references_used": False, "created_unix": time.time()}
    plan["plan_hash"] = digest(plan)
    out = ROOT / PLAN
    if out.exists():
        raise FileExistsError("plan exists; never overwrite")
    if not all(checks.values()):
        raise SystemExit("BLOCK: " + json.dumps([k for k, v in checks.items() if not v]))
    atomic_json(out, plan)
    print(json.dumps({"plan_hash": plan["plan_hash"], "checks": checks}))


SOURCES = (SPEC, DESIGN, PANEL, CONFIG, PLAN, "src/csasr/inference_cf/loc0_sites.py", "src/csasr/inference_cf/core_p1.py",
           "src/csasr/inference_cf/unique.py", "src/csasr/inference_cf/directions.py", "src/csasr/lss/sites.py", "src/csasr/models/hooks.py",
           "src/csasr/models/whisper.py", "experiments/inference_cf_p2r.py", "experiments/inference_cf_cached.py", "experiments/inference_cf_p2dir.py",
           "experiments/inference_cf_p2dir_analyze.py", "experiments/inference_cf_st_loc0.py", "experiments/inference_cf_st_loc0_calibrate.py",
           "experiments/inference_cf_st_loc0_analyze.py", "experiments/inference_cf_st_loc0_audit.py", "slurm/inference_cf_st_loc0.sbatch",
           "tests/test_st_loc0_impl.py", "tests/test_st_loc0_contract.py")


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    pre = f"{BASE}/prerun_audit.json"
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES, pre)
    if dirty:
        raise ValueError("commit ST-LOC0 sources before preparing a manifest:\n" + dirty)
    if json.loads((ROOT / pre).read_text())["verdict"] != "PASS_TO_ST_LOC0":
        raise ValueError("pre-run audit did not pass")
    plan = json.loads((ROOT / PLAN).read_text())
    cfg = json.loads((ROOT / CONFIG).read_text())
    P = json.loads((ROOT / PANEL).read_text())
    man = {"schema": SCHEMA, "stage": "run1", "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
           "config_sha256": file_hash(ROOT / CONFIG), "config_hash": digest(cfg), "panel_identity_hash": P["identity_hash"], "plan_hash": plan["plan_hash"],
           "AB_membership_hashes": {d: f["membership_sha256"] for d, f in P["calibration_membership"]["folds"].items()},
           "sites": plan["sites"], "direction_algorithms": {"V1": file_hash(ROOT / "src/csasr/inference_cf/core_p1.py"),
                                                            "V2": file_hash(ROOT / "src/csasr/inference_cf/unique.py")},
           "bootstrap": cfg["bootstrap"], "energy": cfg["energy"], "output_root": args.out,
           "reference_access_boundary": "no evaluator/reference before output seal pushed + primary audit PASS; calibration seal pushed before pulses",
           "environment": prep.environment(), "model": {"dir": str(prep.MODEL), "files": prep.model_hashes()},
           "sources": {p: file_hash(ROOT / p) for p in SOURCES + (pre,)}, "references_used": False, "created_unix": time.time()}
    man["manifest_hash"] = digest(man)
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("manifest exists; never overwrite")
    atomic_json(out / "manifest.json", man)
    print(json.dumps({"manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}))


# ---------------------------------------------------------------------------------------------------------------------

class Engine:
    """GPU state: model, B/E replay, pulse execution (reference-free)."""

    def __init__(self, bundle, cfg, nfp, counters):
        import experiments.inference_cf_p2r as p2r
        self.p2r, self.bundle, self.cfg, self.nfp, self.c = p2r, bundle, cfg, nfp, counters
        gen = bundle.model.generation_config
        self.suppress, self.begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
        self.e_star = float(cfg["energy"]["e_star"])
        self.max_rel_sq = float(cfg["energy"]["max_relative_squared_error"])

    def recorder(self):
        from csasr.lss.sites import DecoderPostCrossAttnRecorder
        return DecoderPostCrossAttnRecorder(self.bundle, list(LAYERS), keep_last_only=True)

    @staticmethod
    def site_states(rec) -> dict:
        out = {}
        for L in LAYERS:
            out[SITE_KEYS[(L, SITES[0])]] = rec.q_states[L][0, -1].float().cpu()
            out[SITE_KEYS[(L, SITES[1])]] = rec.states[L][0, -1].float().cpu()
        return out

    def step(self, br, new, *, attention, hook=None):
        from csasr.inference_cf.loc0_sites import Composite, assert_no_any_site_hooks
        rec = self.recorder()
        logits, _, _ = br.step(new, capture_layer=None, attention=attention, hook=Composite(hook, rec))
        self.c["forwards"] += 1
        assert_no_any_site_hooks(self.bundle)
        return logits, self.site_states(rec), hook

    def pulse(self, br, L, new, query, layer, site, v_fn):
        """One single pulse from the cropped pre-query cache; returns logits, sites, hook, info."""
        from csasr.inference_cf.loc0_sites import SelfAttnResidualInterventionHook, cross_pulse_hook, pulse_action
        br.crop(L)
        info: dict = {}
        act = pulse_action(query, v_fn, self.e_star, info, self.max_rel_sq, self.p2r.solve_scale, self.p2r.scaled_direction)
        hook = cross_pulse_hook(self.bundle, layer, act, self.nfp) if site == SITES[1] else \
            SelfAttnResidualInterventionHook(self.bundle, layer, act, num_forced_prefix=self.nfp)
        logits, st, hook = self.step(br, new, attention=True, hook=hook)
        self.c["pulse_calls"] += 1
        self.c["solver_evals"] += int(info.get("evals", 0) or 0)
        return logits, st, hook, info


def _rec_of(hook) -> dict | None:
    if not hook.records:
        return None
    r = hook.records[-1]
    return r.to_dict() if hasattr(r, "to_dict") else dict(r)


def cell_record(eng: Engine, arm: dict, hook, info: dict, before: dict, after: dict, logits, base_logits) -> tuple[dict, dict]:
    """Per-cell compact record + arrays; integrity failures listed (stage INVALID if any)."""
    import torch
    sk = SITE_KEYS[(arm["layer"], arm["site"])]
    audit = _rec_of(hook)
    steered = bool(audit and audit["steered"])
    e = eng.e_star
    sol = {k: v for k, v in info.items() if k not in ("v", "_proposed")}
    rec = {"arm": arm["id"], "site_key": sk, "steered": steered, "solver": sol, "no_edit_reason": None if steered else info.get("no_edit_reason"),
           "pre_norm": audit["pre_norm"] if audit else None, "edit_norm": audit["edit_norm"] if steered else 0.0,
           "argmax": None, "integrity_failures": []}
    arrays = {}
    if steered:
        cons = after[sk].double() - before[sk].double()
        rec["consumed_edit_norm"] = float(torch.linalg.vector_norm(cons))
        prop = info["_proposed"].double() - before[sk].double()
        rec["proposed_edit_norm"] = float(torch.linalg.vector_norm(prop))
        rec["relative_sq_energy"] = rec["edit_norm"] ** 2 / float(before[sk].double().norm() ** 2)
        rec["relative_magnitude"] = rec["edit_norm"] / float(before[sk].double().norm())
        chk = {"proposed_within_2pct": abs(rec["edit_norm"] / e - 1) <= 0.02, "consumed_within_2pct": abs(rec["consumed_edit_norm"] / e - 1) <= 0.02,
               "rel_sq_err": abs(rec["edit_norm"] ** 2 / e ** 2 - 1) <= eng.cfg["energy"]["max_relative_squared_error"],
               "consumed_vs_proposed": abs(rec["consumed_edit_norm"] - rec["edit_norm"]) <= eng.cfg["energy"]["consumed_vs_proposed_relative_norm_tolerance"] * rec["edit_norm"],
               "solver_matches_hook": abs(sol.get("edit_norm", -1) - rec["edit_norm"]) <= 1e-6 * max(1.0, rec["edit_norm"])}
        rec["checks"] = chk
        rec["integrity_failures"] += [k for k, v in chk.items() if not v]
        rec["valid"] = all(chk.values())
        arrays["proposed"] = info["_proposed"].numpy()
        arrays["consumed"] = after[sk].numpy()
        arrays["solver_v"] = info["v"].numpy()
    else:
        rec["valid"] = False
        if rec["no_edit_reason"] not in NO_EDIT_REASONS:
            rec["integrity_failures"].append(f"unexpected_no_edit:{rec['no_edit_reason']}")
        if not torch.equal(logits, base_logits):
            rec["integrity_failures"].append("no_edit_logits_not_baseline")
    if not bool(torch.isfinite(logits).all()):
        rec["integrity_failures"].append("nonfinite_logits")
    return rec, arrays


def cmd_run(args) -> None:
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.core_p1 import direction, processed_argmax
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.inference_cf.unique import array_hash
    from csasr.inference_cf.loc0_sites import cache_fingerprint, assert_no_any_site_hooks
    from csasr.lss.sites import num_forced_prefix_from
    from csasr.models.whisper import batch_model_inputs, load_whisper
    from csasr.utils.config import load_config
    import experiments.inference_cf_p2dir as p2dir
    import experiments.inference_cf_st_loc0_calibrate as calib
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
    tok = bundle.processor.tokenizer
    if tokenizer_partition(tok)["hash"] != plan["partition_hash"]:
        raise ValueError("partition")
    gen = bundle.model.generation_config
    if digest({"suppress": list(gen.suppress_tokens or []), "begin": list(gen.begin_suppress_tokens or [])}) != plan["suppression_hash"]:
        raise ValueError("suppression")
    nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    if nfp != 4:
        raise ValueError("forced prefix")
    counters = {"forwards": 0, "pulse_calls": 0, "solver_evals": 0, "zero_dose_checks": 0, "restore_checks": 0, "cache_fingerprints": 0,
                "autograd_calls": 0, "LID_calls": 0, "encoder_passes": 0}
    eng = Engine(bundle, cfg, nfp, counters)
    rp = plan["runtime"]
    Q, Ut = rp["runtime_queries"], rp["utterances"]
    by_u: dict = {}
    for j, q in enumerate(Q):
        by_u.setdefault(q["utterance_id"], []).append((j, q))
    arms = {a["id"]: a for a in plan["arms"]}
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "start_unix": t_start,
               "status": "running", "phase": "A", "counters": counters}
    for d in ("calibration", "barrier", "pulses"):
        (out / d).mkdir(parents=True, exist_ok=True)
    atomic_json(out / "runtime.json", runtime)
    torch.cuda.reset_peak_memory_stats()
    ext = np.load(ROOT / EXTRACT / "states.npz")
    HBx, HEx = ext["H_B"], ext["H_E"]
    encs: dict = {}
    failures: list = []

    def encode(u):
        inputs = batch_model_inputs(bundle, [u["audio_path"]])
        with torch.inference_mode():
            h = bundle.model.model.encoder(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state
        counters["encoder_passes"] += 1
        return BaseModelOutput(last_hidden_state=h)

    def hist_row(i):
        r = json.loads((ROOT / EXP1 / "rows" / f"{i:03d}.json").read_text())
        with np.load(ROOT / EXP1 / "rows" / f"{i:03d}.npz") as z:
            vec = {k: z[k] for k in z.files}
        with np.load(ROOT / EXP1 / "rows" / f"{i:03d}_logits.npz") as z:
            lg = {k: z[k] for k in z.files}
        return r, vec, lg

    try:
        # ===================== A: passive paired extraction at four sites =====================
        t_a = time.time()
        SB = {sk: np.zeros((180, 1280), np.float32) for sk in SITE_KEYS.values()}
        SE = {sk: np.zeros((180, 1280), np.float32) for sk in SITE_KEYS.values()}
        base_logits, base_argmax, idA = {}, {}, {}
        for i, u in enumerate(Ut):
            uid = u["utterance_id"]
            enc = encode(u)
            encs[uid] = enc
            hrow, hvec, hlg = hist_row(i)
            toks = u["baseline_content_tokens"]
            jt = {q["t"]: j for j, q in by_u[uid]}
            B, E = eng.p2r.DiagBranch(bundle, enc, CB, "B"), eng.p2r.DiagBranch(bundle, enc, CE, "E")
            new, new_e = list(CB), list(CE)
            for t in range(max(jt) + 1):
                lb, sb, _ = eng.step(B, new, attention=True)
                _, se, _ = eng.step(E, new_e, attention=False)
                if t in jt:
                    j = jt[t]
                    for sk in SITE_KEYS.values():
                        SB[sk][j], SE[sk][j] = sb[sk].numpy(), se[sk].numpy()
                    base_logits[j] = lb
                    base_argmax[j] = processed_argmax(lb, t, eng.suppress, eng.begin)
                    hl = torch.from_numpy(np.asarray(hlg[f"t{t}_none"], dtype=np.int16).copy()).view(torch.bfloat16).float()
                    idA[j] = {"hb_bitwise": bool(np.array_equal(SB["L16_CROSS"][j], HBx[j])), "he_bitwise": bool(np.array_equal(SE["L16_CROSS"][j], HEx[j])),
                              "none_logits_bitwise": bool(torch.equal(lb, hl)), "argmax_equals_historical": base_argmax[j] == hrow["positions"][str(t)]["identity"]["argmax"],
                              "finite": bool(torch.isfinite(lb).all()) and all(bool(np.isfinite(SB[sk][j]).all()) for sk in SITE_KEYS.values())}
                    if not all(idA[j].values()):
                        failures.append(f"A:{uid}:t{t}:{[k for k, v in idA[j].items() if not v]}")
                if t < len(toks):
                    new, new_e = [toks[t]], [toks[t]]
            print(f"ST-LOC0 A {i + 1}/80 {uid}", flush=True)
        runtime["phaseA_sec"] = time.time() - t_a
        runtime["phaseA_identity"] = {"rows": len(idA), "all_ok": all(all(v.values()) for v in idA.values())}
        for sk in SITE_KEYS.values():
            np.savez(out / "calibration" / f"states_{sk}.npz", H_B=SB[sk], H_E=SE[sk])
        np.savez_compressed(out / "calibration" / "baseline_logits.npz", **{f"q{j:03d}": p2dir.pack_bf16(base_logits[j]) for j in range(180)})
        if failures:
            raise RuntimeError("phase A historical identity failed: " + "; ".join(failures[:10]))
        # ===================== B: offline calibration (V2 folds), V1 per query, random, D2 =====================
        t_b = time.time()
        Pfull = json.loads((ROOT / PANEL).read_text())
        calib_in = {k: Pfull[k] for k in calib.ALLOWED}
        fold_summ, fold_vec = {}, {}
        hist_folds = json.loads((ROOT / FOLDS).read_text())["folds"]
        for sk in SITE_KEYS.values():
            folds = calib.fit_site(calib_in, SB[sk])
            fold_summ[sk] = calib.save_site(out / "calibration", sk, folds)
            fold_vec[sk] = {d: f["vector"] for d, f in folds.items()}
        histcheck = {}
        refit_l16 = {d: v for d, v in fold_vec["L16_CROSS"].items() if v is not None}
        if sorted(hist_folds) != sorted(fold_vec["L16_CROSS"]) or any(hf["status"] != "ok" for hf in hist_folds.values()):
            raise RuntimeError("historical fold set")
        for d, hf in hist_folds.items():
            hv = np.load(ROOT / hf["vector_path"]) if hf["status"] == "ok" else None
            nv = fold_vec["L16_CROSS"].get(d)
            histcheck[d] = (hv is None and nv is None) or (hv is not None and nv is not None and float(np.max(np.abs(hv - nv))) <= cfg["historical_reproduction"]["D1_vector_max_abs"]
                                                          and fold_summ["L16_CROSS"][d]["selected_index"] == hf.get("selected_index") and fold_summ["L16_CROSS"][d]["status"] == hf["status"])
        if not all(histcheck.values()):
            raise RuntimeError(f"historical D1 fold reproduction failed: {[d for d, v in histcheck.items() if not v]}")
        # spec section 4: L16 DG-02 REUSES the original clean fold files (refit above is provenance only)
        hist_vec = {}
        for d, hf in hist_folds.items():
            hv = np.load(ROOT / hf["vector_path"]).astype(np.float32)
            if file_hash(ROOT / hf["vector_path"]) != hf["vector_file_sha256"] or array_hash(hv) != hf["vector_sha256"]:
                raise RuntimeError(f"historical D1 vector hash mismatch {d}")
            hist_vec[d] = hv
        fold_vec["L16_CROSS"] = hist_vec
        np.savez(out / "calibration" / "folds_L16_CROSS_historical_vectors.npz", **hist_vec)
        refit_maxabs = {d: float(np.max(np.abs(hist_vec[d] - fold_summ_vec))) for d, fold_summ_vec in refit_l16.items()}
        V1, v1rec = {}, {}
        for j, q in enumerate(Q):
            for sk in SITE_KEYS.values():
                r = direction(torch.from_numpy(SE[sk][j].copy()), torch.from_numpy(SB[sk][j].copy()))
                v1rec[f"q{j:03d}_{sk}"] = {"status": r["status"], "norm": r["norm"], "sha256": None if r["d"] is None else array_hash(r["d"].numpy())}
                if r["d"] is not None:
                    V1[f"q{j:03d}_{sk}"] = r["d"].numpy()
        rv = random_vectors(cfg)
        D2, d0hist = {}, {}
        for i, u in enumerate(Ut):
            hrow, hvec, _ = hist_row(i)
            for j, q in by_u[u["utterance_id"]]:
                t = q["t"]
                drec = hrow["positions"][str(t)]["directions"]
                D2[f"q{j:03d}"] = hvec[f"t{t}_D2"]
                if array_hash(hvec[f"t{t}_D2"].astype(np.float32)) != drec["D2"]["sha256"]:
                    raise RuntimeError(f"historical D2 vector hash mismatch q{j}")
                d0hist[j] = drec["D0"]["sha256"] == v1rec[f"q{j:03d}_L16_CROSS"]["sha256"]
        if not all(d0hist.values()):
            raise RuntimeError("historical D0 vector reproduction failed")
        np.savez(out / "calibration" / "v1_vectors.npz", **V1)
        np.savez(out / "calibration" / "random_vectors.npz", **rv)
        np.savez(out / "calibration" / "d2_vectors.npz", **D2)
        atomic_json(out / "calibration" / "v1_records.json", v1rec)
        cal_files = sorted(p for p in (out / "calibration").iterdir() if p.is_file() and p.name not in ("calibration_seal.json", "GO.json"))
        seal = {"schema": SCHEMA + "_calibration_seal", "manifest_hash": m["manifest_hash"], "files": {p.name: file_hash(p) for p in cal_files},
                "historical_D1_folds_reproduced": True, "historical_D0_vectors_reproduced": True,
                "L16_CROSS_v_unq_source": "historical folds_run1 vectors (hash-verified); refit provenance maxabs per fold below",
                "L16_CROSS_refit_maxabs": refit_maxabs,
                "fold_status": {sk: {d: s["status"] for d, s in v.items()} for sk, v in fold_summ.items()},
                "V1_invalid": sum(1 for r in v1rec.values() if r["sha256"] is None), "created_unix": time.time()}
        seal["seal_hash"] = digest(seal)
        atomic_json(out / "calibration" / "calibration_seal.json", seal)
        runtime["phaseB_sec"] = time.time() - t_b
        runtime["phase"] = "WAIT_GO"
        atomic_json(out / "runtime.json", runtime)
        print(f"ST-LOC0 calibration sealed {seal['seal_hash']}; waiting for GO", flush=True)
        t_w = time.time()
        go = out / "calibration" / "GO.json"
        while not go.exists():
            if time.time() - t_w > GO_TIMEOUT_SEC:
                raise RuntimeError("GO not received: calibration seal not confirmed on remote")
            time.sleep(10)
        g = json.loads(go.read_text())
        if g.get("calibration_seal_hash") != seal["seal_hash"] or subprocess.run(["git", "cat-file", "-e", g["remote_commit"]], cwd=ROOT).returncode != 0:
            raise RuntimeError("GO does not confirm this calibration seal")
        runtime["wait_sec"] = time.time() - t_w
        runtime["go"] = g
        # ===================== C: pulses (C1 barrier, then C2) =====================

        def v_fn_for(arm, j, dlg):
            fam, sk = arm["family"], SITE_KEYS[(arm["layer"], arm["site"])]
            if fam == "v_prompt":
                v = V1.get(f"q{j:03d}_{sk}")
                return (lambda r: (None, "tiny_or_nonfinite_V1_delta")) if v is None else (lambda r, v=v: (torch.from_numpy((arm["sign"] * v).astype(np.float32)), "ok"))
            if fam == "v_unq":
                v = fold_vec[sk].get(dlg)
                return (lambda r: (None, "invalid_new_site_D1_fold")) if v is None else (lambda r, v=v: (torch.from_numpy((arm["sign"] * v).astype(np.float32)), "ok"))
            if fam == "random":
                return lambda r: (torch.from_numpy(rv[arm["id"]].copy()), "ok")
            if fam == "D2":
                return lambda r: (torch.from_numpy(D2[f"q{j:03d}"].astype(np.float32)), "ok")
            raise ValueError(fam)

        def run_pass(name, arm_ids, barrier):
            t_p = time.time()
            for i, u in enumerate(Ut):
                uid = u["utterance_id"]
                enc = encs[uid]
                toks = u["baseline_content_tokens"]
                jt = {q["t"]: (j, q) for j, q in by_u[uid]}
                hrow, hvec, hlg = hist_row(i) if barrier else (None, None, None)
                B = eng.p2r.DiagBranch(bundle, enc, CB, "B")
                new = list(CB)
                rows, logits_out, arrays_out = {}, {}, {}
                for t in range(max(jt) + 1):
                    L = B.length
                    if t in jt:
                        j, q = jt[t]
                        query = q["absolute_query"]
                        fp0 = cache_fingerprint(B.cache, L) if B.cache is not None else "empty"
                        counters["cache_fingerprints"] += 1
                        lb, sb, _ = eng.step(B, new, attention=True)
                        if L + len(new) - 1 != query:
                            raise RuntimeError("query index mismatch")
                        clean_ok = bool(torch.equal(lb, base_logits[j])) and all(np.array_equal(sb[sk].numpy(), SB[sk][j]) for sk in SITE_KEYS.values())
                        prow = {"q": j, "t": t, "query": query, "clean_replay_bitwise": clean_ok, "cells": {}}
                        if not clean_ok:
                            failures.append(f"{name}:{uid}:t{t}:clean_replay")
                        for sk in SITE_KEYS.values():
                            arrays_out[f"q{j:03d}_before_{sk}"] = sb[sk].numpy()
                        if barrier:
                            prow["zero_dose"] = {}
                            for L_, s_ in SITE_KEYS:
                                la, _, _, _ = eng.pulse(B, L, new, query, L_, s_, lambda r: (None, "zero_dose"))
                                counters["zero_dose_checks"] += 1
                                prow["zero_dose"][SITE_KEYS[(L_, s_)]] = bool(torch.equal(la, lb))
                                if not torch.equal(la, lb):
                                    failures.append(f"{name}:{uid}:t{t}:zero_dose:{SITE_KEYS[(L_, s_)]}")
                        for aid in arm_ids:
                            arm = arms[aid]
                            la, sa, hook, info = eng.pulse(B, L, new, query, arm["layer"], arm["site"], v_fn_for(arm, j, q["dialogue_id"]))
                            rec, arr = cell_record(eng, arm, hook, info, sb, sa, la, lb)
                            B.crop(L)
                            rec["cache_prefix_unchanged_after_arm"] = (cache_fingerprint(B.cache, L) if B.cache is not None else "empty") == fp0
                            rec["cache_positions_ok"] = B.positions == list(range(L)) and B.cache.get_seq_length() == L
                            counters["cache_fingerprints"] += 1
                            if not (rec["cache_prefix_unchanged_after_arm"] and rec["cache_positions_ok"]):
                                rec["integrity_failures"].append("cache_lineage")
                            rec["argmax"] = processed_argmax(la, t, eng.suppress, eng.begin)
                            if barrier:
                                h = HIST_NAME[aid]
                                hl = torch.from_numpy(np.asarray(hlg[f"t{t}_{h}"], dtype=np.int16).copy()).view(torch.bfloat16).float()
                                hs = hrow["positions"][str(t)]["arms"][h]["solver"]
                                bar = {"logits_bitwise": bool(torch.equal(la, hl)),
                                       "consumed_state_bitwise": bool(np.array_equal(sa["L16_CROSS"].numpy(), hvec[f"t{t}_post_{h}"])),
                                       "solver_s": hs.get("status") == info.get("status") and abs(float(info.get("s", 0)) - float(hs.get("s", 0))) <= 1e-9 * max(1.0, abs(float(hs.get("s", 0)))),
                                       "edit_norm_equal": rec["edit_norm"] == hrow["positions"][str(t)]["arms"][h]["edit_norm"]}
                                rec["historical_barrier"] = bar
                                if not all(bar.values()):
                                    failures.append(f"{name}:{uid}:t{t}:{aid}:{[k for k, v in bar.items() if not v]}")
                            if rec["integrity_failures"]:
                                failures.append(f"{name}:{uid}:t{t}:{aid}:{rec['integrity_failures']}")
                            prow["cells"][aid] = rec
                            logits_out[f"q{j:03d}_{aid}"] = p2dir.pack_bf16(la)
                            for k_, a_ in arr.items():
                                arrays_out[f"q{j:03d}_{aid}_{k_}"] = a_
                        B.crop(L)
                        fp1 = cache_fingerprint(B.cache, L) if B.cache is not None else "empty"
                        counters["cache_fingerprints"] += 1
                        lb2, sb2, _ = eng.step(B, new, attention=True)
                        counters["restore_checks"] += 1
                        prow["restore_bitwise"] = bool(torch.equal(lb2, lb)) and all(torch.equal(sb2[sk], sb[sk]) for sk in SITE_KEYS.values())
                        prow["cache_prefix_unchanged"] = fp0 == fp1
                        if not (prow["restore_bitwise"] and prow["cache_prefix_unchanged"]):
                            failures.append(f"{name}:{uid}:t{t}:restore/cache")
                        rows[str(t)] = prow
                    else:
                        eng.step(B, new, attention=True)
                    if t < len(toks):
                        new = [toks[t]]
                np.savez_compressed(out / name / f"{i:03d}_logits.npz", **logits_out)
                np.savez_compressed(out / name / f"{i:03d}_arrays.npz", **arrays_out)
                atomic_json(out / name / f"{i:03d}.json", {"identity": uid, "utterance_index": i, "manifest_hash": m["manifest_hash"], "positions": rows,
                                                         "logits_sha256": file_hash(out / name / f"{i:03d}_logits.npz"),
                                                         "arrays_sha256": file_hash(out / name / f"{i:03d}_arrays.npz"), "status": "ok"})
                print(f"ST-LOC0 {name} {i + 1}/80 {uid} failures {len(failures)}", flush=True)
            runtime[f"{name}_sec"] = time.time() - t_p

        runtime["phase"] = "C1_barrier"
        atomic_json(out / "runtime.json", runtime)
        run_pass("barrier", list(BARRIER_ARMS), True)
        runtime["barrier"] = {"pass": not failures, "failures": list(failures[:50])}
        atomic_json(out / "runtime.json", runtime)
        if failures:
            raise RuntimeError("historical barrier failed (no new-site pulses run): " + "; ".join(failures[:10]))
        runtime["phase"] = "C2_pulses"
        main_arms = [a["id"] for a in plan["arms"] if a["id"] not in BARRIER_ARMS]
        if len(main_arms) != 18:
            raise RuntimeError("arm table")
        run_pass("pulses", main_arms, False)
        status = "completed" if not failures else "failed"
        if failures:
            runtime["failure"] = {"reason": "pulse integrity: " + "; ".join(failures[:20])}
    except Exception as exc:
        import traceback
        status = "failed"
        runtime["failure"] = {"reason": repr(exc), "traceback": traceback.format_exc()}
    assert_no_any_site_hooks(bundle)
    runtime.update(status=status, failures=failures[:200], n_failures=len(failures), end_unix=time.time(),
                   peak_alloc=int(torch.cuda.max_memory_allocated()), peak_reserved=int(torch.cuda.max_memory_reserved()),
                   model_grads_none=all(p.grad is None for p in bundle.model.parameters()))
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)
    if status != "completed":
        raise SystemExit("ST-LOC0 run failed: " + runtime["failure"]["reason"])


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
