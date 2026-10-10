#!/usr/bin/env python
"""P2-SEL runner (GPU; separate inference_cf development diagnostic). Frozen contract:
docs/inference_cf/P2_SEL_SPEC.md, configs/inference_cf/p2_sel.json, P2_SEL_CODEX_DESIGN.md.

``s1``        180-position gate-coupled single-pulse screen on the frozen P2-RJ/P2-DIR states.
              Reuses (after bitwise identity checks): P2-DIR extracted B/E states and NONE logits,
              the sealed P2-DIR D2 vectors (no readout recomputation, zero autograd), and the P2-R
              run3 B-branch gate rows (E, R_B, g = E*R_B) joined by (utterance, absolute query).
              Pass A: C1 OLD-GATED and C2 D2-GATED with the unchanged P2 ``_edit_hook``
              (alpha 2, gain g, phi id, NormPreserve). Then the pooled reference-free C3 target
              e_broad = sqrt(sum ||edit_C2||^2 / 180). Pass B: C3 D2-BROAD (alpha 2, g = 1, direction
              (lambda/2) d with lambda from the unchanged P2-R solver). Reads NO reference information.
``manifest``  immutable stage manifest (CPU).

No new direction, layer, alpha, gate, localizer or dose map exists here.
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
import torch

from csasr.inference_cf.core import atomic_json, digest, file_hash
from csasr.inference_cf.core_p1 import direction as old_direction, processed_argmax
from csasr.inference_cf.core_r2 import conflict_from_logits
from csasr.lss.sites import DecoderPostCrossAttnInterventionHook, assert_no_site_hooks
import experiments.inference_cf_cached as cached
import experiments.inference_cf_p2dir as p2dir
import experiments.inference_cf_p2r as p2r

SCHEMA = "p2_sel_v1"
CONFIG = "configs/inference_cf/p2_sel.json"
P2DIR_RUN = "results/inference_cf/p2dir/exp1_run1"
P2R_RUN = "results/inference_cf/p2r/run3"
CONSTRUCTION = "results/inference_cf/p2dir/construction_population.json"
ALPHA = 2.0
LAYER = 16
N_ELIGIBLE = 180
GATED = ("C1", "C2")


# ---- frozen pieces ---------------------------------------------------------------------------------

def gated_hook(bundle, query: int, g: float, d: torch.Tensor | None, nfp: int):
    """The unchanged P2 deployment hook: alpha 2, gain g (phi id), NormPreserve; zero gain or no
    direction -> exact no-edit."""
    applied = bool(ALPHA * g > 0 and d is not None and query >= nfp)
    return cached._edit_hook(bundle, LAYER, query, ALPHA, g if applied else 0.0, d if applied else None, nfp), applied


def broad_target(c2_edit_norms: list[float], n: int = N_ELIGIBLE) -> float:
    """Pooled reference-free C3 chord target sqrt(Q2 / N); zero if Q2 == 0."""
    if len(c2_edit_norms) != n:
        raise ValueError("C2 edit energies must cover every frozen row")
    q2 = float(sum(float(e) ** 2 for e in c2_edit_norms))
    return math.sqrt(q2 / n) if q2 > 0 else 0.0


def broad_hook(bundle, query: int, v: torch.Tensor, target: float, nfp: int, info: dict):
    """C3: alpha 2, gain exactly 1, direction tensor (lambda/2) * unit D2, lambda from the unchanged
    P2-R ``solve_scale`` on the live pre-edit site (8 evaluations max). target 0 -> exact no-edit."""
    def action_fn(q, u_source, r, abs_pos):
        n = r.shape[1]
        g = torch.zeros((1, n), device=r.device, dtype=r.dtype)
        dirs = torch.zeros((1, n, r.shape[-1]), device=r.device, dtype=r.dtype)
        hit = (abs_pos == query).nonzero().flatten()
        if hit.numel() and target > 0:
            k = int(hit[0])
            sol = p2r.solve_scale(r[0, k], v, target)
            info.update(sol)
            if sol["status"] == "ok":
                vv = v.detach().cpu().double()
                vv = vv / torch.linalg.vector_norm(vv)
                g[0, k] = 1.0
                dirs[0, k] = p2r.scaled_direction(vv, sol["s"] / ALPHA, r)
        elif hit.numel():
            info["status"] = "zero_target"
        return g, dirs
    return DecoderPostCrossAttnInterventionHook(bundle, LAYER, None, alpha=ALPHA, num_forced_prefix=nfp,
                                                action_fn=action_fn, norm_preserve=True, record=True,
                                                record_last_only=True)


def join_gates(construction: dict, p2r_rows: dict, p2dir_rows: dict) -> dict:
    """(uid, t) -> frozen run3 B-branch gate step; exact unique (utterance, absolute query) join."""
    out = {}
    for p in construction["positions"]:
        uid, t = p["utterance_id"], int(p["t"])
        query = p2dir_rows[uid]["positions"][str(t)]["query"]
        m = [s for s in p2r_rows[uid]["current"]["steps"] if s["query"] == query]
        if len(m) != 1 or m[0]["t"] != t:
            raise ValueError(f"gate join failed for {uid} t={t}")
        s = m[0]
        out[(uid, t)] = {"query": query, "E": s["E"], "R_B": s["R_B"], "g": s["g"], "fb": s["fb"], "window": s["window"]}
    return out


def _arm_record(hook, la, h_cons, hb, t, ctx, extra=None) -> dict:
    audit = hook.records[-1].to_dict() if hook.records else None
    steered = bool(audit and audit["steered"] and audit["edit_norm"] > 0)
    rec = {"steered": steered, "edit_norm": audit["edit_norm"] if steered else 0.0,
           "pre_norm": audit["pre_norm"] if audit else None,
           "argmax": processed_argmax(la, t, ctx.suppress, ctx.begin)}
    if steered:
        rec["consumed_edit_norm"] = float(torch.linalg.vector_norm(h_cons.double() - hb.double()))
        rec["relative_sq_energy"] = rec["edit_norm"] ** 2 / float(hb.double().norm() ** 2)
    if extra:
        rec.update(extra)
    return rec


# ---- S1 passes ---------------------------------------------------------------------------------------

def s1_pass_a(ctx, *, encoded, tokens, ts, gates, ref_vecs, ref_none) -> tuple[dict, dict, dict]:
    """C1 OLD-GATED and C2 D2-GATED single pulses from the identical unedited B cache."""
    B, E = p2r.DiagBranch(ctx.bundle, encoded, ctx.cB, "B"), p2r.DiagBranch(ctx.bundle, encoded, ctx.cE, "E")
    new, new_e = list(ctx.cB), list(ctx.cE)
    recs, vecs, logits = {}, {}, {}
    for t in range(max(ts) + 1):
        L = B.length
        lb, _, hb = B.step(new, capture_layer=ctx.layer, attention=True)
        _, _, he = E.step(new_e, capture_layer=ctx.layer, attention=False)
        assert_no_site_hooks(ctx.bundle)
        if t in ts:
            pre = f"t{t}_"
            gate = gates[t]
            query = L + len(new) - 1
            d0 = old_direction(he, hb)["d"]
            d2_np = ref_vecs.get(pre + "D2")
            d2 = None if d2_np is None else torch.from_numpy(d2_np.copy())
            bc = conflict_from_logits(lb, ctx.partition)
            rec = {"t": t, "query": query, "gate": gate,
                   "identity": {"query": query == gate["query"],
                                "hb_bitwise_p2dir": bool(np.array_equal(hb.numpy(), ref_vecs[pre + "hb"])),
                                "he_bitwise_p2dir": bool(np.array_equal(he.numpy(), ref_vecs[pre + "he"])),
                                "none_logits_bitwise_p2dir": bool(np.array_equal(p2dir.pack_bf16(lb), ref_none[pre + "none"])),
                                "d0_bitwise_p2dir": d0 is not None and bool(np.array_equal(d0.numpy(), ref_vecs.get(pre + "D0"))),
                                "R_B_recomputed": bc["R"], "R_B_abs_diff": abs(bc["R"] - gate["R_B"]),
                                "g_product_abs_diff": abs(gate["E"] * gate["R_B"] - gate["g"])},
                   "directions": {"C1": {"status": "ok" if d0 is not None else "invalid"},
                                  "C2": {"status": "ok" if d2 is not None else "invalid"}},
                   "arms": {}}
            for arm, d in (("C1", d0), ("C2", d2)):
                B.crop(L)
                hook, applied = gated_hook(ctx.bundle, query, float(gate["g"]), d, ctx.nfp)
                la, _, h_cons = B.step(new, capture_layer=ctx.layer, attention=True, hook=hook)
                assert_no_site_hooks(ctx.bundle)
                a = _arm_record(hook, la, h_cons, hb, t, ctx, {"applied": applied, "gain": float(gate["g"]) if applied else 0.0,
                                                                "dose": ALPHA * float(gate["g"]) if applied else 0.0})
                a["zero_gain_bitwise_none"] = None if applied else bool(torch.equal(la, lb))
                rec["arms"][arm] = a
                logits[pre + arm] = p2dir.pack_bf16(la)
                if a["steered"]:
                    vecs[pre + "post_" + arm] = h_cons.numpy()
            B.crop(L)
            lb2, _, hb2 = B.step(new, capture_layer=ctx.layer, attention=True)
            rec["restore_bitwise"] = bool(torch.equal(lb2, lb) and torch.equal(hb2, hb))
            recs[str(t)] = rec
        if t < len(tokens):
            new, new_e = [tokens[t]], [tokens[t]]
    return recs, vecs, logits


def s1_pass_b(ctx, *, encoded, tokens, ts, target, ref_vecs) -> tuple[dict, dict, dict]:
    """C3 D2-BROAD single pulses at the pooled chord target."""
    B = p2r.DiagBranch(ctx.bundle, encoded, ctx.cB, "B")
    new = list(ctx.cB)
    recs, vecs, logits = {}, {}, {}
    for t in range(max(ts) + 1):
        L = B.length
        lb, _, hb = B.step(new, capture_layer=ctx.layer, attention=True)
        if t in ts:
            pre = f"t{t}_"
            query = L + len(new) - 1
            d2 = torch.from_numpy(ref_vecs[pre + "D2"].copy())
            B.crop(L)
            info: dict = {}
            hook = broad_hook(ctx.bundle, query, d2, target, ctx.nfp, info)
            la, _, h_cons = B.step(new, capture_layer=ctx.layer, attention=True, hook=hook)
            assert_no_site_hooks(ctx.bundle)
            a = _arm_record(hook, la, h_cons, hb, t, ctx, {"solver": info, "target": target})
            a["zero_bitwise_none"] = None if a["steered"] else bool(torch.equal(la, lb))
            logits[pre + "C3"] = p2dir.pack_bf16(la)
            if a["steered"]:
                vecs[pre + "post_C3"] = h_cons.numpy()
            B.crop(L)
            lb2, _, hb2 = B.step(new, capture_layer=ctx.layer, attention=True)
            recs[str(t)] = {"t": t, "query": query, "arm": a,
                            "hb_bitwise_p2dir": bool(np.array_equal(hb.numpy(), ref_vecs[pre + "hb"])),
                            "restore_bitwise": bool(torch.equal(lb2, lb) and torch.equal(hb2, hb))}
        if t < len(tokens):
            new = [tokens[t]]
    return recs, vecs, logits


# ---- manifest / job entry -----------------------------------------------------------------------------

SOURCES = ("docs/inference_cf/P2_SEL_SPEC.md", "docs/inference_cf/P2_SEL_CODEX_DESIGN.md", CONFIG,
           "docs/inference_cf/P2_SEL_MINI_PANEL.json", "docs/inference_cf/P2_SEL_PRE_RUN_AUDIT.md",
           "results/inference_cf/p2rj/positions.json", CONSTRUCTION,
           "src/csasr/inference_cf/readout.py", "src/csasr/inference_cf/directions.py",
           "src/csasr/inference_cf/core_p1.py", "src/csasr/inference_cf/core_r2.py", "src/csasr/inference_cf/broad.py",
           "src/csasr/lss/sites.py", "src/csasr/models/hooks.py", "experiments/inference_cf_cached.py",
           "experiments/inference_cf_p2r.py", "experiments/inference_cf_p2dir.py",
           "experiments/inference_cf_p2sel.py", "experiments/inference_cf_p2sel_analyze.py",
           "experiments/inference_cf_p2sel_audit.py", "slurm/inference_cf_p2sel.sbatch",
           f"{P2DIR_RUN}/manifest.json", f"{P2DIR_RUN}/directions_sealed.json", f"{P2R_RUN}/manifest.json")


def cmd_manifest(args) -> None:
    import experiments.inference_cf_p2dir_prepare as prep
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition
    cfg = json.loads((ROOT / CONFIG).read_text())
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--", *SOURCES],
                                    cwd=ROOT, text=True).strip()
    if dirty:
        raise ValueError("commit P2-SEL sources before preparing a manifest:\n" + dirty)
    pre = json.loads((ROOT / "results/inference_cf/p2sel/prerun_audit.json").read_text())
    if pre["verdict"] != "PASS_TO_P2_SEL_S1":
        raise ValueError("pre-run audit did not pass")
    con = json.loads((ROOT / CONSTRUCTION).read_text())
    partition = tokenizer_partition(WhisperProcessor.from_pretrained(prep.MODEL, local_files_only=True).tokenizer)
    gen = GenerationConfig.from_pretrained(prep.MODEL, local_files_only=True)
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    man = {"schema": SCHEMA, "stage": args.stage, "git_commit": git("rev-parse", "HEAD"),
           "git_tree": git("rev-parse", "HEAD^{tree}"), "config": CONFIG, "config_hash": digest(cfg),
           "construction": CONSTRUCTION, "construction_hash": con["construction_hash"],
           "p2dir_run": P2DIR_RUN, "p2r_run": P2R_RUN, "partition_hash": partition["hash"],
           "suppression_hash": digest({"suppress": list(gen.suppress_tokens or []),
                                       "begin": list(gen.begin_suppress_tokens or [])}),
           "prerun_audit_sha256": file_hash(ROOT / "results/inference_cf/p2sel/prerun_audit.json"),
           "role": "D-dev-select", "firewall": prep.firewall(con["utterances"]), "environment": prep.environment(),
           "model": {"dir": str(prep.MODEL), "files": prep.model_hashes()},
           "frozen": cfg["frozen_intervention"], "sources": {p: file_hash(ROOT / p) for p in SOURCES},
           "created_unix": time.time()}
    man["manifest_hash"] = digest(man)
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("P2-SEL manifest exists; never overwrite a frozen run")
    atomic_json(out / "manifest.json", man)
    print(json.dumps({"stage": args.stage, "manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}))


def verify_manifest(out: Path, stage: str) -> dict:
    m = json.loads((out / "manifest.json").read_text())
    if m.get("schema") != SCHEMA or m.get("stage") != stage or \
            digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError(f"invalid P2-SEL {stage} manifest")
    for path, expected in m["sources"].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"source changed after freeze: {path}")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != m["git_commit"]:
        raise ValueError("Git commit changed after P2-SEL manifest freeze")
    return m


def load_p2dir_rows(con: dict) -> tuple[dict, dict, dict]:
    sealed = json.loads((ROOT / P2DIR_RUN / "directions_sealed.json").read_text())
    rows, vecs, nones = {}, {}, {}
    for i, uid in enumerate(con["utterances"]):
        for rel in (f"rows/{i:03d}.json", f"rows/{i:03d}.npz", f"rows/{i:03d}_logits.npz"):
            if file_hash(ROOT / P2DIR_RUN / rel) != sealed["files"][rel]:
                raise ValueError(f"P2-DIR sealed file changed: {rel}")
        rows[uid] = json.loads((ROOT / P2DIR_RUN / f"rows/{i:03d}.json").read_text())
        with np.load(ROOT / P2DIR_RUN / f"rows/{i:03d}.npz") as z:
            vecs[uid] = {k: z[k] for k in z.files}
        with np.load(ROOT / P2DIR_RUN / f"rows/{i:03d}_logits.npz") as z:
            nones[uid] = {k: z[k] for k in z.files if k.endswith("_none")}
    return rows, vecs, nones


def run_s1(args) -> None:
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.lss.sites import num_forced_prefix_from
    from csasr.models.whisper import batch_model_inputs, load_whisper
    from csasr.utils.config import load_config

    out = ROOT / args.out
    m = verify_manifest(out, "s1")
    con = json.loads((ROOT / CONSTRUCTION).read_text())
    if con["construction_hash"] != m["construction_hash"]:
        raise ValueError("construction hash")
    p2d_rows, p2d_vecs, p2d_none = load_p2dir_rows(con)
    p2r_pop = json.loads((ROOT / "results/inference_cf/p2r/population.json").read_text())
    pidx = {u: i for i, u in enumerate(p2r_pop["utterances"])}
    p2r_rows = {u: json.loads((ROOT / P2R_RUN / f"rows/{pidx[u]:03d}.json").read_text()) for u in con["utterances"]}
    gates = join_gates(con, p2r_rows, p2d_rows)
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    bundle.model.requires_grad_(False)
    partition = tokenizer_partition(bundle.processor.tokenizer)
    if partition["hash"] != m["partition_hash"]:
        raise ValueError("partition hash")
    nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    if nfp != 4:
        raise ValueError("forced prefix")
    ctx = p2dir.Ctx(bundle, partition, nfp, 0.0)
    base_dir = ROOT / p2dir.BASELINE_RUN
    bidx = {r["utterance_id"]: i for i, r in enumerate(json.loads((base_dir / "panel.json").read_text())["rows"])}
    audio = {r["utterance_id"]: r for r in json.loads((ROOT / p2dir.AUDIO_PANEL).read_text())["rows"]}
    by_utt: dict = {}
    for p in con["positions"]:
        by_utt.setdefault(p["utterance_id"], []).append(int(p["t"]))
    torch.cuda.reset_peak_memory_stats()
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
               "gpu": torch.cuda.get_device_name(0), "start_unix": time.time(), "status": "running",
               "autograd_calls": 0, "readout_calls": 0, "passes": {}}
    atomic_json(out / "runtime.json", runtime)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    encs: dict = {}

    def encode(uid):
        inputs = batch_model_inputs(bundle, [audio[uid]["audio_path"]])
        with torch.inference_mode():
            return BaseModelOutput(last_hidden_state=bundle.model.model.encoder(
                input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state)

    failures = 0
    c2_norms = []
    t_a = time.time()
    for i, uid in enumerate(con["utterances"]):
        toks = json.loads((base_dir / "rows" / f"{bidx[uid]:03d}.json").read_text())["systems"][p2dir.BASELINE_SYSTEM]["tokens"]
        ts = sorted(by_utt[uid])
        try:
            with torch.inference_mode():
                recs, vecs, logits = s1_pass_a(ctx, encoded=encode(uid), tokens=toks, ts=ts,
                                               gates={t: gates[(uid, t)] for t in ts}, ref_vecs=p2d_vecs[uid],
                                               ref_none=p2d_none[uid])
            np.savez_compressed(out / "rows" / f"{i:03d}_a.npz", **vecs)
            np.savez_compressed(out / "rows" / f"{i:03d}_a_logits.npz", **logits)
            res = {"status": "ok", "positions": recs, "vectors_sha256": file_hash(out / "rows" / f"{i:03d}_a.npz"),
                   "logits_sha256": file_hash(out / "rows" / f"{i:03d}_a_logits.npz")}
            c2_norms += [recs[str(t)]["arms"]["C2"]["edit_norm"] for t in ts]
        except Exception as exc:
            import traceback
            failures += 1
            res = {"status": "failure", "reason": repr(exc), "traceback": traceback.format_exc()}
        res.update(identity=uid, manifest_hash=m["manifest_hash"])
        atomic_json(out / "rows" / f"{i:03d}_a.json", res)
        print(f"P2-SEL S1/A {i + 1}/{len(con['utterances'])} {uid} {res['status']}", flush=True)
    runtime["passes"]["A"] = time.time() - t_a
    if failures:
        runtime.update(status="failed", failures=failures, end_unix=time.time())
        atomic_json(out / "runtime.json", runtime)
        raise SystemExit("pass A failures")
    target = broad_target(c2_norms)
    q2 = float(sum(e * e for e in c2_norms))
    broad = {"Q2": q2, "N": N_ELIGIBLE, "e_broad": target, "c2_rows": len(c2_norms),
             "rows_a_sha256": {f"rows/{i:03d}_a.json": file_hash(out / "rows" / f"{i:03d}_a.json")
                               for i in range(len(con["utterances"]))}}
    atomic_json(out / "broad_target.json", broad)
    t_b = time.time()
    for i, uid in enumerate(con["utterances"]):
        toks = json.loads((base_dir / "rows" / f"{bidx[uid]:03d}.json").read_text())["systems"][p2dir.BASELINE_SYSTEM]["tokens"]
        ts = sorted(by_utt[uid])
        try:
            with torch.inference_mode():
                recs, vecs, logits = s1_pass_b(ctx, encoded=encode(uid), tokens=toks, ts=ts, target=target,
                                               ref_vecs=p2d_vecs[uid])
            np.savez_compressed(out / "rows" / f"{i:03d}_b.npz", **vecs)
            np.savez_compressed(out / "rows" / f"{i:03d}_b_logits.npz", **logits)
            res = {"status": "ok", "positions": recs, "vectors_sha256": file_hash(out / "rows" / f"{i:03d}_b.npz"),
                   "logits_sha256": file_hash(out / "rows" / f"{i:03d}_b_logits.npz")}
        except Exception as exc:
            import traceback
            failures += 1
            res = {"status": "failure", "reason": repr(exc), "traceback": traceback.format_exc()}
        res.update(identity=uid, manifest_hash=m["manifest_hash"])
        atomic_json(out / "rows" / f"{i:03d}_b.json", res)
        print(f"P2-SEL S1/B {i + 1}/{len(con['utterances'])} {uid} {res['status']}", flush=True)
    runtime["passes"]["B"] = time.time() - t_b
    assert_no_site_hooks(bundle)
    if any(p.grad is not None for p in bundle.model.parameters()):
        raise RuntimeError("parameter gradient present")
    runtime.update(end_unix=time.time(), status="completed" if failures == 0 else "failed", failures=failures,
                   peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_vram_reserved_bytes=torch.cuda.max_memory_reserved())
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)
    if failures:
        raise SystemExit(f"{failures} failures")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mf = sub.add_parser("manifest")
    mf.add_argument("--stage", required=True, choices=("s1",))
    mf.add_argument("--out", required=True)
    s = sub.add_parser("s1")
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    {"manifest": cmd_manifest, "s1": run_s1}[args.cmd](args)


if __name__ == "__main__":
    main()
