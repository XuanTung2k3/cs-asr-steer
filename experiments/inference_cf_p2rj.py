#!/usr/bin/env python
"""P2-RJ Jacobian runner (GPU; evaluator-side diagnostic namespace).

At the exact frozen P2-R D1 states (L16 DG-02 site of the unedited forced-ZH B branch, teacher
forced on the matched-baseline tokens) this computes the gradient of the evaluator-only reference
margin with respect to the site state, its tangent projection, and its alignment with the
realized P2-R arm edits (+d, -d, frozen random) -- geometry only.

EVALUATOR-ONLY. The gradient is never applied as a steering direction: every model forward in
this module is either unhooked or carries a probe whose added value is exactly zero (asserted at
forward time). No deployable module is modified or given reference information.
"""
from __future__ import annotations

import argparse
import copy
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
from csasr.inference_cf.core_p1 import direction, processed_argmax
from csasr.lss.sites import _first, _rebuild, assert_dropout_disabled, assert_no_site_hooks
from csasr.models.hooks import apply_steering
import experiments.inference_cf_cached as cached
from experiments.inference_cf_p2r import DiagBranch, random_direction, scaled_direction, solve_scale

SCHEMA = "p2rj_jacobian_diagnosis_v1"
ARMS = ("plus_d", "minus_d", "random")
TARGETS = ("margin", "logp_ref", "log_PE", "log_PM")
PASSES = ("bf16", "fp32")


# ---- evaluator-only gradient probe ---------------------------------------------------------

class SiteGradientProbe:
    """Adds a ZERO-valued, grad-requiring probe to u_source at one query of the DG-02 site
    (post-cross-attention, pre-FFN), so autograd can read d(target)/d(r). The forward is value
    identical to the unedited step. The probe never carries a nonzero value and exposes no way to
    supply a direction; it is not a steering hook."""

    def __init__(self, bundle, layer: int, query: int):
        self.bundle, self.layer, self.query = bundle, int(layer), int(query)
        dim = bundle.decoder_layer(self.layer).encoder_attn.out_proj.out_features
        dev = next(bundle.model.parameters()).device
        self.delta = torch.zeros(dim, dtype=torch.float32, device=dev, requires_grad=True)
        self.site: torch.Tensor | None = None
        self.calls = 0
        self._q = None
        self._cache_position = None
        self._handles: list = []

    def _assert_zero(self) -> None:
        if int(torch.count_nonzero(self.delta.detach())) != 0:
            raise RuntimeError("SiteGradientProbe delta is nonzero: the gradient probe must never edit")

    def _layer_pre_hook(self, _mod, args, kwargs):
        self._cache_position = kwargs.get("cache_position", None)
        return None

    def _pre_hook(self, _mod, args):
        self._q = args[0]
        return None

    def _hook(self, _mod, _inp, output):
        self._assert_zero()
        u = _first(output)
        T = u.shape[1]
        cp = self._cache_position
        if cp is None or int(cp.reshape(-1)[-1]) != self.query:
            raise RuntimeError("probe query does not match the step's last cache position")
        self.site = (self._q + u)[0, T - 1].detach().clone()
        self.calls += 1
        mask = torch.zeros((1, T, 1), device=u.device, dtype=u.dtype)
        mask[0, T - 1, 0] = 1
        return _rebuild(output, u + mask * self.delta.to(u.dtype).view(1, 1, -1))

    def __enter__(self):
        self._assert_zero()
        assert_dropout_disabled(self.bundle, [self.layer])
        layer = self.bundle.decoder_layer(self.layer)
        self._handles.append(layer.register_forward_pre_hook(self._layer_pre_hook, with_kwargs=True))
        self._handles.append(layer.encoder_attn_layer_norm.register_forward_pre_hook(self._pre_hook))
        self._handles.append(layer.encoder_attn.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._q = self._cache_position = None
        return False


# ---- differentiable evaluator targets ------------------------------------------------------

def processed_logits(z: torch.Tensor, step: int, suppress, begin) -> torch.Tensor:
    """Differentiable version of the generate() suppression (same rule as cached._processed)."""
    mask = torch.zeros_like(z, dtype=torch.bool)
    if suppress:
        mask[list(suppress)] = True
    if step == 0 and begin:
        mask[list(begin)] = True
    return z.masked_fill(mask, -float("inf"))


def evaluator_targets(z: torch.Tensor, step: int, suppress, begin, target_ids: list[int], competitor: int,
                      partition: dict) -> dict[str, torch.Tensor]:
    """m = logsumexp z(Y_ref) - z(c*) (normalizer-free), log p(ref), log P_E, log P_M (processed)."""
    zp = processed_logits(z.float(), step, suppress, begin)
    ref = torch.logsumexp(zp[target_ids], 0)
    logz = torch.logsumexp(zp, 0)
    return {"margin": ref - zp[competitor], "logp_ref": ref - logz,
            "log_PE": torch.logsumexp(zp[partition["embedded_ids"]], 0) - logz,
            "log_PM": torch.logsumexp(zp[partition["matrix_ids"]], 0) - logz}


def grad_step(ctx, br: DiagBranch, new: list[int], t: int, target_ids, competitor) -> dict:
    """One gradient-enabled forward of step t on a deep copy of ``br``'s cache (the frozen branch
    is not advanced), probe at the query; returns values, gradients, site, logits."""
    b = ctx.bundle
    start = br.length
    query = start + len(new) - 1
    ids = torch.tensor([list(new)], device=b.device, dtype=torch.long)
    positions = torch.arange(start, start + len(new), device=b.device)
    cache = copy.deepcopy(br.cache)
    probe = SiteGradientProbe(b, ctx.layer, query)
    with torch.enable_grad():
        with probe:
            out = b.model(encoder_outputs=br.encoded, decoder_input_ids=ids, past_key_values=cache,
                          use_cache=True, cache_position=positions, output_attentions=True, return_dict=True)
        assert_no_site_hooks(b)
        z = out.logits[0, -1]
        vals = evaluator_targets(z, t, ctx.suppress, ctx.begin, target_ids, competitor, ctx.partition)
        grads = {}
        for k in TARGETS:
            (g,) = torch.autograd.grad(vals[k], probe.delta, retain_graph=True)
            grads[k] = g.detach().double().cpu()
    probe._assert_zero()
    if probe.calls != 1:
        raise RuntimeError(f"probe fired {probe.calls} times")
    del out, cache
    return {"values": {k: float(v.detach()) for k, v in vals.items()}, "grads": grads,
            "site": probe.site, "logits": z.detach().float().cpu(), "query": query}


# ---- geometry (state-vector arithmetic only; no model forward) -----------------------------

def tangent(g: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    rhat = r / torch.linalg.vector_norm(r)
    return g - rhat * torch.dot(rhat, g)


def hook_edit(site_vec: torch.Tensor, v: torch.Tensor, s: float) -> torch.Tensor:
    """Realized P2-R hook edit (steered - site) in the site dtype, returned as float64 CPU."""
    x = site_vec.reshape(1, 1, -1)
    vv = v.detach().cpu().double()
    vv = vv / torch.linalg.vector_norm(vv)
    dirs = scaled_direction(vv, s, x).reshape(1, 1, -1)
    gain = torch.ones((1, 1), device=x.device, dtype=x.dtype)
    st = apply_steering(x, dirs, 1.0, 1.0, gain, True)
    return (st.double() - x.double()).reshape(-1).cpu()


def chord_edit(r: torch.Tensor, v: torch.Tensor, e: float) -> tuple[torch.Tensor | None, str]:
    """Analytic float64 NormPreserve edit of exact chord length e along unit v (fp32 pass)."""
    r = r.double(); v = v.double() / torch.linalg.vector_norm(v.double())
    rn = float(torch.linalg.vector_norm(r))
    phi = math.acos(max(-1.0, min(1.0, float(torch.dot(r, v)) / rn)))
    if e >= 2 * rn:
        return None, "energy_unreachable"
    theta = 2 * math.asin(e / (2 * rn))
    if phi <= theta + 1e-9:
        return None, "energy_unreachable"
    s = rn * math.sin(theta) / math.sin(phi - theta)
    til = r + s * v
    return til * (rn / float(torch.linalg.vector_norm(til))) - r, "ok"


def position_stats(r: torch.Tensor, grads: dict, arm_dirs: dict, arm_edits: dict, e_star: float) -> dict:
    """All per-position P2-RJ quantities from float64 vectors (shared by runner and analysis)."""
    r = r.double()
    out = {"r_norm": float(torch.linalg.vector_norm(r)), "targets": {}, "arms": {}}
    rhat = r / torch.linalg.vector_norm(r)
    tans = {k: tangent(g.double(), r) for k, g in grads.items()}
    for k, g in grads.items():
        g = g.double()
        out["targets"][k] = {"g_norm": float(torch.linalg.vector_norm(g)),
                             "g_tan_norm": float(torch.linalg.vector_norm(tans[k])),
                             "g_rad": float(torch.dot(rhat, g))}
    gt = tans["margin"]
    gtn = float(torch.linalg.vector_norm(gt))
    A = gtn * e_star
    out["A"] = A
    for a, v in arm_dirs.items():
        rec = {}
        if v is not None:
            v = v.double() / torch.linalg.vector_norm(v.double())
            for k in TARGETS:
                tn = float(torch.linalg.vector_norm(tans[k]))
                rec[f"cos_gtan_v_{k}"] = float(torch.dot(tans[k], v)) / tn if tn > 0 else None
        dlt = arm_edits.get(a)
        if dlt is not None:
            dlt = dlt.double()
            rec["edit_norm"] = float(torch.linalg.vector_norm(dlt))
            for k in TARGETS:
                rec[f"pred_{k}"] = float(torch.dot(grads[k].double(), dlt))
            rec["kappa"] = float(torch.dot(gt, dlt)) / A if A > 0 else None
        out["arms"][a] = rec
    for k in ("log_PE", "log_PM", "logp_ref"):
        tn = float(torch.linalg.vector_norm(tans[k]))
        out["targets"][k]["cos_with_margin_tan"] = float(torch.dot(tans[k], gt)) / (tn * gtn) if tn > 0 and gtn > 0 else None
    return out


# ---- per-utterance pass ----------------------------------------------------------------------

class Ctx:
    def __init__(self, bundle, cfg, partition, nfp):
        self.bundle, self.cfg, self.partition, self.nfp = bundle, cfg, partition, nfp
        self.layer = int(cfg["site"]["layer"])
        self.cond = cfg["conditions"]
        self.e_star = float(cfg["energy"]["e_star"])
        gen = bundle.model.generation_config
        self.suppress, self.begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
        self.eos = bundle.processor.tokenizer.eos_token_id


def nograd_step(br: DiagBranch, new, **kw):
    """P2-R Branch.step (identical code) under no_grad instead of inference_mode, so that the
    cache can feed a later gradient-enabled forward. Numerically identical."""
    with torch.no_grad():
        return cached.Branch.step.__wrapped__(br, new, **kw)


def run_utterance(ctx: Ctx, *, encoded, uid: str, tokens: list[int], positions: list[dict], mode: str) -> tuple[dict, dict]:
    cB, cE = ctx.cond["cB"], ctx.cond["cE"]
    B, E = DiagBranch(ctx.bundle, encoded, cB, "B"), DiagBranch(ctx.bundle, encoded, cE, "E")
    new, new_e = list(cB), list(cE)
    jobs = {p["t"]: p for p in positions}
    last = max(jobs)
    recs, vecs = {}, {}
    for t in range(last + 1):
        gres = None
        if t in jobs:
            p = jobs[t]
            gres = grad_step(ctx, B, new, t, p["target_ids"], p["competitor"])
        lb, _, hb = nograd_step(B, new, capture_layer=ctx.layer, attention=True)
        _, _, he = nograd_step(E, new_e, capture_layer=ctx.layer, attention=False)
        assert_no_site_hooks(ctx.bundle)
        if gres is not None:
            p = jobs[t]
            site_dev = gres["site"]                                   # r in the site dtype, on device
            r = site_dev.double().cpu()
            dirn = direction(he, hb)
            d = dirn["d"]
            rv, rstat = random_direction(uid, t, site_dev)
            arm_dirs = {"plus_d": None if d is None else d.double(), "minus_d": None if d is None else -d.double(),
                        "random": rv}
            arm_edits, solver = {}, {}
            for a, v in arm_dirs.items():
                if v is None:
                    solver[a] = {"status": dirn["status"]}
                    continue
                if mode == "bf16":
                    vin = (d if a == "plus_d" else -d) if a != "random" else v
                    sol = solve_scale(site_dev, vin, ctx.e_star)
                    solver[a] = sol
                    if sol["status"] == "ok":
                        arm_edits[a] = hook_edit(site_dev, vin, sol["s"])
                else:
                    dlt, st = chord_edit(r, v, ctx.e_star)
                    solver[a] = {"status": st}
                    if dlt is not None:
                        arm_edits[a] = dlt
            stats = position_stats(r, gres["grads"], arm_dirs, arm_edits, ctx.e_star)
            summ = cached._processed(lb, t, ctx.suppress, ctx.begin)
            lp = torch.log_softmax(summ, -1)
            ref = torch.logsumexp(lp[p["target_ids"]], 0)
            others = lp.clone(); others[p["target_ids"]] = -float("inf")
            rec = {"t": t, "query": gres["query"], "stratum": p["stratum"], "values": gres["values"],
                   "identity": {"argmax": processed_argmax(lb, t, ctx.suppress, ctx.begin),
                                "logp_ref": float(ref), "margin_best_other": float(ref - others.max()),
                                "competitor_recomputed": int(torch.argmax(others)),
                                # tie-aware identity (v1.1): bf16 logits can tie; argmax/topk order differ
                                "logp_competitor": float(lp[p["competitor"]]), "max_other_logp": float(others.max()),
                                "logp_baseline_token": None if p.get("baseline_token") is None else float(lp[p["baseline_token"]]),
                                "logp_p2r_argmax": None if p.get("p2r") is None else float(lp[p["p2r"]["argmax"]]),
                                "r_norm_site": float(torch.linalg.vector_norm(r)),
                                # the P2-R hook-audit norm, computed exactly as _emit_records does
                                "pre_norm_audit_style": float(site_dev.reshape(1, 1, -1).detach().float().norm(dim=-1)[0, 0]),
                                "cos_d_state": None if d is None else
                                float(torch.dot(d.double(), hb.double()) / torch.linalg.vector_norm(hb.double())),
                                "dir": dirn["status"], "dir_norm": dirn["norm"],
                                "logits_bitwise_equal": bool(torch.equal(gres["logits"], lb)),
                                "site_equals_recorder": bool(torch.equal(site_dev.float().cpu(), hb)),
                                "random_status": rstat},
                   "solver": {a: {k: v for k, v in s.items()} for a, s in solver.items()},
                   "stats": stats}
            recs[str(t)] = rec
            pre = f"t{t}_"
            vecs[pre + "r"] = r.float().numpy()                   # exact: bf16/fp32 site
            for k, g in gres["grads"].items():
                vecs[pre + "g_" + k] = g.float().numpy()          # exact: read from a float32 probe
            if d is not None:
                vecs[pre + "d"] = d.float().numpy()
            vecs[pre + "v_random"] = rv.double().numpy()
            for a, dl in arm_edits.items():
                vecs[pre + "delta_" + a] = dl.double().numpy()
        if t < len(tokens):
            new, new_e = [tokens[t]], [tokens[t]]
    return recs, vecs


# ---- job entry -------------------------------------------------------------------------------

def main() -> None:
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.core import validated_row
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.lss.sites import num_forced_prefix_from
    from csasr.models.whisper import batch_model_inputs, load_whisper
    from csasr.utils.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="technical smoke only (never under results/)")
    args = ap.parse_args()
    out = ROOT / args.out
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA or digest({k: v for k, v in manifest.items() if k != "manifest_hash"}) != manifest["manifest_hash"]:
        raise ValueError("invalid P2-RJ manifest")
    for path, expected in manifest["sources"].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"source changed after freeze: {path}")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != manifest["git_commit"]:
        raise ValueError("Git commit changed after P2-RJ manifest freeze")
    cfg = json.loads((ROOT / manifest["config"]).read_text())
    pos = json.loads((ROOT / manifest["positions"]).read_text())
    if pos["positions_hash"] != manifest["positions_hash"]:
        raise ValueError("positions hash mismatch")
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    partition = tokenizer_partition(bundle.processor.tokenizer)
    if partition["hash"] != manifest["partition_hash"]:
        raise ValueError("tokenizer partition mismatch")
    nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    if nfp != int(cfg["site"]["num_forced_prefix"]):
        raise ValueError("forced prefix mismatch")
    bundle.model.requires_grad_(False)       # only the probe delta requires grad
    ctx = Ctx(bundle, cfg, partition, nfp)
    base_dir = ROOT / cfg["baseline_run"]
    base_panel = json.loads((base_dir / "panel.json").read_text())["rows"]
    audio = {r["utterance_id"]: r for r in json.loads((ROOT / "results/inference_cf/p0_r2/inference_panel.json").read_text())["rows"]}
    base_idx = {r["utterance_id"]: i for i, r in enumerate(base_panel)}
    torch.cuda.reset_peak_memory_stats()
    runtime = {"job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
               "gpu": torch.cuda.get_device_name(0), "start_unix": time.time(), "status": "running", "passes": {}}
    atomic_json(out / "runtime.json", runtime)
    utts = pos["utterances"][: args.limit] if args.limit else pos["utterances"]
    for mode in PASSES:
        if mode == "fp32":
            bundle.model.float()
            if next(bundle.model.parameters()).dtype != torch.float32:
                raise RuntimeError("fp32 cast failed")
        t_pass = time.time()
        for i, uid in enumerate(utts):
            dest = out / "rows" / f"{i:03d}_{mode}.json"
            if validated_row(dest, uid, manifest["manifest_hash"]):
                continue
            plist = [p for p in pos["positions"] if p["utterance_id"] == uid]
            if not plist:
                continue
            t0 = time.time()
            try:
                bsys = json.loads((base_dir / "rows" / f"{base_idx[uid]:03d}.json").read_text())["systems"][cfg["baseline_system"]]
                inputs = batch_model_inputs(bundle, [audio[uid]["audio_path"]])
                with torch.no_grad():
                    feats = inputs["input_features"].to(next(bundle.model.parameters()).dtype)
                    enc = BaseModelOutput(last_hidden_state=bundle.model.model.encoder(
                        input_features=feats, attention_mask=inputs["attention_mask"]).last_hidden_state)
                recs, vecs = run_utterance(ctx, encoded=enc, uid=uid, tokens=bsys["tokens"], positions=plist, mode=mode)
                np.savez_compressed(out / "rows" / f"{i:03d}_{mode}.npz", **vecs)
                res = {"status": "ok", "positions": recs,
                       "vectors_sha256": file_hash(out / "rows" / f"{i:03d}_{mode}.npz")}
            except Exception as exc:
                import traceback
                res = {"status": "failure", "reason": repr(exc), "traceback": traceback.format_exc()}
            res.update(identity=uid, manifest_hash=manifest["manifest_hash"], mode=mode, elapsed_sec=time.time() - t0)
            atomic_json(dest, res)
            print(f"P2-RJ {mode} {i + 1}/{len(utts)} {uid} {res['status']} {res['elapsed_sec']:.1f}s", flush=True)
        runtime["passes"][mode] = {"elapsed_sec": time.time() - t_pass}
        atomic_json(out / "runtime.json", runtime)
    assert_no_site_hooks(bundle)
    runtime.update(end_unix=time.time(), status="completed", peak_vram_bytes=torch.cuda.max_memory_allocated())
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / "runtime.json", runtime)


if __name__ == "__main__":
    main()
