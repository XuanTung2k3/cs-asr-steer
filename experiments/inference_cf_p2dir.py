#!/usr/bin/env python
"""P2-DIR runner (GPU; separate inference_cf development stage). Frozen spec:
docs/inference_cf/P2_DIR_DIRECTION_IDENTIFICATION_SPEC.md.

Modes (each guarded by its own immutable manifest written by inference_cf_p2dir_prepare.py):

* ``extract``   -- unedited B (forced-ZH) and E (forced-EN) L16 DG-02 site states at the 180 frozen
  positions, teacher-forced on the matched-baseline B0M_L16 tokens. Reads ONLY the construction
  projection (ids, dialogue, t, stratum). No edit, no outcome.
* ``exp1``      -- direction-only causal screen. Per position, from the identical unedited B cache:
  D2 READOUT from the pre-step cache, then the clean B/E step, D0 OLD, D1 UNIQUE (sealed fold of
  the position's dialogue); all three directions are sealed (hashed, saved) before any pulse;
  then single pulses D0, D1, D2 (fixed order) at the exact realized edit norm e* with the P2-R
  solver/hook; cache cropped between arms and the clean step restored bitwise. Reads NO reference
  information: evaluator statistics are computed later from the saved full logits.
* ``exp1-eval`` -- evaluator-only Jacobian pass, refused unless the exp1 directions are sealed:
  gradients of the fixed-competitor reference margin (and log p(ref), log P_E, log P_M) at the
  same unedited states (P2-RJ zero probe). Never used as a direction.
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
import torch

from csasr.inference_cf.core import atomic_json, digest, file_hash
from csasr.inference_cf.core_p1 import processed_argmax
from csasr.inference_cf.directions import (DirectionContext, OldDirection, ReadoutDirection,
                                           UniqueDirection)
from csasr.inference_cf.readout import clone_scratch
from csasr.lss.sites import assert_no_site_hooks
import experiments.inference_cf_p2r as p2r

SCHEMA = "p2dir_direction_identification_v1"
CONFIG = "configs/inference_cf/p2_dir_direction_identification.json"
CB = [50258, 50260, 50360, 50364]
CE = [50258, 50259, 50360, 50364]
BASELINE_RUN = "results/inference_cf/p2_A_r1_L16"
BASELINE_SYSTEM = "B0M_L16"
AUDIO_PANEL = "results/inference_cf/p0_r2/inference_panel.json"
ARMS = ("D0", "D1", "D2")
LAYER = 16


# ---- bf16 lossless logit storage -------------------------------------------------------------

def pack_bf16(x: torch.Tensor) -> np.ndarray:
    """Float32 logits that came from a bf16 model output, stored losslessly as raw bf16 bits."""
    b = x.detach().cpu().to(torch.bfloat16)
    if not torch.equal(b.float(), x.detach().cpu().float()):
        raise ValueError("logits are not exactly bf16-representable")
    return b.view(torch.int16).numpy()


def unpack_bf16(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(a, dtype=np.int16).copy()).view(torch.bfloat16).float()


# ---- context ---------------------------------------------------------------------------------

class Ctx:
    def __init__(self, bundle, partition: dict, nfp: int, e_star: float, cB=CB, cE=CE):
        self.bundle, self.partition, self.nfp, self.e_star = bundle, partition, int(nfp), float(e_star)
        self.cB, self.cE = list(cB), list(cE)
        self.layer = LAYER
        gen = bundle.model.generation_config
        self.suppress, self.begin = list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
        self.eos = bundle.processor.tokenizer.eos_token_id


def _arm_v_fn(res):
    if res.direction is None:
        return lambda r: (None, res.reason)
    return lambda r: (res.direction, "ok")


# ---- extraction -------------------------------------------------------------------------------

def extract_utterance(ctx: Ctx, *, encoded, tokens: list[int], ts: list[int]) -> tuple[dict, dict]:
    """Unedited B/E states at the requested steps (no edit, no reference)."""
    B, E = p2r.DiagBranch(ctx.bundle, encoded, ctx.cB, "B"), p2r.DiagBranch(ctx.bundle, encoded, ctx.cE, "E")
    new, new_e = list(ctx.cB), list(ctx.cE)
    recs, states = {}, {}
    for t in range(max(ts) + 1):
        L = B.length
        lb, _, hb = B.step(new, capture_layer=ctx.layer, attention=True)
        _, _, he = E.step(new_e, capture_layer=ctx.layer, attention=False)
        if t in ts:
            expected = tokens[t] if t < len(tokens) else ctx.eos
            d0 = OldDirection()(DirectionContext(h_b=hb, h_e=he))
            recs[str(t)] = {"t": t, "query": L + len(new) - 1,
                            "argmax": processed_argmax(lb, t, ctx.suppress, ctx.begin), "expected_token": expected,
                            "pre_norm_audit_style": float(hb.reshape(1, 1, -1).norm(dim=-1)[0, 0]),
                            "d0": d0.record(),
                            "cos_d_state": None if d0.direction is None else
                            float(torch.dot(d0.direction.double(), hb.double()) / torch.linalg.vector_norm(hb.double()))}
            states[str(t)] = (hb.clone(), he.clone())
        if t < len(tokens):
            new, new_e = [tokens[t]], [tokens[t]]
    return recs, states


# ---- exp1 pulses --------------------------------------------------------------------------------

def exp1_utterance(ctx: Ctx, *, encoded, tokens: list[int], ts: list[int], d1: UniqueDirection,
                   extracted: dict) -> tuple[dict, dict, dict]:
    """Seal D0/D1/D2 at each frozen position, then run the three single pulses at e*."""
    B, E = p2r.DiagBranch(ctx.bundle, encoded, ctx.cB, "B"), p2r.DiagBranch(ctx.bundle, encoded, ctx.cE, "E")
    d0p = OldDirection()
    d2p = ReadoutDirection(ctx.bundle, layer=ctx.layer, suppress=ctx.suppress, begin=ctx.begin,
                           partition=ctx.partition)
    new, new_e = list(ctx.cB), list(ctx.cE)
    recs, vecs, logits = {}, {}, {}
    for t in range(max(ts) + 1):
        L = B.length
        r2 = None
        if t in ts:      # D2 from the PRE-step B cache, before the clean step is taken
            r2 = d2p(DirectionContext(b_cache_pre_step=B.cache, encoded=encoded, new_tokens=list(new),
                                      start=L, step=t))
        lb, _, hb = B.step(new, capture_layer=ctx.layer, attention=True)
        _, _, he = E.step(new_e, capture_layer=ctx.layer, attention=False)
        assert_no_site_hooks(ctx.bundle)
        if t in ts:
            query = L + len(new) - 1
            hb_x, he_x = extracted[str(t)]
            r0 = d0p(DirectionContext(h_b=hb, h_e=he))
            r1 = d1(DirectionContext())
            res = {"D0": r0, "D1": r1, "D2": r2}
            pre = f"t{t}_"
            # ---- seal: directions fixed (copied, hashed) before any pulse outcome exists ----
            sealed = {a: (None if r.direction is None else r.direction.detach().cpu().float().clone()) for a, r in res.items()}
            rec = {"t": t, "query": query,
                   "identity": {"extracted_hb_bitwise": bool(torch.equal(hb, hb_x)),
                                "extracted_he_bitwise": bool(torch.equal(he, he_x)),
                                "d2_logits_bitwise": bool(torch.equal(r2.extras["logits"], lb)),
                                "d2_site_bitwise": bool(torch.equal(r2.extras["site"], hb)),
                                "argmax": processed_argmax(lb, t, ctx.suppress, ctx.begin),
                                "pre_norm_audit_style": float(hb.reshape(1, 1, -1).norm(dim=-1)[0, 0])},
                   "directions": {a: r.record() for a, r in res.items()},
                   "d2_runtime_sec": r2.extras["runtime_sec"], "arms": {}}
            rec["identity"]["cos_d_state"] = None if r0.direction is None else float(
                torch.dot(r0.direction.double(), hb.double()) / torch.linalg.vector_norm(hb.double()))
            vecs[pre + "hb"], vecs[pre + "he"] = hb.numpy(), he.numpy()
            for a, d in sealed.items():
                if d is not None:
                    vecs[pre + a] = d.numpy()
            if r2.extras.get("gradient") is not None:
                vecs[pre + "g_readout"] = r2.extras["gradient"].numpy()
            logits[pre + "none"] = pack_bf16(lb)
            for a in ARMS:
                B.crop(L)
                info: dict = {}
                hook = p2r.pulse_hook(ctx.bundle, ctx.layer, query, _arm_v_fn(res[a]), ctx.e_star, ctx.nfp, info)
                t0 = time.perf_counter()
                la, _, h_cons = B.step(new, capture_layer=ctx.layer, attention=True, hook=hook)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                assert_no_site_hooks(ctx.bundle)
                audit = hook.records[-1].to_dict() if hook.records else None
                v = info.pop("v", None)
                arm = {"solver": info, "edit_norm": audit["edit_norm"] if audit and audit["steered"] else 0.0,
                       "pre_norm": audit["pre_norm"] if audit else None, "steered": bool(audit and audit["steered"]),
                       "argmax": processed_argmax(la, t, ctx.suppress, ctx.begin), "pulse_runtime_sec": dt}
                if arm["steered"]:
                    cons = h_cons.double() - hb.double()
                    arm["consumed_edit_norm"] = float(torch.linalg.vector_norm(cons))
                    arm["cos_edit_v"] = float(torch.dot(cons, v.cpu()) / (arm["consumed_edit_norm"] + 1e-30))
                    arm["relative_sq_energy"] = arm["edit_norm"] ** 2 / float(hb.double().norm() ** 2)
                    arm["solver_matches_hook"] = abs(info.get("edit_norm", -1) - arm["edit_norm"]) <= 1e-6 * max(1, arm["edit_norm"])
                    vecs[pre + "post_" + a] = h_cons.numpy()
                    vecs[pre + "solver_" + a] = v.cpu().double().numpy()
                rec["arms"][a] = arm
                logits[pre + a] = pack_bf16(la)
            B.crop(L)
            lb2, _, hb2 = B.step(new, capture_layer=ctx.layer, attention=True)
            rec["restore_bitwise"] = bool(torch.equal(lb2, lb) and torch.equal(hb2, hb))
            for a in ARMS:      # sealed directions untouched by the pulses
                cur = res[a].direction
                rec["directions"][a]["unchanged_after_pulses"] = (sealed[a] is None and cur is None) or bool(
                    cur is not None and sealed[a] is not None and torch.equal(cur.detach().cpu().float(), sealed[a]))
            recs[str(t)] = rec
        if t < len(tokens):
            new, new_e = [tokens[t]], [tokens[t]]
    return recs, vecs, logits


# ---- exp1 evaluator pass (post-seal) ------------------------------------------------------------

def eval_utterance(ctx: Ctx, *, encoded, tokens: list[int], jobs: dict, extracted: dict) -> tuple[dict, dict]:
    """Evaluator-only fixed-competitor margin Jacobian at the same unedited states (zero probe)."""
    import experiments.inference_cf_p2rj as rj
    B = p2r.DiagBranch(ctx.bundle, encoded, ctx.cB, "B")
    new = list(ctx.cB)
    recs, vecs = {}, {}
    for t in range(max(jobs) + 1):
        L = B.length
        g = None
        if t in jobs:
            p = jobs[t]
            query = L + len(new) - 1
            scratch, enc = clone_scratch(B.cache, encoded)
            dev = next(ctx.bundle.model.parameters()).device
            probe = rj.SiteGradientProbe(ctx.bundle, ctx.layer, query)
            with torch.inference_mode(False), torch.enable_grad():
                with probe:
                    out = ctx.bundle.model(encoder_outputs=enc, decoder_input_ids=torch.tensor([list(new)], device=dev),
                                           past_key_values=scratch, use_cache=True,
                                           cache_position=torch.arange(L, L + len(new), device=dev),
                                           output_attentions=False, return_dict=True)
                assert_no_site_hooks(ctx.bundle)
                z = out.logits[0, -1]
                vals = rj.evaluator_targets(z, t, ctx.suppress, ctx.begin, p["target_ids"], p["competitor"], ctx.partition)
                grads = {}
                for k in rj.TARGETS:
                    (gk,) = torch.autograd.grad(vals[k], probe.delta, retain_graph=True)
                    grads[k] = gk.detach().float().cpu()
            probe._assert_zero()
            g = {"values": {k: float(v.detach()) for k, v in vals.items()}, "grads": grads,
                 "logits": z.detach().float().cpu(), "site": probe.site.float().cpu()}
            del out, scratch, enc, probe
        lb, _, hb = B.step(new, capture_layer=ctx.layer, attention=True)
        if g is not None:
            hb_x, _ = extracted[str(t)]
            recs[str(t)] = {"t": t, "values": g["values"],
                            "logits_bitwise": bool(torch.equal(g["logits"], lb)),
                            "site_bitwise": bool(torch.equal(g["site"], hb)),
                            "extracted_hb_bitwise": bool(torch.equal(hb, hb_x))}
            for k, gk in g["grads"].items():
                vecs[f"t{t}_g_{k}"] = gk.numpy()
        if t < len(tokens):
            new = [tokens[t]]
    return recs, vecs


# ---- job entry --------------------------------------------------------------------------------

def verify_manifest(out: Path, stage: str) -> dict:
    m = json.loads((out / "manifest.json").read_text())
    if m.get("schema") != SCHEMA or m.get("stage") != stage or \
            digest({k: v for k, v in m.items() if k != "manifest_hash"}) != m["manifest_hash"]:
        raise ValueError(f"invalid P2-DIR {stage} manifest")
    for path, expected in m["sources"].items():
        if file_hash(ROOT / path) != expected:
            raise ValueError(f"source changed after freeze: {path}")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != m["git_commit"]:
        raise ValueError("Git commit changed after P2-DIR manifest freeze")
    return m


def load_extracted(ext_dir: Path, construction: dict) -> dict:
    with np.load(ext_dir / "states.npz") as z:
        HB, HE = z["H_B"], z["H_E"]
    out: dict = {}
    for i, p in enumerate(construction["positions"]):
        out.setdefault(p["utterance_id"], {})[str(p["t"])] = (torch.from_numpy(HB[i].copy()), torch.from_numpy(HE[i].copy()))
    return out


def assemble_states(out: Path, construction: dict, manifest_hash: str) -> None:
    """Ordered (construction order) float32 state matrices from the per-utterance extraction rows."""
    idx = {u: i for i, u in enumerate(construction["utterances"])}
    HB, HE = [], []
    cache: dict = {}
    for p in construction["positions"]:
        i = idx[p["utterance_id"]]
        if i not in cache:
            row = json.loads((out / "rows" / f"{i:03d}.json").read_text())
            if row["status"] != "ok" or row["identity"] != p["utterance_id"] or row["manifest_hash"] != manifest_hash:
                raise ValueError(f"extraction row {i} not ok")
            if file_hash(out / "rows" / f"{i:03d}.npz") != row["vectors_sha256"]:
                raise ValueError(f"extraction vectors {i} changed")
            with np.load(out / "rows" / f"{i:03d}.npz") as z:
                cache[i] = {k: z[k] for k in z.files}
        HB.append(cache[i][f"t{p['t']}_hb"])
        HE.append(cache[i][f"t{p['t']}_he"])
    np.savez(out / "states.npz", H_B=np.stack(HB).astype(np.float32), H_E=np.stack(HE).astype(np.float32))


def main() -> None:
    from transformers.modeling_outputs import BaseModelOutput
    from csasr.inference_cf.core import validated_row
    from csasr.inference_cf.core_r2 import tokenizer_partition
    from csasr.lss.sites import num_forced_prefix_from
    from csasr.models.whisper import batch_model_inputs, load_whisper
    from csasr.utils.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("extract", "exp1", "exp1-eval"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    stage = "exp1" if args.mode == "exp1-eval" else args.mode
    m = verify_manifest(out, stage)
    cfg = json.loads((ROOT / CONFIG).read_text())
    if digest(cfg) != m["config_hash"]:
        raise ValueError("config changed")
    construction = json.loads((ROOT / m["construction"]).read_text())
    if construction["construction_hash"] != m["construction_hash"]:
        raise ValueError("construction projection hash mismatch")
    bundle = load_whisper(load_config(ROOT / "configs/model/whisper_large_v3.yaml"))
    if bundle.device != "cuda" or bundle.dtype != torch.bfloat16:
        raise ValueError("requires CUDA bf16")
    partition = tokenizer_partition(bundle.processor.tokenizer)
    if partition["hash"] != m["partition_hash"]:
        raise ValueError("tokenizer partition mismatch")
    nfp = num_forced_prefix_from(bundle.processor, language="zh", task="transcribe")
    if nfp != 4:
        raise ValueError("forced prefix mismatch")
    gen = bundle.model.generation_config
    if digest({"suppress": list(gen.suppress_tokens or []), "begin": list(gen.begin_suppress_tokens or [])}) != m["suppression_hash"]:
        raise ValueError("suppression lists changed")
    ctx = Ctx(bundle, partition, nfp, cfg["energy"]["e_star"])
    base_dir = ROOT / BASELINE_RUN
    base_panel = json.loads((base_dir / "panel.json").read_text())["rows"]
    base_idx = {r["utterance_id"]: i for i, r in enumerate(base_panel)}
    audio = {r["utterance_id"]: r for r in json.loads((ROOT / AUDIO_PANEL).read_text())["rows"]}
    by_utt: dict = {}
    for p in construction["positions"]:
        by_utt.setdefault(p["utterance_id"], []).append(p)
    utts = construction["utterances"]
    extracted = None
    if args.mode in ("exp1", "exp1-eval"):
        ext_dir = ROOT / m["extraction_run"]
        if file_hash(ext_dir / "states.npz") != m["extraction_states_hash"]:
            raise ValueError("extraction states hash mismatch")
        extracted = load_extracted(ext_dir, construction)
    folds = None
    if args.mode == "exp1":
        folds = json.loads((ROOT / m["folds"]).read_text())
        if digest({k: v for k, v in folds.items() if k != "folds_hash"}) != folds["folds_hash"] or folds["folds_hash"] != m["folds_hash"]:
            raise ValueError("fold artifact hash mismatch")
    jobs_ref = None
    if args.mode == "exp1-eval":
        sealed = json.loads((out / "directions_sealed.json").read_text())
        if sealed.get("status") != "SEALED" or sealed["manifest_hash"] != m["manifest_hash"]:
            raise ValueError("exp1 directions are not sealed; evaluator pass refused")
        for rel, h in sealed["files"].items():
            if file_hash(out / rel) != h:
                raise ValueError(f"sealed file changed: {rel}")
        # evaluator references are loaded only now, after every direction artifact is sealed
        pos = json.loads((ROOT / m["positions"]).read_text())
        if pos["positions_hash"] != m["positions_hash"]:
            raise ValueError("positions hash mismatch")
        jobs_ref = {}
        for p in pos["positions"]:
            jobs_ref.setdefault(p["utterance_id"], {})[int(p["t"])] = {"target_ids": p["target_ids"], "competitor": p["competitor"]}
    torch.cuda.reset_peak_memory_stats()
    rt_name = {"extract": "runtime.json", "exp1": "runtime.json", "exp1-eval": "runtime_eval.json"}[args.mode]
    runtime = {"mode": args.mode, "job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(),
               "gpu": torch.cuda.get_device_name(0), "start_unix": time.time(), "status": "running",
               "autograd_calls": 0}
    atomic_json(out / rt_name, runtime)
    if args.mode == "exp1":
        bundle.model.requires_grad_(False)
    failures = 0
    (out / "rows").mkdir(parents=True, exist_ok=True)
    for i, uid in enumerate(utts):
        suffix = {"extract": "", "exp1": "", "exp1-eval": "_eval"}[args.mode]
        dest = out / "rows" / f"{i:03d}{suffix}.json"
        ts = sorted(int(p["t"]) for p in by_utt[uid])
        prev = validated_row(dest, uid, m["manifest_hash"])
        if prev is not None and prev.get("status") == "ok":
            continue
        t0 = time.time()
        try:
            bsys = json.loads((base_dir / "rows" / f"{base_idx[uid]:03d}.json").read_text())["systems"][BASELINE_SYSTEM]
            inputs = batch_model_inputs(bundle, [audio[uid]["audio_path"]])
            with torch.inference_mode():
                enc = BaseModelOutput(last_hidden_state=bundle.model.model.encoder(
                    input_features=inputs["input_features"], attention_mask=inputs["attention_mask"]).last_hidden_state)
                if args.mode == "extract":
                    recs, states = extract_utterance(ctx, encoded=enc, tokens=bsys["tokens"], ts=ts)
                    vecs = {}
                    for k, (a, b) in states.items():
                        vecs[f"t{k}_hb"], vecs[f"t{k}_he"] = a.numpy(), b.numpy()
                    np.savez(out / "rows" / f"{i:03d}.npz", **vecs)
                    res = {"status": "ok", "positions": recs, "vectors_sha256": file_hash(out / "rows" / f"{i:03d}.npz")}
                elif args.mode == "exp1":
                    dlg = by_utt[uid][0]["dialogue_id"]
                    fold = folds["folds"][dlg]
                    vec = None
                    if fold["status"] == "ok":
                        vec = np.load(ROOT / fold["vector_path"], allow_pickle=False)
                    d1 = UniqueDirection(vec, {**fold, "dialogue": dlg})
                    recs, vecs, logits = exp1_utterance(ctx, encoded=enc, tokens=bsys["tokens"], ts=ts, d1=d1,
                                                        extracted=extracted[uid])
                    np.savez_compressed(out / "rows" / f"{i:03d}.npz", **vecs)
                    np.savez_compressed(out / "rows" / f"{i:03d}_logits.npz", **logits)
                    res = {"status": "ok", "positions": recs,
                           "vectors_sha256": file_hash(out / "rows" / f"{i:03d}.npz"),
                           "logits_sha256": file_hash(out / "rows" / f"{i:03d}_logits.npz")}
                    runtime["autograd_calls"] += sum(r["directions"]["D2"]["counters"].get("autograd_calls", 0) for r in recs.values())
            if args.mode == "exp1-eval":
                recs, vecs = eval_utterance(ctx, encoded=enc, tokens=bsys["tokens"], jobs=jobs_ref[uid],
                                            extracted=extracted[uid])
                runtime["evaluator_autograd_calls"] = runtime.get("evaluator_autograd_calls", 0) + 4 * len(recs)
                np.savez_compressed(out / "rows" / f"{i:03d}_eval.npz", **vecs)
                res = {"status": "ok", "positions": recs, "vectors_sha256": file_hash(out / "rows" / f"{i:03d}_eval.npz")}
        except Exception as exc:
            import traceback
            failures += 1
            res = {"status": "failure", "reason": repr(exc), "traceback": traceback.format_exc()}
        res.update(identity=uid, manifest_hash=m["manifest_hash"], elapsed_sec=time.time() - t0)
        atomic_json(dest, res)
        print(f"P2-DIR {args.mode} {i + 1}/{len(utts)} {uid} {res['status']} {res['elapsed_sec']:.1f}s", flush=True)
    assert_no_site_hooks(bundle)
    if args.mode == "extract" and failures == 0:
        assemble_states(out, construction, m["manifest_hash"])
    if args.mode == "exp1" and failures == 0:
        files = {}
        for i in range(len(utts)):
            for suf in (".json", ".npz", "_logits.npz"):
                rel = f"rows/{i:03d}{suf}"
                files[rel] = file_hash(out / rel)
        atomic_json(out / "directions_sealed.json", {"status": "SEALED", "manifest_hash": m["manifest_hash"],
                                                     "files": files, "sealed_unix": time.time()})
    runtime.update(end_unix=time.time(), status="completed" if failures == 0 else "failed", failures=failures,
                   peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_vram_reserved_bytes=torch.cuda.max_memory_reserved())
    runtime["elapsed_sec"] = runtime["end_unix"] - runtime["start_unix"]
    atomic_json(out / rt_name, runtime)
    if failures:
        raise SystemExit(f"{failures} utterance failures")


if __name__ == "__main__":
    main()
