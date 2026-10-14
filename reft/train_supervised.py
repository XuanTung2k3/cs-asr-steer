#!/usr/bin/env python
"""P2 supervised representation adaptation under the frozen Auto-LID contract (docs/reft/P2_0_PROTOCOL.md).

  audit           -- CPU: G1 data/exposure integrity, frozen training manifest (500) and dev manifest (150)
  smoke           -- GPU: P2-0 engineering acceptance (tests A-F): model/site identity, native-vs-two-stage Auto parity on
                     dev150, zero-update parity, TF alignment/causality, beam/KV-cache consistency, one optimizer step,
                     save/load, runtime/VRAM. At most one optimizer step per method; no training.
  train --method  -- GPU: P2-1 training of R2 (LoReFT) or L1 (LoRA) with configs/reft/p2_1_frozen.yaml.
                     NOT executed in P2-0; requires project-owner approval.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import art, cfg, firewall, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from reft.provenance import cublas_consistent, git_state, loaded_cuda_libs, now, sha, slurm, weights_hash  # noqa: E402

P20 = ROOT / "configs/reft/p2_0.yaml"
P21 = ROOT / "configs/reft/p2_1_frozen.yaml"
FREEZE_SOURCES = ["docs/reft/P2_0_PROTOCOL.md", "configs/reft/p2_0.yaml", "configs/reft/p2_1_frozen.yaml",
                  "reft/adapters.py", "reft/auto_policy.py", "reft/train_supervised.py", "reft/evaluate_supervised.py",
                  "tests/reft/test_p2_0.py", "slurm/reft_p2.sbatch", "slurm/reft_p2_cpu.sbatch"]
GPU_FORBIDDEN = ("role_", "p0/evaluation", "error_inventory", "p2_1/evaluation")


def c20() -> dict:
    return yaml.safe_load(P20.read_text())


def c21() -> dict:
    return yaml.safe_load(P21.read_text())


def jhash(obj) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# ====================================================================== freeze

def run_freeze():
    rec = {"schema": "reft_p2_0_freeze_v1", "utc": now(), "git": git_state(),
           "sources": {p: sha(ROOT / p) for p in FREEZE_SOURCES}}
    print(write_json(art("freeze/p2_0.json"), rec))


def check_freeze() -> dict:
    p = art("freeze/p2_0.json")
    rec = read_json(p)
    bad = [s for s, h in rec["sources"].items() if sha(ROOT / s) != h]
    if bad:
        raise SystemExit(f"P2 sources changed since freeze: {bad}")
    return {"freeze_sha256": sha(p), "freeze_git_head": rec["git"]["head"]}


# ====================================================================== audit (CPU, G1)

def run_audit():
    import pyarrow.parquet as pq
    from reft.data_audit import ID_COLS, KEYS, assignment, role_rows
    C, D = c20(), cfg()["data"]
    p1panel = read_json(art("p1/panel.json"))["rows"]
    tg = read_json(art("p1/targets.json"))["targets"]
    sd = cfg()["p1"]["shared_decode"]
    dec_p = art(sd["path"])
    checks = {"p1_decode_sha": sha(dec_p) == sd["sha256"]}
    dec = {d["id"]: d for d in read_jsonl(dec_p)}
    seal = read_json(art("p0/output_seal.json"))
    checks["p0_seal_ok"] = all(sha(art(f)) == h for f, h in seal["files"].items())
    p0out = {o["id"]: o for o in read_jsonl(art("p0/auto_outputs.jsonl"))}
    devp = read_json(art("p0/panel.json"))["rows"]
    # full identity columns from the role manifests (ID columns only; no transcript of dev is read)
    train_role = {}
    for r in D["train_roles"]:
        for x in role_rows(r):
            train_role[x["utterance_id"]] = x
    dev_role = {x["utterance_id"]: x for x in role_rows(D["dev_role"])}
    tr = [train_role[r["utterance_id"]] for r in p1panel]
    dv = [dev_role[r["utterance_id"]] for r in devp]
    shared = {k: len({x[k] for x in tr} & {x[k] for x in dv}) for k in KEYS}
    checks["train_dev_disjoint_all_keys"] = all(v == 0 for v in shared.values())
    asg = assignment()
    locked = {d for d, ro in asg.items() if ro in D["locked_roles_never_opened"]}
    checks["no_locked_dialogues"] = not ({x["dialogue_id"] for x in tr + dv} & locked)
    checks["train_roles_only"] = {asg[x["dialogue_id"]] for x in tr} <= set(D["train_roles"])
    checks["dev_role_only"] = {asg[x["dialogue_id"]] for x in dv} == {D["dev_role"]}
    # targets and frozen LID
    gen = json.loads((Path(cfg()["model"]["path"]) / "generation_config.json").read_text())
    suppress = set(gen["suppress_tokens"])
    eos = int(gen["eos_token_id"])
    lang_tok = set(gen["lang_to_id"].values())
    rows, lens, sup_hits, bad = [], [], 0, 0
    for r in p1panel:
        i = r["utterance_id"]
        t = tg[i]["target_ids"]
        ok = t[-1] == eos and t.count(eos) == 1 and all(0 <= v < eos for v in t[:-1])
        bad += not ok
        sup_hits += sum(v in suppress for v in t[:-1])
        lens.append(len(t))
        p = dec[i]["prompt"]
        assert p[1] in lang_tok
        rows.append({"utterance_id": i, "dialogue_id": r["dialogue_id"], "audio_path": r["audio_path"],
                     "audio_sha256": r["audio_sha256"], "p1_partition": r["partition"], "frozen_lid_token": p[1],
                     "prompt": p, "target_ids": t})
    checks["targets_valid_eos_once"] = bad == 0
    checks["targets_max_len_ok"] = max(lens) <= C["training"]["max_target_tokens_incl_eos"]
    dev_rows = []
    for r in devp:
        o = p0out[r["utterance_id"]]
        dev_rows.append({k: r[k] for k in ("utterance_id", "dialogue_id", "audio_path", "audio_sha256")} |
                        {"frozen_lid_token": o["prompt"][1], "prompt": o["prompt"], "p0_tokens_sha": jhash(o["tokens"])})
    train_m = {"schema": "reft_p2_train_manifest_v1", "rows": rows}
    dev_m = {"schema": "reft_p2_dev_manifest_v1", "rows": dev_rows}
    tp, dp = art("p2_0/train_manifest.json"), art("p2_0/dev_manifest.json")
    write_json(tp, train_m)
    write_json(dp, dev_m)
    rec = {"schema": "reft_p2_0_data_audit_v1", "utc": now(), "git": git_state(), "checks": checks, "valid": all(checks.values()),
           "train": {"n": len(rows), "dialogues": len({r["dialogue_id"] for r in rows}),
                     "former_p1_audit_rows_now_training": sum(r["p1_partition"] == "audit" for r in rows),
                     "frozen_lid": dict(collections.Counter(r["frozen_lid_token"] for r in rows)),
                     "target_tokens_incl_eos": {"min": min(lens), "max": max(lens), "total": sum(lens)},
                     "content_tokens_in_generation_suppress_list": sup_hits},
           "dev": {"n": len(dev_rows), "dialogues": len({r["dialogue_id"] for r in dev_rows}),
                   "frozen_lid": dict(collections.Counter(r["frozen_lid_token"] for r in dev_rows))},
           "shared_identity_counts": shared,
           "identity_fields_checked": KEYS, "identity_fields_not_checkable": ["segment time overlap (no timestamps in manifest)"],
           "manifest_sha256": {"train": sha(tp), "dev": sha(dp)},
           "exposure": "dev150 = A3-ACL0-DEV150 (exposed; P0); 109 former P1 audit utterances now enter P2 training and can no longer serve as P2 evaluation data"}
    write_json(art("p2_0/data_audit.json"), rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("git",)}, indent=1, default=str))


# ====================================================================== GPU helpers

def load_backbone():
    from reft.baseline_auto import load_model
    b, env = load_model()
    return b, env


def encode(b, paths):
    import torch
    from csasr.models.whisper import batch_model_inputs
    inp = batch_model_inputs(b, paths)
    with torch.no_grad():
        enc = b.model.model.encoder(inp["input_features"]).last_hidden_state
    return inp, enc


def tf_forward(b, adapter, enc, X, grad=False):
    import contextlib
    import torch
    ctx = adapter.active(b.model) if adapter is not None else contextlib.nullcontext()
    with torch.set_grad_enabled(grad), ctx:
        return b.model(encoder_outputs=(enc,), decoder_input_ids=X, use_cache=False).logits


def batch_loss(b, adapter, rows, grad):
    from reft.auto_policy import build_tf_batch, tf_loss
    eos = b.processor.tokenizer.eos_token_id
    X, Y = build_tf_batch([r["prompt"] for r in rows], [r["target_ids"] for r in rows], eos, eos)
    _, enc = encode(b, [r["audio_path"] for r in rows])
    logits = tf_forward(b, adapter, enc, X.to(b.device), grad=grad)
    loss, n = tf_loss(logits, Y.to(b.device))
    return loss, n, logits, X, Y


def suppress_masks(b):
    import torch
    gen = b.model.generation_config
    V = b.model.config.vocab_size
    m = torch.zeros(V, dtype=torch.bool, device=b.device)
    m[list(gen.suppress_tokens or [])] = True
    m0 = m.clone()
    m0[list(gen.begin_suppress_tokens or [])] = True
    return m, m0


def gen_with_scores(b, path, code, beams, adapter):
    import contextlib
    import torch
    from csasr.models.whisper import batch_model_inputs
    D = cfg()["decode"]
    inp = batch_model_inputs(b, [path])
    ctx = adapter.active(b.model) if adapter is not None else contextlib.nullcontext()
    with torch.inference_mode(), ctx:
        out = b.model.generate(**inp, task="transcribe", language=code, do_sample=False, temperature=0.0,
                               max_new_tokens=D["max_new_tokens"], condition_on_prev_tokens=False, num_beams=beams,
                               return_dict_in_generate=True, output_scores=True)
    return out


def randomize(adapter, scale_w, scale_b, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if adapter.method == "loreft":
            m = adapter.module
            m.learned.weight.add_((torch.randn(m.learned.weight.shape, generator=g) * scale_w).to(m.learned.weight.device))
            m.learned.bias.add_((torch.randn(m.learned.bias.shape, generator=g) * scale_b).to(m.learned.bias.device))
        else:
            for v in adapter.wrapped.values():
                v.B.add_((torch.randn(v.B.shape, generator=g) * scale_w).to(v.B.device))


# ====================================================================== smoke (GPU)

def run_smoke():
    import torch
    import torch.nn.functional as F
    from reft.adapters import make_adapter, site_hook_count
    from reft.auto_policy import build_tf_batch, decode_frozen_lid, frozen_lid, prompt_ids
    from reft.baseline_auto import generate
    from reft.site import Site
    t_start = time.time()
    C, C1, P = c20(), c21(), cfg()
    fz = check_freeze()
    audit = read_json(art("p2_0/data_audit.json"))
    assert audit["valid"], "data audit invalid"
    tm = read_json(art("p2_0/train_manifest.json"))
    dm = read_json(art("p2_0/dev_manifest.json"))
    assert sha(art("p2_0/train_manifest.json")) == C1["data"]["train_manifest_sha256"]
    assert sha(art("p2_0/dev_manifest.json")) == C1["data"]["dev_manifest_sha256"]
    p0 = {o["id"]: o for o in read_jsonl(art("p0/auto_outputs.jsonl"))}
    firewall(GPU_FORBIDDEN)
    torch.manual_seed(C1["training"]["seed"])
    b, env = load_backbone()
    w0 = weights_hash(b.model)
    torch.cuda.reset_peak_memory_stats()
    R = {}
    train, dev = tm["rows"], dm["rows"]
    eos = b.processor.tokenizer.eos_token_id
    L = C1["methods"]["R2"]["layer"]
    S = C["smoke"]
    tiny = sorted(train, key=lambda r: r["utterance_id"])[:S["tf_rows"]]

    # ---- A. model and site identity
    _, enc = encode(b, [tiny[0]["audio_path"]])
    X, _ = build_tf_batch([tiny[0]["prompt"]], [tiny[0]["target_ids"]], eos, eos)
    X = X.to(b.device)
    lay, got = b.model.model.decoder.layers[L], {}
    hk = [lay.register_forward_hook(lambda m_, a_, o_: got.__setitem__("block", o_[0] if isinstance(o_, tuple) else o_)),
          b.model.model.decoder.layer_norm.register_forward_pre_hook(lambda m_, a_: got.__setitem__("ln_in", a_[0]))]
    try:
        with torch.no_grad(), Site(b.model, L, record=True) as s:
            b.model(encoder_outputs=(enc,), decoder_input_ids=X, use_cache=False)
            h = s.ffn_in
            ffn = lay.fc2(lay.activation_fn(lay.fc1(lay.final_layer_norm(h))))
            R["A"] = {"decoder_layers": len(b.model.model.decoder.layers), "d_model": b.d_model, "layer_is_last": L == len(b.model.model.decoder.layers) - 1,
                      "h_equals_q_plus_u": bool(torch.equal(h, s.q + s.s)),
                      "block_equals_h_plus_ffn_exact": bool(torch.equal(got["block"], h + ffn)),
                      "block_equals_h_plus_ffn_close": bool(torch.allclose(got["block"].float(), (h + ffn).float(), atol=1e-2)),
                      "final_ln_consumes_block_output": bool(torch.equal(got["block"], got["ln_in"])),
                      "site_differs_from_block": not bool(torch.equal(got["block"], h)),
                      "weights_equal_expected": w0 == P["model"]["expected_weights_hash"]}
    finally:
        for x in hk:
            x.remove()
    libs = loaded_cuda_libs()
    R["F_libs"] = {"cuda_libs": libs, "consistent": cublas_consistent(libs)}

    # ---- G2. frozen Auto: native vs two-stage on dev150 (beam-5), greedy two-stage on all, native greedy subset
    par = []
    base_b5, base_gr = [], []
    t0 = time.time()
    for k, r in enumerate(dev):
        nat, inp = generate(b, r["audio_path"], None, 5)
        lid = frozen_lid(b, inp["input_features"])
        two, _ = decode_frozen_lid(b, r["audio_path"], lid, 5)
        gr, _ = decode_frozen_lid(b, r["audio_path"], lid, 1)
        o = p0[r["utterance_id"]]
        d = {"id_index": k, "lid_equal_manifest": lid["token"] == r["frozen_lid_token"], "lid_equal_native_prompt": lid["token"] == nat["prompt"][1],
             "native_equals_two_stage": nat["tokens"] == two["tokens"] and nat["prompt"] == two["prompt"],
             "two_stage_equals_p0_tokens": two["tokens"] == o["tokens"], "termination_equal_p0": two["terminated"] == o["terminated"],
             "text_hash_equal_p0": hashlib.sha256(two["text"].encode()).hexdigest() == hashlib.sha256(o["text"].encode()).hexdigest()}
        if k < S["native_greedy_rows"]:
            ng, _ = generate(b, r["audio_path"], None, 1)
            d["native_greedy_equals_two_stage"] = ng["tokens"] == gr["tokens"]
        par.append(d)
        base_b5.append({"id": r["utterance_id"], **two})
        base_gr.append({"id": r["utterance_id"], **gr})
    write_jsonl(art("p2_0/frozen_baseline_dev150_beam5.jsonl"), base_b5)
    write_jsonl(art("p2_0/frozen_baseline_dev150_greedy.jsonl"), base_gr)
    agg = {k: sum(bool(x.get(k)) for x in par) for k in par[0]}
    agg.pop("id_index", None)
    agg["native_greedy_equals_two_stage"] = sum(bool(x.get("native_greedy_equals_two_stage")) for x in par)
    R["G2"] = {"n": len(par), "counts": agg, "seconds": time.time() - t0,
               "mismatch_indices": [x["id_index"] for x in par if not (x["native_equals_two_stage"] and x["two_stage_equals_p0_tokens"])]}
    # training-population LID cache
    t0 = time.time()
    lid_ok = 0
    from csasr.models.whisper import batch_model_inputs
    for r in train:
        inp = batch_model_inputs(b, [r["audio_path"]])
        lid_ok += frozen_lid(b, inp["input_features"])["token"] == r["frozen_lid_token"]
    R["G2"]["train_lid_recomputed_equal"] = [lid_ok, len(train)]
    R["G2"]["train_lid_seconds"] = time.time() - t0
    b5 = {x["id"]: x for x in base_b5}
    grd = {x["id"]: x for x in base_gr}

    # ---- B. zero-update parity (both adapters)
    R["B"], R["E"], R["D"], R["C"] = {}, {}, {}, {}
    masks = suppress_masks(b)
    for meth in ("R2", "L1"):
        ad = make_adapter(meth, b.model, C1, b.device)
        ex = C1["methods"][meth]["expected_trainable_parameters"]
        out = {"n_trainable": ad.n_trainable(), "expected": ex,
               "backbone_trainable": sum(p.requires_grad for p in b.model.parameters())}
        from csasr.models.whisper import batch_model_inputs
        Xb, Yb = build_tf_batch([r["prompt"] for r in tiny], [r["target_ids"] for r in tiny], eos, eos)
        _, encb = encode(b, [r["audio_path"] for r in tiny])
        Xb = Xb.to(b.device)
        lf = tf_forward(b, None, encb, Xb)
        la = tf_forward(b, ad, encb, Xb)
        out["tf_logits_bitwise_equal"] = bool(torch.equal(lf, la))
        lid_eq = 0
        for r in dev[:S["zero_lid_rows"]]:
            inp = batch_model_inputs(b, [r["audio_path"]])
            with ad.active(b.model):
                lid_eq += frozen_lid(b, inp["input_features"], ad)["token"] == r["frozen_lid_token"]
        out["lid_unchanged_with_adapter_installed"] = [lid_eq, S["zero_lid_rows"]]
        t0 = time.time()
        eq5 = eqg = 0
        for k, r in enumerate(dev):
            lid = {"token": r["frozen_lid_token"], "language": {v: kk for kk, v in b.model.generation_config.lang_to_id.items()}[r["frozen_lid_token"]]}
            rec, _ = decode_frozen_lid(b, r["audio_path"], lid, 5, ad)
            eq5 += rec["tokens"] == b5[r["utterance_id"]]["tokens"]
            if k < S["zero_greedy_rows"]:
                g1, _ = decode_frozen_lid(b, r["audio_path"], lid, 1, ad)
                eqg += g1["tokens"] == grd[r["utterance_id"]]["tokens"]
        out["beam5_tokens_equal_frozen"] = [eq5, len(dev)]
        out["greedy_tokens_equal_frozen"] = [eqg, S["zero_greedy_rows"]]
        out["decode_seconds"] = time.time() - t0
        out["hooks_after"] = site_hook_count(b.model, L)
        out["lora_targets_restored"] = all(not hasattr(getattr(b.model.model.decoder.layers[L].encoder_attn, n), "A") for n in ("q_proj", "v_proj"))
        R["B"][meth] = out

        # ---- C. TF alignment / causality (random nonzero adapter) and D. cache / beam consistency
        rnd = make_adapter(meth, b.model, C1, b.device)
        randomize(rnd, S["random_scale_w"][meth], S["random_scale_b"], seed=7)
        cc = {"greedy_cache_vs_tf": [], "prefix_logits_unchanged": None}
        for r in dev[:S["cache_rows"]]:
            lid = {"token": r["frozen_lid_token"], "language": {v: kk for kk, v in b.model.generation_config.lang_to_id.items()}[r["frozen_lid_token"]]}
            g1, _ = decode_frozen_lid(b, r["audio_path"], lid, 1, rnd)
            _, e1 = encode(b, [r["audio_path"]])
            Xs = torch.tensor([g1["prompt"] + g1["tokens"][:-1]], device=b.device)
            lg = tf_forward(b, rnd, e1, Xs)[0].float()
            am = [int(lg[3 + t].masked_fill(masks[1] if t == 0 else masks[0], -float("inf")).argmax()) for t in range(len(g1["tokens"]))]
            mis = sum(x != y for x, y in zip(am, g1["tokens"]))
            sh = sum(1 for t in range(len(g1["tokens"]) - 1) if am[t] != g1["tokens"][t + 1])
            fz_ = decode_frozen_lid(b, r["audio_path"], lid, 1)[0]
            cc["greedy_cache_vs_tf"].append({"n": len(am), "mismatch": mis, "shifted_mismatch": sh, "changed_vs_frozen": g1["tokens"] != fz_["tokens"]})
        lf = tf_forward(b, None, encb, Xb)
        la = tf_forward(b, rnd, encb, Xb)
        cc["prefix_positions_0_2_unchanged"] = bool(torch.equal(lf[:, :3], la[:, :3])) if meth == "R2" else "n/a (LoRA acts inside cross-attention at every position; LID is frozen separately)"
        cc["content_positions_changed"] = not bool(torch.equal(lf[:, 3:], la[:, 3:]))
        if meth == "R2":
            # causality at a content position p: an update applied only from p onward leaves logits < p unchanged
            from reft.adapters import LoReFTAdapter
            p = 8
            loc = LoReFTAdapter(b.d_model, C1["methods"]["R2"]["rank"], L, C1["methods"]["R2"]["init_seed"], start_pos=p, device=b.device)
            loc.load_state_dict(rnd.state_dict())
            lp_ = tf_forward(b, loc, encb, Xb)
            cc["update_from_p_leaves_earlier_logits"] = bool(torch.equal(lf[:, :p], lp_[:, :p])) and not bool(torch.equal(lf[:, p], lp_[:, p]))
            # absolute positions seen by the hook in cached decoding
            seen = []
            orig = rnd._layer_pre

            def spy(mod, args, kwargs):
                seen.append(kwargs["cache_position"].tolist())
                return orig(mod, args, kwargs)
            rnd._layer_pre = spy
            lid = {"token": dev[0]["frozen_lid_token"], "language": {v: kk for kk, v in b.model.generation_config.lang_to_id.items()}[dev[0]["frozen_lid_token"]]}
            gg, _ = decode_frozen_lid(b, dev[0]["audio_path"], lid, 1, rnd)
            n_seen_g = len(seen)
            seen_g = list(seen)
            seen.clear()
            bb, _ = decode_frozen_lid(b, dev[0]["audio_path"], lid, 5, rnd)
            rnd._layer_pre = orig
            cc["greedy_cache_positions"] = {"first": seen_g[0], "steps": n_seen_g,
                                            "incremental_ok": all(seen_g[i] == [seen_g[0][-1] + i] for i in range(1, n_seen_g)),
                                            "n_tokens": len(gg["tokens"])}
            cc["beam5_cache_positions"] = {"first": seen[0], "steps": len(seen),
                                           "incremental_ok": all(seen[i] == [seen[0][-1] + i] for i in range(1, len(seen)))}
        R["C"][meth] = cc
        beam = []
        for tag, adp in (("zero", ad), ("random", rnd)):
            for r in dev[:S["beam_rows"]]:
                code = {v: kk for kk, v in b.model.generation_config.lang_to_id.items()}[r["frozen_lid_token"]].strip("<|>")
                out_ = gen_with_scores(b, r["audio_path"], code, 5, adp)
                seq = out_.sequences[0].tolist()
                toks = seq[4:]
                if eos in toks:
                    toks = toks[:toks.index(eos) + 1]
                vals = []
                for t, y in enumerate(toks):
                    v = out_.scores[t]
                    if v.dim() == 2:
                        v = v[0] if v.shape[0] == 1 else v[int(out_.beam_indices[0, t])]
                    vals.append(float(v[y]))
                _, e1 = encode(b, [r["audio_path"]])
                Xs = torch.tensor([seq[:4] + toks[:-1]], device=b.device)
                lp = F.log_softmax(tf_forward(b, adp, e1, Xs)[0].float(), -1)[3:]
                tf_ = lp.gather(-1, torch.tensor(toks, device=b.device)[:, None])[:, 0].cpu()
                ts = torch.tensor(vals)
                beam.append({"adapter": tag, "n": len(toks), "max_abs_diff": float((ts - tf_).abs().max()),
                             "mean_abs_diff": float((ts - tf_).abs().mean())})
        R["D"][meth] = {"beam5_transition_vs_tf": beam}
        lid = {"token": dev[0]["frozen_lid_token"], "language": {v: kk for kk, v in b.model.generation_config.lang_to_id.items()}[dev[0]["frozen_lid_token"]]}
        x1, _ = decode_frozen_lid(b, dev[0]["audio_path"], lid, 5, rnd)
        lid2 = {"token": dev[1]["frozen_lid_token"], "language": {v: kk for kk, v in b.model.generation_config.lang_to_id.items()}[dev[1]["frozen_lid_token"]]}
        decode_frozen_lid(b, dev[1]["audio_path"], lid2, 5, rnd)
        x2, _ = decode_frozen_lid(b, dev[0]["audio_path"], lid, 5, rnd)
        R["D"][meth]["xyx_no_state_leakage"] = x1["tokens"] == x2["tokens"]
        R["D"][meth]["hooks_after"] = site_hook_count(b.model, L)
        # batch vs batch-1 teacher forcing (random adapter)
        lb = tf_forward(b, rnd, encb, Xb).float()
        diffs, agree, tot = [], 0, 0
        for i, r in enumerate(tiny):
            n = 3 + len(r["target_ids"])
            l1 = tf_forward(b, rnd, encb[i:i + 1], Xb[i:i + 1, :n]).float()
            diffs.append(float((lb[i, 3:n] - l1[0, 3:n]).abs().max()))
            agree += int((lb[i, 3:n].argmax(-1) == l1[0, 3:n].argmax(-1)).sum())
            tot += n - 3
        R["D"][meth]["batch_vs_single_tf"] = {"max_abs_logit_diff": max(diffs), "argmax_agreement": agree / tot}

        # ---- E. one optimizer step
        tr = C1["training"]
        opt = torch.optim.AdamW(ad.parameters(), lr=C1["methods"][meth]["lr"], betas=tuple(tr["betas"]), eps=tr["eps"],
                                weight_decay=tr["weight_decay"])
        loss, ntok, _, _, _ = batch_loss(b, ad, tiny, grad=True)
        loss.backward()
        grads = {n_: (float(p.grad.norm()) if p.grad is not None else None) for n_, p in
                 (ad.module.named_parameters() if meth == "R2" else [(f"{k}.{q}", getattr(v, q)) for k, v in ad.wrapped.items() for q in ("A", "B")])}
        gn = float(torch.nn.utils.clip_grad_norm_(ad.parameters(), tr["clip_norm"]))
        bb_grad = [n_ for n_, p in b.model.named_parameters() if p.grad is not None or p.requires_grad]
        opt.step()
        opt.zero_grad(set_to_none=True)
        # second backward (no second step): with LoRA, A receives gradient only once B != 0
        loss2, _, _, _, _ = batch_loss(b, ad, tiny, grad=True)
        loss2.backward()
        grads2 = {n_: float(p.grad.norm()) if p.grad is not None else None for n_, p in
                  (ad.module.named_parameters() if meth == "R2" else [(f"{k}.{q}", getattr(v, q)) for k, v in ad.wrapped.items() for q in ("A", "B")])}
        opt.zero_grad(set_to_none=True)
        la2 = tf_forward(b, ad, encb, Xb)
        lf2 = tf_forward(b, None, encb, Xb)
        sdict = ad.state_dict()
        ck = art(f"p2_0/smoke_{meth}_adapter.pt")
        torch.save(sdict, ck)
        ad2 = make_adapter(meth, b.model, C1, b.device)
        ad2.load_state_dict(torch.load(ck))
        la3 = tf_forward(b, ad2, encb, Xb)
        e = {"loss0": float(loss), "tokens": ntok, "grad_norms": grads,
             "grads_step0_finite": all(v is not None and v == v and v != float("inf") for v in grads.values()),
             "grads_step0_nonzero": {k_: bool(v and v > 0) for k_, v in grads.items()},
             "grads_after_step_all_nonzero": all(v is not None and v > 0 and v == v for v in grads2.values()), "grad_norms_after_step": grads2,
             "loss_after_step": float(loss2),
             "grad_norm_before_clip": gn, "backbone_params_with_grad_or_trainable": len(bb_grad),
             "update_effective": float((la2.float() - lf2.float()).abs().max()), "frozen_logits_unchanged": bool(torch.equal(lf2, lf)),
             "save_load_bitwise": bool(torch.equal(la2, la3)), "checkpoint_sha256": sha(ck)}
        e["update_nonzero"] = e["update_effective"] > 0
        if meth == "R2":
            Rm = ad.module.rot.weight.detach()
            e["R_rows_orthonormal_after_step"] = float((Rm @ Rm.T - torch.eye(Rm.shape[0], device=Rm.device)).abs().max())
        R["E"][meth] = e
        # ---- F. runtime of one micro-batch (forward + backward), warm
        mb = sorted(train, key=lambda r: r["utterance_id"])[:tr["micro_batch"]]
        ts_ = []
        for _ in range(S["timing_reps"]):
            torch.cuda.synchronize()
            t0 = time.time()
            l_, _, _, _, _ = batch_loss(b, ad, mb, grad=True)
            l_.backward()
            torch.cuda.synchronize()
            ts_.append(time.time() - t0)
            opt.zero_grad(set_to_none=True)
        R.setdefault("F", {})[meth] = {"micro_batch_seconds": ts_, "micro_batch": tr["micro_batch"]}
        del ad, ad2, rnd, opt
    w1 = weights_hash(b.model)
    peak = int(torch.cuda.max_memory_allocated())
    # projection for P2-1 (per method): 3 epochs x ceil(500/8) micro-batches + per-epoch dev decode (beam5 + greedy + LID)
    nmb = -(-len(train) // C1["training"]["micro_batch"])
    dev_s = (sum(x["seconds"] for x in base_b5) + sum(x["seconds"] for x in base_gr)) / len(dev) + 0.15  # + LID re-check
    proj = {m: C1["training"]["epochs"] * (nmb * (sum(R["F"][m]["micro_batch_seconds"][1:]) / max(1, len(R["F"][m]["micro_batch_seconds"]) - 1))
                                           + len(dev) * dev_s) + len(dev) * dev_s + 180 for m in ("R2", "L1")}
    R["F"].update({"peak_gpu_bytes": peak, "projected_p2_1_seconds_per_method": proj, "smoke_wall_seconds": time.time() - t_start,
                   "weights_unchanged": w0 == w1, "precision": "backbone bf16 (frozen, eager attention); adapter parameters and update float32, cast to bf16 at the site"})
    # ---- gates evaluated here (engineering)
    g = {}
    g["A_site"] = all(v for k, v in R["A"].items() if isinstance(v, bool) and k != "block_equals_h_plus_ffn_exact") and R["A"]["decoder_layers"] == 32 and R["A"]["d_model"] == 1280
    c = R["G2"]["counts"]
    n = R["G2"]["n"]
    g["G2_auto"] = (c["lid_equal_manifest"] == n and c["lid_equal_native_prompt"] == n and c["native_equals_two_stage"] == n
                    and c["two_stage_equals_p0_tokens"] == n and c["termination_equal_p0"] == n and c["text_hash_equal_p0"] == n
                    and c["native_greedy_equals_two_stage"] == S["native_greedy_rows"]
                    and R["G2"]["train_lid_recomputed_equal"][0] == len(train))
    g["B_zero_update"] = all(x["n_trainable"] == x["expected"] and x["backbone_trainable"] == 0 and x["tf_logits_bitwise_equal"]
                             and x["lid_unchanged_with_adapter_installed"][0] == S["zero_lid_rows"]
                             and x["beam5_tokens_equal_frozen"][0] == len(dev) and x["greedy_tokens_equal_frozen"][0] == S["zero_greedy_rows"]
                             and x["hooks_after"] == 0 and x["lora_targets_restored"] for x in R["B"].values())
    g["C_alignment"] = all(all(q["mismatch"] <= 0.02 * q["n"] and q["shifted_mismatch"] >= 0.5 * (q["n"] - 1) for q in R["C"][m]["greedy_cache_vs_tf"])
                           and R["C"][m]["content_positions_changed"] for m in R["C"]) \
        and R["C"]["R2"]["prefix_positions_0_2_unchanged"] is True and R["C"]["R2"]["update_from_p_leaves_earlier_logits"] \
        and R["C"]["R2"]["greedy_cache_positions"]["first"] == [0, 1, 2, 3] and R["C"]["R2"]["greedy_cache_positions"]["incremental_ok"] \
        and R["C"]["R2"]["beam5_cache_positions"]["first"] == [0, 1, 2, 3] and R["C"]["R2"]["beam5_cache_positions"]["incremental_ok"]
    tol = S["beam_score_tol"]
    g["D_beam_kv"] = all(all(x["max_abs_diff"] <= tol["max"] and x["mean_abs_diff"] <= tol["mean"] for x in R["D"][m]["beam5_transition_vs_tf"])
                         and R["D"][m]["xyx_no_state_leakage"] and R["D"][m]["hooks_after"] == 0
                         and R["D"][m]["batch_vs_single_tf"]["argmax_agreement"] >= S["batch_argmax_agreement_min"] for m in R["D"])
    g["E_one_step"] = all(x["grads_step0_finite"] and x["grads_after_step_all_nonzero"]
                          and all(v for k_, v in x["grads_step0_nonzero"].items() if not k_.endswith(".A"))
                          and x["backbone_params_with_grad_or_trainable"] == 0 and x["update_nonzero"]
                          and x["frozen_logits_unchanged"] and x["save_load_bitwise"] for x in R["E"].values()) \
        and R["E"]["R2"]["R_rows_orthonormal_after_step"] < 1e-5
    g["F_compute"] = R["F_libs"]["consistent"] and R["F"]["weights_unchanged"] and peak < S["vram_limit_bytes"] \
        and all(v < S["job_limit_seconds"] for v in proj.values())
    res = {"schema": "reft_p2_0_smoke_v1", "utc": now(), **fz, "git": git_state(), "env": env, "slurm": slurm(), "results": R,
           "engineering_gates": g, "all_pass": all(g.values())}
    out = art(f"p2_0/smoke_{slurm()['SLURM_JOB_ID'] or 'local'}.json")
    write_json(out, res)
    print(json.dumps({"engineering_gates": g, "all_pass": res["all_pass"],
                      "summary": {k: v for k, v in R.items() if k not in ("F_libs",)}}, indent=1, default=str)[:20000])


# ====================================================================== train (GPU, P2-1; not executed in P2-0)

def run_train(method: str):
    import numpy as np
    import torch
    from reft.adapters import make_adapter
    from reft.auto_policy import decode_frozen_lid, frozen_lid
    C1 = c21()
    fz = check_freeze()
    auth = art("p2_1/AUTHORIZATION.json")
    if not auth.exists():
        raise SystemExit(f"P2-1 training is not authorized: {auth} is absent (project-owner approval required)")
    assert sha(art("p2_0/train_manifest.json")) == C1["data"]["train_manifest_sha256"]
    assert sha(art("p2_0/dev_manifest.json")) == C1["data"]["dev_manifest_sha256"]
    tm, dm = read_json(art("p2_0/train_manifest.json"))["rows"], read_json(art("p2_0/dev_manifest.json"))["rows"]
    base = {x["id"]: x for x in read_jsonl(art("p2_0/frozen_baseline_dev150_beam5.jsonl"))}
    out = art(f"p2_1/{method}")
    if out.exists():
        raise SystemExit(f"{out} exists; runs are write-once")
    firewall(GPU_FORBIDDEN)
    tr = C1["training"]
    torch.manual_seed(tr["seed"])
    b, env = load_backbone()
    w0 = weights_hash(b.model)
    ad = make_adapter(method, b.model, C1, b.device)
    assert ad.n_trainable() == C1["methods"][method]["expected_trainable_parameters"]
    opt = torch.optim.AdamW(ad.parameters(), lr=C1["methods"][method]["lr"], betas=tuple(tr["betas"]), eps=tr["eps"],
                            weight_decay=tr["weight_decay"])
    inv = {v: k for k, v in b.model.generation_config.lang_to_id.items()}
    torch.cuda.reset_peak_memory_stats()
    t_all = time.time()
    hist, steps, tokens, train_s = [], 0, 0, 0.0

    def dev_decode(ep):
        res = {"beam5": [], "greedy": []}
        t0 = time.time()
        for r in dm:
            lid = {"token": r["frozen_lid_token"], "language": inv[r["frozen_lid_token"]]}
            for reg, nb in (("beam5", 5), ("greedy", 1)):
                rec, inp = decode_frozen_lid(b, r["audio_path"], lid, nb, ad)
                res[reg].append({"id": r["utterance_id"], **rec})
            with ad.active(b.model):
                assert frozen_lid(b, inp["input_features"], ad)["token"] == r["frozen_lid_token"], "LID changed under adapter"
        for reg, v in res.items():
            write_jsonl(out / f"epoch{ep}" / f"dev_{reg}.jsonl", v)
        return time.time() - t0, res

    sec, res0 = dev_decode(0)
    ident = sum(x["tokens"] == base[x["id"]]["tokens"] for x in res0["beam5"])
    if ident != len(dm):
        raise SystemExit(f"zero-init adapter does not reproduce the frozen baseline ({ident}/{len(dm)})")
    hist.append({"epoch": 0, "zero_init_identity_beam5": ident, "dev_decode_seconds": sec})
    mb, acc = tr["micro_batch"], tr["grad_accum"]
    for ep in range(1, tr["epochs"] + 1):
        order = np.random.default_rng(tr["seed"] + ep).permutation(len(tm))
        batches = [[tm[i] for i in order[j:j + mb]] for j in range(0, len(order), mb)]
        t0 = time.time()
        run_loss, run_tok, pending, gnorms = 0.0, 0, 0, []
        for k, rows in enumerate(batches):
            loss, ntok, _, _, _ = batch_loss(b, ad, rows, grad=True)
            (loss / acc).backward()
            run_loss += float(loss) * ntok
            run_tok += ntok
            pending += 1
            if pending == acc or k == len(batches) - 1:
                gnorms.append(float(torch.nn.utils.clip_grad_norm_(ad.parameters(), tr["clip_norm"])))
                opt.step()
                opt.zero_grad(set_to_none=True)
                steps += 1
                pending = 0
        torch.cuda.synchronize()
        ep_s = time.time() - t0
        train_s += ep_s
        tokens += run_tok
        (out / f"epoch{ep}").mkdir(parents=True, exist_ok=True)
        torch.save(ad.state_dict(), out / f"epoch{ep}" / "adapter.pt")
        sec, _ = dev_decode(ep)
        hist.append({"epoch": ep, "train_ce": run_loss / run_tok, "updates_total": steps, "grad_norm_mean": float(np.mean(gnorms)),
                     "train_seconds": ep_s, "dev_decode_seconds": sec, "adapter_sha256": sha(out / f"epoch{ep}" / "adapter.pt")})
        print(json.dumps(hist[-1]), flush=True)
    w1 = weights_hash(b.model)
    lat = {reg: float(np.mean([x["seconds"] for x in read_jsonl(out / f"epoch{tr['epochs']}" / f"dev_{reg}.jsonl")])) for reg in ("beam5", "greedy")}
    man = {"schema": "reft_p2_1_run_v1", "utc": now(), **fz, "git": git_state(), "env": env, "slurm": slurm(), "method": method,
           "config": C1["methods"][method], "training": tr, "history": hist, "weights_before": w0, "weights_after": w1,
           "weights_unchanged": w0 == w1, "cuda_libs": loaded_cuda_libs(),
           "efficiency": {"trainable_parameters": ad.n_trainable(), "adapter_bytes_fp32": 4 * ad.n_trainable(), "optimizer_updates": steps,
                          "supervised_tokens": tokens, "train_seconds": train_s, "wall_seconds": time.time() - t_all,
                          "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()), "mean_decode_seconds": lat,
                          "baseline_mean_decode_seconds_beam5": float(np.mean([x["seconds"] for x in base.values()]))}}
    write_json(out / "manifest.json", man)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "audit", "smoke", "train"])
    ap.add_argument("--method", choices=["R2", "L1"])
    a = ap.parse_args(argv)
    if a.cmd == "train":
        run_train(a.method)
    else:
        {"freeze": run_freeze, "audit": run_audit, "smoke": run_smoke}[a.cmd]()


if __name__ == "__main__":
    main()
