#!/usr/bin/env python
"""P1-A -- gradient capture at the frozen layer-24 post-cross-attention site.

  smoke    -- GPU engineering acceptance on 4 construction utterances (no accuracy is evaluated; P1 stops on failure).
  capture  -- GPU: F1 Auto beam-5 decode of the P1 panel behind the reference firewall, then (training labels opened)
              teacher-forced float32 passes: rows A (free-decoding first divergence), B (teacher-forced gold-prefix
              errors, secondary) and P (preservation controls); per row the margin gradient q, the site activation h
              and the script-readout gradient g_s.

Margins are logit differences m = z(correct) - z(competitor) (= log-probability differences). Gradients come from
torch.autograd.grad with respect to a zero delta added at the site (reft/site.py); backbone weights never require grad.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.baseline_auto import FORBIDDEN, generate, language_scores, load_model, special_ids  # noqa: E402
from reft.common import art, audio_hash_ok, cfg, fillers, firewall, firewall_off, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from reft.provenance import check_freeze, git_state, now, sha, slurm, weights_hash  # noqa: E402
from reft.site import Site, n_hooks  # noqa: E402
from reft.text import (fd_category, first_divergence, is_han, script_of, tok_bytes_fn, token_is_lexical,  # noqa: E402
                       token_unit_map, tokens_format_equivalent, units, utterance_stratum)

P0 = 4  # prompt length [sot, lang, transcribe, notimestamps]; logits at index P0-1+t predict content token t


# ====================================================================== vocabulary helpers

def vocab_sets(tb, n_vocab):
    """Latin: all-ASCII token bytes with a letter. Han: contains a Han character, or only non-ASCII bytes that do not
    decode to a complete character (partial UTF-8). Special tokens (empty bytes) excluded."""
    latin, han = [], []
    for i in range(n_vocab):
        b = tb(i)
        if not b:
            continue
        if all(x < 0x80 for x in b):
            if any(chr(x).isalpha() for x in b):
                latin.append(i)
        else:
            s = b.decode("utf-8", "ignore")
            if any(is_han(c) for c in s) or (all(x >= 0x80 for x in b) and not s):
                han.append(i)
    return latin, han


def suppress_masks(b, device):
    import torch
    gen = b.model.generation_config
    V = b.model.config.vocab_size
    m = torch.zeros(V, dtype=torch.bool, device=device)
    m[list(gen.suppress_tokens or [])] = True
    m0 = m.clone()
    m0[list(gen.begin_suppress_tokens or [])] = True
    return m, m0


def masked(logits, t, masks):
    """logits [V] at content position t; Whisper's suppress tokens (+ begin-suppress at t == 0) set to -inf."""
    import torch
    m, m0 = masks
    return logits.masked_fill(m0 if t == 0 else m, -torch.inf)


# ====================================================================== model passes

def to_float32(b):
    import torch
    b.model.float()
    b.dtype = torch.float32
    for p in b.model.parameters():
        p.requires_grad_(False)
    return b


def encode(b, path):
    import torch
    from csasr.models.whisper import batch_model_inputs
    with torch.no_grad():
        feats = batch_model_inputs(b, [path])["input_features"]
        return b.model.model.encoder(feats).last_hidden_state


def tf_logits(b, enc, X, delta=None, record=False):
    """Teacher-forced decoder pass; enc [1, F, d] is expanded to the batch of X."""
    L = cfg()["p1"]["layer"]
    site = Site(b.model, L, delta=delta, record=record)
    with site:
        out = b.model(encoder_outputs=(enc.expand(X.shape[0], -1, -1),), decoder_input_ids=X, use_cache=False).logits
    return out, site


# ====================================================================== rows

def build_rows(uid, tgt, base, argmax, runner, tb, eos, FILL, Pmax_b, Pmax_p):
    """Rows for one utterance. ``argmax``/``runner``: masked top-1/top-2 ids at content positions 0..T-1 under teacher
    forcing on the target. ``base``: AUTO content tokens (incl. EOS when terminated)."""
    from csasr.data.language_tags import tag_unit
    T = len(tgt["target_ids"])
    ids = tgt["target_ids"]
    fd = first_divergence(ids, base, tb, eos, lambda u: u in FILL)
    rows = []
    a_pos = None
    if fd["cls"] == "LEXICAL_FIRST" and fd.get("sub") == "TOKEN":
        k = fd["k"]
        a_pos = k
        rows.append({"source": "A", "t": k, "correct": ids[k], "competitor": base[k], "category": fd_category(fd),
                     "lang": tag_unit(fd["ref_unit"]) if fd.get("ref_unit") else "NONE",
                     "ref_unit": fd.get("ref_unit"), "base_unit": fd.get("base_unit")})
    umap = token_unit_map(tgt["target_text"], ids[:-1], tb)
    U = units(tgt["target_text"])

    def info(t):
        if umap is None or t >= len(umap) or not umap[t]:
            return None
        su = [U[j] for j in umap[t]]
        tags = {tag_unit(x) for x in su}
        return {"units": su, "lang": "ZH" if tags == {"ZH"} else "EN" if tags == {"EN"} else "OTHER",
                "filler": any(x in FILL for x in su)}

    nb = 0
    for t in range(T - 1):
        if nb >= Pmax_b:
            break
        if t == a_pos:
            continue
        tok, am = ids[t], argmax[t]
        if am == tok or am == eos or not (token_is_lexical(tb(tok)) and token_is_lexical(tb(am))) \
                or tokens_format_equivalent(tb(tok), tb(am)):
            continue
        inf = info(t)
        if inf is None or inf["filler"]:
            continue
        sc = script_of(tb(am).decode("utf-8", "ignore")) if tb(am).decode("utf-8", "ignore") else "han"
        cat = {("EN", "han"): "translation_like", ("EN", "latin"): "en_lexical_substitution",
               ("ZH", "han"): "zh_lexical_substitution", ("ZH", "latin"): "zh_to_latin"}.get((inf["lang"], sc), "other")
        rows.append({"source": "B", "t": t, "correct": tok, "competitor": am, "category": cat, "lang": inf["lang"],
                     "ref_unit": " ".join(inf["units"]), "base_unit": None})
        nb += 1
    prefix = range(T - 1) if fd["cls"] == "IDENTICAL" else range(min(fd["k"], T - 1)) if fd.get("k") is not None else range(0)
    cand = []
    for t in prefix:
        tok = ids[t]
        if argmax[t] != tok or not token_is_lexical(tb(tok)):
            continue
        inf = info(t)
        if inf is None or inf["filler"] or inf["lang"] not in ("EN", "ZH"):
            continue
        cand.append((hashlib.sha256(f"reft-pres:{uid}:{t}".encode()).hexdigest(), t, inf))
    for _, t, inf in sorted(cand)[:Pmax_p]:
        rows.append({"source": "P", "t": t, "correct": ids[t], "competitor": runner[t], "category": f"correct_{inf['lang'].lower()}",
                     "lang": inf["lang"], "ref_unit": " ".join(inf["units"]), "base_unit": None})
    return fd, rows


def utterance_pass(b, enc, prompt, tgt, masks, latin, han, want_rows):
    """Teacher-forced float32 pass with a differentiable site. Returns masked argmax/runner-up per content position and a
    closure computing (m, q, h, g_s) for a row."""
    import torch
    ids = tgt["target_ids"]
    X = torch.tensor([prompt + ids[:-1]], device=b.device)
    delta = torch.zeros(1, X.shape[1], b.d_model, device=b.device, dtype=torch.float32, requires_grad=True)
    logits, site = tf_logits(b, enc, X, delta=delta, record=True)
    z = logits[0].float()
    am, ru = [], []
    for t in range(len(ids)):
        zz = masked(z[P0 - 1 + t].detach(), t, masks)
        top = zz.topk(2).indices.tolist()
        am.append(top[0])
        ru.append(top[1])
    h_site = site.ffn_in.detach()[0]

    def grads(row):
        pos = P0 - 1 + row["t"]
        zr = z[pos]
        m = zr[row["correct"]] - zr[row["competitor"]]
        q = torch.autograd.grad(m, delta, retain_graph=True)[0][0, pos].detach().cpu()
        s = torch.logsumexp(zr[latin], 0) - torch.logsumexp(zr[han], 0)
        gs = torch.autograd.grad(s, delta, retain_graph=True)[0][0, pos].detach().cpu()
        # vectors leave the GPU here: cuBLAS level-1/2 calls (e.g. cublasSdot) raise SIGFPE on this MIG slice in
        # float32 (amendment r3); all vector algebra runs on CPU
        return float(m.detach()), q, h_site[pos].cpu(), gs
    return am, ru, grads, (X, logits, site, delta)


def margin_at(b, enc, X, pos, correct, competitor, u):
    """m after adding u (CPU or device vector) at position pos only (no grad)."""
    import torch
    d = torch.zeros(1, X.shape[1], b.d_model, device=b.device)
    d[0, pos] = u.to(b.device, torch.float32)
    with torch.no_grad():
        lg, _ = tf_logits(b, enc, X, delta=d)
    return float(lg[0, pos, correct] - lg[0, pos, competitor])


# ====================================================================== smoke

def run_smoke():
    import torch
    C = cfg()
    P1 = C["p1"]
    fz = check_freeze("p1")
    panel = read_json(art("p1/panel.json"))["rows"]
    rows = [r for r in panel if r["partition"] == "construction"][:P1["smoke_utterances"]]
    firewall(FORBIDDEN)
    b, env = load_model()
    w0 = weights_hash(b.model)
    S = special_ids(b)
    torch.cuda.reset_peak_memory_stats()
    res = {"weights_equal_expected": w0 == C["model"]["expected_weights_hash"]}
    # --- 1. indexing: teacher-forced bf16 argmax vs cached greedy decode (no off-by-one)
    masks = suppress_masks(b, b.device)
    mism = shifted = steps = 0
    dec, t_dec = {}, []
    for r in rows:
        g, _ = generate(b, r["audio_path"], None, 1)
        enc = encode(b, r["audio_path"])
        toks = g["tokens"]
        X = torch.tensor([g["prompt"] + toks[:-1]], device=b.device)
        with torch.no_grad():
            lg, _ = tf_logits(b, enc, X)
        for t, y in enumerate(toks):
            am = int(masked(lg[0, P0 - 1 + t].float(), t, masks).argmax())
            mism += am != y
            shifted += t + 1 < len(toks) and am != toks[t + 1]
            steps += 1
        t0 = time.time()
        dec[r["utterance_id"]], _ = generate(b, r["audio_path"], None, C["decode"]["num_beams"])
        t_dec.append(time.time() - t0)
    res["greedy_tf_steps"] = steps
    res["greedy_tf_mismatch"] = mism
    res["greedy_tf_shifted_mismatch"] = shifted
    res["indexing_ok"] = mism <= 0.02 * steps and shifted >= 0.5 * steps
    # --- 2. labels opened; float32 local passes
    firewall_off()
    tg = read_json(art("p1/targets.json"))["targets"]
    to_float32(b)
    masks = suppress_masks(b, b.device)
    tb = tok_bytes_fn(b.processor.tokenizer)
    latin, han = vocab_sets(tb, b.model.config.vocab_size)
    latin_t, han_t = torch.tensor(latin, device=b.device), torch.tensor(han, device=b.device)
    FILL = fillers()
    gen = torch.Generator().manual_seed(C["seed"])
    checks = collections.defaultdict(list)
    t_tf = []
    for r in rows:
        uid = r["utterance_id"]
        tgt = tg[uid]
        enc = encode(b, r["audio_path"])
        t0 = time.time()
        am, ru, grads, (X, logits, site, delta) = utterance_pass(b, enc, dec[uid]["prompt"], tgt, masks, latin_t, han_t, True)
        with torch.no_grad():
            lg_plain = b.model(encoder_outputs=(enc,), decoder_input_ids=X, use_cache=False).logits
        checks["zero_delta_bitwise"].append(bool(torch.equal(lg_plain, logits.detach())))
        checks["ffn_in_equals_q_plus_s"].append(bool(torch.equal(site.ffn_in.detach(), (site.q + site.s).detach())))
        checks["site_called_once"].append(site.calls == 1)
        fd, rws = build_rows(uid, tgt, dec[uid]["tokens"], am, ru, tb, S["eos"], FILL, P1["source_b_max_per_utterance"],
                             P1["preservation_max_per_utterance"])
        if not rws:  # engineering fallback for the smoke only: first lexical gold position vs its runner-up
            t = next(t for t in range(len(tgt["target_ids"]) - 1) if token_is_lexical(tb(tgt["target_ids"][t])))
            rws = [{"source": "SMOKE", "t": t, "correct": tgt["target_ids"][t], "competitor": ru[t] if am[t] == tgt["target_ids"][t] else am[t]}]
        for row in rws[:3]:
            m0, q, h, gs = grads(row)
            pos = P0 - 1 + row["t"]
            qn = float(q.norm())
            checks["shape_1280"].append(tuple(q.shape) == (b.d_model,))
            checks["finite"].append(bool(torch.isfinite(q).all() and torch.isfinite(gs).all()))
            checks["nonzero"].append(qn > 0)
            eta = P1["fd_rel_step"] * float(h.norm())
            qh = q / q.norm()
            mp = margin_at(b, enc, X, pos, row["correct"], row["competitor"], eta * qh)
            mm = margin_at(b, enc, X, pos, row["correct"], row["competitor"], -eta * qh)
            m00 = margin_at(b, enc, X, pos, row["correct"], row["competitor"], torch.zeros_like(q))
            fd_q = (mp - mm) / (2 * eta)
            rv = torch.randn(b.d_model, generator=gen, dtype=torch.float64).float()
            rv = rv / rv.norm()
            fp = margin_at(b, enc, X, pos, row["correct"], row["competitor"], eta * rv)
            fm = margin_at(b, enc, X, pos, row["correct"], row["competitor"], -eta * rv)
            fd_r = (fp - fm) / (2 * eta)
            checks["fd_along_q_rel_err"].append(abs(fd_q - qn) / qn)
            checks["fd_random_abs_err_over_qnorm"].append(abs(fd_r - float(q @ rv)) / qn)
            checks["plus_increases"].append(mp > m00)
            checks["minus_decreases"].append(mm < m00)
            checks["m0_reproduced"].append(abs(m00 - m0) <= 1e-3 + 1e-3 * abs(m0))
            _, _, grads2, _ = utterance_pass(b, enc, dec[uid]["prompt"], tgt, masks, latin_t, han_t, True)
            checks["repeat_bitwise"].append(bool(torch.equal(grads2(row)[1], q)))
            checks["row_source"].append(row["source"])
        checks["param_grads_none"].append(all(p.grad is None for p in b.model.parameters()))
        checks["hooks_removed"].append(n_hooks(b.model, P1["layer"]) == 0)
        t_tf.append(time.time() - t0)
    tol = P1["fd_rel_tol"]
    res.update({k: v for k, v in checks.items()})
    res["fd_ok"] = all(e <= tol for e in checks["fd_along_q_rel_err"]) and all(e <= tol for e in checks["fd_random_abs_err_over_qnorm"])
    bool_keys = ["zero_delta_bitwise", "ffn_in_equals_q_plus_s", "site_called_once", "shape_1280", "finite", "nonzero",
                 "plus_increases", "minus_decreases", "m0_reproduced", "repeat_bitwise", "param_grads_none", "hooks_removed"]
    res["bool_checks_ok"] = {k: all(checks[k]) for k in bool_keys}
    w1 = weights_hash(b.model)
    res["weights_unchanged"] = w0 == w1
    n = len(read_json(art("p1/panel.json"))["rows"])
    per_dec = sum(t_dec) / len(t_dec)
    per_tf = sum(t_tf) / len(t_tf)
    res["timing"] = {"beam5_decode_s_per_utt": per_dec, "tf_smoke_s_per_utt_incl_fd": per_tf,
                     "projected_capture_s": n * (per_dec + per_tf), "peak_gpu_bytes": int(torch.cuda.max_memory_allocated())}
    res["runtime_ok"] = res["timing"]["projected_capture_s"] < 2.5 * 3600
    res["all_pass"] = bool(res["weights_equal_expected"] and res["indexing_ok"] and res["fd_ok"]
                           and all(res["bool_checks_ok"].values()) and res["weights_unchanged"] and res["runtime_ok"])
    out = {"schema": "reft_p1_smoke_v1", "utc": now(), **fz, "git": git_state(), "env": env, "slurm": slurm(),
           "utterances": [r["utterance_id"] for r in rows], **res}
    path = art(f"p1/smoke_{slurm()['SLURM_JOB_ID'] or 'local'}.json")
    write_json(path, out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("env",)}, indent=1, default=str))
    if not res["all_pass"]:
        raise SystemExit("P1 smoke FAILED")


# ====================================================================== capture

def latest_smoke():
    sm = sorted(art("p1").glob("smoke_*.json"), key=lambda p: p.stat().st_mtime)
    return sm[-1] if sm else None


def run_capture():
    import torch
    C = cfg()
    P1 = C["p1"]
    fz = check_freeze("p1")
    sm = latest_smoke()
    assert sm and read_json(sm)["all_pass"], "smoke has not passed"
    audit = read_json(art("p1/data_audit.json"))
    assert audit["valid"], "P1 data audit invalid"
    panel = read_json(art("p1/panel.json"))["rows"]
    firewall(FORBIDDEN)
    b, env = load_model()
    w0 = weights_hash(b.model)
    S = special_ids(b)
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    # --- 1. F1 Auto beam-5 decode of the training panel (labels unopened)
    dec, audio_ok = [], 0
    for r in panel:
        audio_ok += audio_hash_ok(r["audio_path"], r["audio_sha256"])
        rec, inp = generate(b, r["audio_path"], None, C["decode"]["num_beams"])
        rec.update(id=r["utterance_id"], language_top3=language_scores(b, inp["input_features"]))
        dec.append(rec)
    write_jsonl(art("p1/decode.jsonl"), dec)
    t_decode = time.time() - t_start
    # --- 2. training labels; float32 teacher-forced capture
    firewall_off()
    tg = read_json(art("p1/targets.json"))["targets"]
    to_float32(b)
    masks = suppress_masks(b, b.device)
    tb = tok_bytes_fn(b.processor.tokenizer)
    latin, han = vocab_sets(tb, b.model.config.vocab_size)
    latin_t, han_t = torch.tensor(latin, device=b.device), torch.tensor(han, device=b.device)
    FILL = fillers()
    meta, Q, H, GS, utt = [], [], [], [], []
    for r, d in zip(panel, dec):
        uid = r["utterance_id"]
        assert d["id"] == uid
        tgt = tg[uid]
        st = utterance_stratum(tgt["reference_raw"], FILL)
        enc = encode(b, r["audio_path"])
        am, ru, grads, _ = utterance_pass(b, enc, d["prompt"], tgt, masks, latin_t, han_t, True)
        fd, rows = build_rows(uid, tgt, d["tokens"], am, ru, tb, S["eos"], FILL, P1["source_b_max_per_utterance"],
                              P1["preservation_max_per_utterance"])
        utt.append({"id": uid, "dialogue": r["dialogue_id"], "partition": r["partition"], "stratum": st["stratum"],
                    "language": d["language"], "fd_class": fd["cls"] + ("/" + fd["sub"] if fd.get("sub") else ""),
                    "fd_k": fd.get("k"), "n_target_tokens": len(tgt["target_ids"]), "terminated": d["terminated"],
                    "rows": dict(collections.Counter(x["source"] for x in rows))})
        for row in rows:
            m0, q, h, gs = grads(row)
            pos = P0 - 1 + row["t"]
            meta.append({**row, "id": uid, "dialogue": r["dialogue_id"], "partition": r["partition"],
                         "stratum": st["stratum"], "pos": pos, "m0": m0, "h_norm": float(h.norm()), "q_norm": float(q.norm()),
                         "gs_norm": float(gs.norm()), "cos_q_gs": float(q @ gs / (q.norm() * gs.norm() + 1e-30)),
                         "correct_is_argmax": am[row["t"]] == row["correct"], "competitor_is_argmax": am[row["t"]] == row["competitor"],
                         "correct_text": tb(row["correct"]).decode("utf-8", "replace"),
                         "competitor_text": tb(row["competitor"]).decode("utf-8", "replace"), "language": d["language"]})
            Q.append(q.cpu())
            H.append(h.cpu())
            GS.append(gs.cpu())
    w1 = weights_hash(b.model)
    Qt, Ht, Gt = torch.stack(Q), torch.stack(H), torch.stack(GS)
    for i, m in enumerate(meta):
        m["row"] = i
    write_jsonl(art("p1/gradients/rows.jsonl"), meta)
    write_jsonl(art("p1/gradients/utterances.jsonl"), utt)
    tp = art("p1/gradients/tensors.pt")
    assert not tp.exists()
    torch.save({"q": Qt, "h": Ht, "gs": Gt, "latin_ids": latin, "han_ids": han}, tp)

    def cov(rs):
        return {"rows": len(rs), "utterances": len({x["id"] for x in rs}), "dialogues": len({x["dialogue"] for x in rs}),
                "by_stratum": dict(collections.Counter(x["stratum"] for x in rs)),
                "by_category": dict(collections.Counter(x["category"] for x in rs)),
                "by_lang": dict(collections.Counter(x["lang"] for x in rs)),
                "m0_negative": sum(x["m0"] < 0 for x in rs),
                "competitor_is_argmax": sum(x["competitor_is_argmax"] for x in rs),
                "q_norm_median": float(torch.tensor([x["q_norm"] for x in rs]).median()) if rs else None,
                "h_norm_median": float(torch.tensor([x["h_norm"] for x in rs]).median()) if rs else None}
    coverage = {f"{p}/{s}": cov([x for x in meta if x["partition"] == p and x["source"] == s])
                for p in ("construction", "audit") for s in ("A", "B", "P")}
    fdc = {p: dict(collections.Counter(u["fd_class"] for u in utt if u["partition"] == p)) for p in ("construction", "audit")}
    gm = {"schema": "reft_p1_gradient_manifest_v1", "utc": now(), "n_rows": len(meta), "coverage": coverage,
          "first_divergence_classes": fdc, "languages": dict(collections.Counter(d["language"] for d in dec)),
          "tensors_sha256": sha(tp), "rows_sha256": sha(art("p1/gradients/rows.jsonl")),
          "decode_sha256": sha(art("p1/decode.jsonl")), "latin_ids": len(latin), "han_ids": len(han),
          "orientation_note": "q stored unnormalized; norms (q_norm, h_norm, gs_norm) and orientation (cos_q_gs) recorded separately"}
    write_json(art("p1/gradient_manifest.json"), gm)
    man = {"schema": "reft_p1_capture_v1", "utc": now(), **fz, "git": git_state(), "env": env, "slurm": slurm(),
           "smoke": sm.name, "weights_before": w0, "weights_after_bf16_cast": w1, "weights_unchanged": w0 == w1,
           "weights_equal_expected": w0 == C["model"]["expected_weights_hash"], "audio_sha_ok": audio_ok, "n_panel": len(panel),
           "decode_seconds": t_decode, "capture_seconds": time.time() - t_start - t_decode, "wall_seconds": time.time() - t_start,
           "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()), "compute_dtype": "float32 (bf16 weights upcast exactly)"}
    write_json(art("p1/manifest_capture.json"), man)
    print(json.dumps({"manifest": {k: v for k, v in man.items() if k != "env"}, "coverage": coverage, "fd": fdc}, indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "capture"])
    a = ap.parse_args(argv)
    {"smoke": run_smoke, "capture": run_capture}[a.cmd]()


if __name__ == "__main__":
    main()
