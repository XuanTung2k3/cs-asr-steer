#!/usr/bin/env python
"""D2 representation edit-locality pilot on the unchanged R2-LoReFT (docs/reft/D2_EDIT_LOCALITY_PROTOCOL.md,
docs/reft/D2_EDIT_LOCALITY_DECISION_RULES.md, configs/reft/d2_edit_locality_frozen.yaml).

  mask                (CPU) training-only Mandarin-positive token metadata -> d2_edit_locality/mask_private.jsonl, mask_static.json
  freeze              (CPU) write-once d2_edit_locality/freeze.json + AUTHORIZATION.json; upstream chains and the mask hash
  freeze-amend --doc  (CPU) write-once amendment record chained to the current head (documented repair only)
  calibrate           (GPU) D2.0: one mu from training-only gradient scales at the historical GoldKL seed-42 epoch-1 state
                      -> calibration_private.json and, only if every stop check passes, write-once mu_freeze.json
  preflight           (GPU) gates G1-G7 -> preflight_<job>.json; D2_READY_FOR_TRAINING or D2_BLOCKED_<gates>
  train --seed        (GPU) one write-once run d2_edit_locality/training/D2_seed{s}
  verify              (GPU) every D2 epoch adapter reloads and re-decodes the first 20 dev utterances as recorded

The objective is P2-1b GoldKL (CE + 1.0 * KL on the teacher-correct mask, kl_masked unchanged) plus mu * L_edit, with
L_edit the mean of ||delta_h||^2 / 1280 over teacher-correct Mandarin-positive target positions. delta_h is the float32
output of the R2 intervention module, read by a forward hook on that module (EditObserver) during the single adapted
forward: no second forward, no extra adapter call, reft/adapters.py unchanged. With mu = 0 the D2 path is the P2-1b
GoldKL path (preflight G4: bitwise replay of the historical KL_seed42 epoch-1 adapter with the observer installed).

D2_DRY_ROOT=<scratch dir> redirects every output for pre-freeze engineering dry runs (no freeze check; train refuses).
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import art, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from reft.provenance import cublas_consistent, git_state, now, sha, slurm, weights_hash  # noqa: E402

CONF = ROOT / "configs/reft/d2_edit_locality_frozen.yaml"
FREEZE_SOURCES = ["docs/reft/D2_EDIT_LOCALITY_PROTOCOL.md", "docs/reft/D2_EDIT_LOCALITY_DECISION_RULES.md",
                  "configs/reft/d2_edit_locality_frozen.yaml", "reft/d2_edit_locality_train.py", "reft/d2_edit_locality_evaluate.py",
                  "tests/reft/test_d2_edit_locality.py", "slurm/reft_d2_edit_locality.sbatch", "slurm/reft_d2_edit_locality_cpu.sbatch",
                  "slurm/reft_d2_edit_locality_submit.sh"]
REUSED_SOURCES = ["reft/adapters.py", "reft/site.py", "reft/auto_policy.py", "reft/baseline_auto.py", "reft/train_supervised.py",
                  "reft/p2_1b_train.py", "reft/p2_1b_evaluate.py", "reft/evaluate_supervised.py", "reft/p2_1d_audit.py",
                  "reft/p2_2_evaluate.py", "reft/arch_s0s1_train.py", "reft/arch_s0s1_adapters.py", "reft/arch_s0s1_evaluate.py",
                  "reft/d1_regionkl_train.py", "reft/d1_regionkl_evaluate.py", "reft/text.py", "reft/common.py", "reft/provenance.py",
                  "src/csasr/data/language_tags.py", "src/csasr/data/normalize.py",
                  "configs/reft/p0p1.yaml", "configs/reft/p2_1_frozen.yaml", "configs/reft/p2_1b_frozen.yaml",
                  "configs/reft/d1_regionkl_frozen.yaml"]
SEEDS = (42, 43, 44)
START = 3
D_MODEL = 1280
DRY = os.environ.get("D2_DRY_ROOT")
MASK_FILES = ("mask_private.jsonl", "mask_static.json")
R2_TRAINABLE = ("rot.parametrizations.weight.original", "learned.weight", "learned.bias")


def cd() -> dict:
    return yaml.safe_load(CONF.read_text())


def out(*p) -> Path:
    return Path(DRY, *p) if DRY else art("d2_edit_locality", *p)


# ====================================================================== pure helpers (unit-tested on CPU)

def classify_zh_tokens(um: list[list[int]], labels: list) -> list[str]:
    """Per content token: 'zh' (every touched unit substantive Mandarin), 'en' (every touched unit substantive English),
    'unmapped' (touches no unit), 'ambiguous_language_boundary' (ZH and EN units), 'ambiguous_nonlexical' (ZH and a
    filler / OTHER / mixed-script unit, no EN) or 'other' (only non-lexical units, or EN with non-lexical units)."""
    cls = []
    for us in um:
        lab = {labels[k] for k in us}
        if not us:
            cls.append("unmapped")
        elif lab == {"ZH"}:
            cls.append("zh")
        elif lab == {"EN"}:
            cls.append("en")
        elif "ZH" in lab and "EN" in lab:
            cls.append("ambiguous_language_boundary")
        elif "ZH" in lab:
            cls.append("ambiguous_nonlexical")
        else:
            cls.append("other")
    return cls


def utterance_mask(t_text: str, ids: list[int], tb, FILL: set, K: int) -> dict:
    """Training-reference edit-mask record of one utterance. ids = target_ids (content tokens + final EOS).
    zh_tokens: Mandarin-positive content-token indices (edit-mask candidates before the teacher mask); en_tokens:
    substantive-English tokens (diagnostics); zh_sb_tokens: the subset inside D1 switch-back windows (diagnostics)."""
    from reft.d1_regionkl_train import classify_tokens, lang_labels, switchback_windows
    from reft.text import token_unit_map, units
    content = ids[:-1]
    U = units(t_text)
    labels = lang_labels(U, FILL)
    rec = {"n_target": len(ids), "n_units": len(U), "n_sub_en": labels.count("EN"), "n_sub_zh": labels.count("ZH")}
    um = token_unit_map(t_text, content, tb)
    bytes_ok = b"".join(tb(i) for i in content) == t_text.encode("utf-8")
    if um is None or not bytes_ok:
        return {**rec, "mappable": False, "zh_tokens": [], "en_tokens": [], "zh_sb_tokens": [], "ambiguous": [], "class_counts": {},
                "zh_units_without_token": labels.count("ZH")}
    cls = classify_zh_tokens(um, labels)
    zh = [t for t, c in enumerate(cls) if c == "zh"]
    wins = switchback_windows(labels, K)
    sb = {t for t, c in enumerate(classify_tokens(um, wins, labels)) if c == "region"}
    if not sb <= set(zh):
        raise AssertionError("switch-back region token outside the Mandarin-positive set")
    covered = {k for t in zh for k in um[t]}
    return {**rec, "mappable": True, "zh_tokens": zh, "en_tokens": [t for t, c in enumerate(cls) if c == "en"],
            "zh_sb_tokens": sorted(sb), "ambiguous": [[t, c] for t, c in enumerate(cls) if c.startswith("ambiguous")],
            "class_counts": dict(collections.Counter(cls)), "n_switchbacks": len(wins),
            "zh_units_without_token": sum(1 for k, lab in enumerate(labels) if lab == "ZH" and k not in covered)}


def token_mask_tensor(Y, row_ids: list[list[int]], row_tokens: list[list[int]], start: int = START):
    """Bool tensor shaped like Y: True at decoder position start + t for every listed content token t (the position whose
    label is target_ids[t]). Asserts the decoder shift and that EOS / unlabelled positions are never included."""
    import torch
    Z = torch.zeros(Y.shape, dtype=torch.bool)
    for i, (ids, toks) in enumerate(zip(row_ids, row_tokens)):
        for t in toks:
            if not (0 <= t < len(ids) - 1) or int(Y[i, start + t]) != ids[t]:
                raise ValueError(f"mask token {t} does not label decoder position {start + t}")
            Z[i, start + t] = True
    return Z


def edit_energy(out_, pos, start: int = START):
    """Per-position squared norm of the applied edit: out_ is the intervention-module output (B, T, d) float32 and the
    adapter applies delta = out_ * (pos >= start) (reft.adapters.LoReFTAdapter._attn); the same expression is used here."""
    import torch
    mask = (pos >= start).to(torch.float32)[None, :, None]
    return (out_ * mask).pow(2).sum(-1)


def edit_loss(E, Mz, d: int = D_MODEL):
    """L_edit = (1 / |R_M|) sum_{t in R_M} E_t / d, with E the per-position squared edit norm (differentiable) and Mz the
    boolean edit mask (teacher-correct AND Mandarin-positive). Empty mask -> differentiable zero. Returns (loss, count)."""
    n = int(Mz.sum())
    if n == 0:
        return E.sum() * 0.0, 0
    return (E[Mz] / d).sum() / n, n


class EditObserver:
    """Forward hook on the R2 intervention module (``ad.module``). The adapter's encoder_attn hook calls that module once
    per decoder-layer call and adds ``module_output * position_mask`` to the cross-attention output; the observer reads the
    same output tensor (differentiable, float32) and records the per-position squared edit norm. It never modifies the
    output. Records are cleared on attach and detach; ``single()`` requires exactly one record (one adapted call)."""

    def __init__(self, ad, keep_input: bool = False):
        self.ad, self.keep_input = ad, keep_input
        self.records: list[dict] = []
        self.handle = None

    def _hook(self, mod, args, out_):
        cp = self.ad.cache_pos
        T = out_.shape[1]
        pos = cp[-T:] if cp.numel() >= T else cp
        rec = {"energy": edit_energy(out_, pos, self.ad.start_pos), "pos": pos.detach().clone()}
        if self.keep_input:
            rec["h"] = args[0].detach()
            rec["out"] = out_.detach()
        self.records.append(rec)

    @contextlib.contextmanager
    def attached(self):
        if self.handle is not None:
            raise RuntimeError("observer already attached")
        self.records = []
        self.handle = self.ad.module.register_forward_hook(self._hook)
        try:
            yield self
        finally:
            self.handle.remove()
            self.handle = None
            self.records = []

    def single(self) -> dict:
        if len(self.records) != 1:
            raise RuntimeError(f"expected exactly one intervention call, saw {len(self.records)}")
        return self.records.pop()


def calibration_rows(uids: list[str], n_batches: int, mb: int = 8) -> list[list[str]]:
    """Frozen calibration micro-batches: utterance ids sorted by sha256 hex digest, the first n_batches * mb in order."""
    s = sorted(uids, key=lambda u: hashlib.sha256(u.encode()).hexdigest())[:n_batches * mb]
    if len(s) < n_batches * mb:
        raise ValueError("not enough training rows for calibration")
    return [s[j:j + mb] for j in range(0, len(s), mb)]


def mu_rule(gb: list[float], ge: list[float], rho: float) -> float:
    """mu = rho * median ||g_base|| / median ||g_edit||."""
    return float(rho * np.median(gb) / np.median(ge))


def calibration_checks(gb: list[float], ge: list[float], counts: list[int], mu: float, mu2: float, S: dict) -> dict:
    """Frozen D2.0 stop checks. gb, ge: per-batch raw gradient norms; counts: edit-mask positions per batch; mu2: the
    second (repeat) pass."""
    fin = all(np.isfinite(x) for x in list(gb) + list(ge)) and np.isfinite(mu) and np.isfinite(mu2)
    nonzero = fin and all(x > S["min_edit_grad_norm"] for x in ge) and all(x > 0 for x in gb)
    r = [b / e for b, e in zip(gb, ge)] if nonzero else []
    scaled = [mu * e / b for b, e in zip(gb, ge)] if nonzero else []
    c = {"finite": bool(fin), "edit_grad_nonzero": bool(nonzero),
         "mask_positions_every_batch": all(n >= S["min_mask_positions_per_batch"] for n in counts),
         "reproducible": bool(fin and mu > 0 and abs(mu2 - mu) <= S["reproducibility_rtol"] * mu),
         "ratio_spread": float(max(r) / min(r)) if r else None,
         "ratio_spread_ok": bool(r and max(r) / min(r) <= S["max_ratio_spread"]),
         "scaled_ratios": [float(x) for x in scaled], "max_scaled_ratio": float(max(scaled)) if scaled else None,
         "no_pathological_dominance": bool(scaled and max(scaled) <= S["max_scaled_ratio"])}
    keys = ("finite", "edit_grad_nonzero", "mask_positions_every_batch", "reproducible", "ratio_spread_ok", "no_pathological_dominance")
    c["ok"] = all(c[k] for k in keys)
    c["failed"] = [k for k in keys if not c[k]]
    return c


def training_order_counts(tm: list[dict], per_utt: dict, mb: int = 8) -> dict:
    """Edit-mask positions per micro-batch in each of the nine seed x epoch training orders (P2-1b order
    default_rng(seed + epoch).permutation over the manifest), from per-utterance counts."""
    res = {}
    for s in SEEDS:
        for ep in (1, 2, 3):
            order = np.random.default_rng(s + ep).permutation(len(tm))
            cnt = [sum(per_utt.get(tm[i]["utterance_id"], 0) for i in order[j:j + mb]) for j in range(0, len(order), mb)]
            res[f"{s}_{ep}"] = {"min": int(min(cnt)), "median": float(np.median(cnt)), "max": int(max(cnt)),
                                "zero_batches": int(sum(c == 0 for c in cnt)), "n_batches": len(cnt)}
    return res


def flat_norm(gs) -> float:
    import torch
    return float(torch.cat([g.reshape(-1).double() for g in gs]).norm())


def flat_cos(ga, gb) -> float:
    import torch
    a = torch.cat([g.reshape(-1).double() for g in ga])
    b = torch.cat([g.reshape(-1).double() for g in gb])
    return float((a @ b) / (a.norm() * b.norm() + 1e-300))


# ====================================================================== mask metadata (CPU, training references only)

def load_mask() -> dict:
    return {r["utterance_id"]: r for r in read_jsonl(out("mask_private.jsonl"))}


def run_mask():
    from transformers import WhisperTokenizer
    from reft.common import cfg, fillers
    from reft.text import tok_bytes_fn
    from reft.train_supervised import c21
    for f in MASK_FILES:
        if out(f).exists():
            raise SystemExit(f"{f} exists; write-once")
    C, C1 = cd(), c21()
    assert sha(art("p2_0/train_manifest.json")) == C1["data"]["train_manifest_sha256"]
    t0 = time.time()
    tm = read_json(art("p2_0/train_manifest.json"))["rows"]
    tg = read_json(art("p1/targets.json"))["targets"]
    tok = WhisperTokenizer.from_pretrained(cfg()["model"]["path"], local_files_only=True)
    tb, eos = tok_bytes_fn(tok), tok.eos_token_id
    FILL = fillers()
    K = C["evaluation"]["K"]
    recs, mismatch = [], []
    for r in tm:
        uid = r["utterance_id"]
        ids = r["target_ids"]
        if ids != tg[uid]["target_ids"] or ids[-1] != eos or eos in ids[:-1]:
            mismatch.append(uid)
        recs.append({"utterance_id": uid, "dialogue_id": r["dialogue_id"], **utterance_mask(tg[uid]["target_text"], ids, tb, FILL, K)})
    t_map = time.time() - t0
    write_jsonl(out("mask_private.jsonl"), recs)
    S = static_summary(recs, tm)
    S.update({"targets_mismatch": len(mismatch), "mapping_seconds": t_map, "K": K,
              "mask_sha256": sha(out("mask_private.jsonl")), "train_manifest_sha256": sha(art("p2_0/train_manifest.json")),
              "targets_sha256": sha(art("p1/targets.json")), "utc": now(), "git": git_state()})
    write_json(out("mask_static.json"), S)
    print(json.dumps({k: v for k, v in S.items() if k not in ("per_batch", "git")}, indent=1))


def static_summary(recs: list[dict], tm: list[dict]) -> dict:
    """Training-only counts that do not need the teacher (aggregate, no identifiers)."""
    mp = [r for r in recs if r["mappable"]]
    cls = collections.Counter()
    for r in mp:
        cls.update(r["class_counts"])
    amb = collections.Counter(c for r in mp for _, c in r["ambiguous"])
    zh = {r["utterance_id"]: r["zh_tokens"] for r in recs}
    per_batch = {}
    for s in SEEDS:
        for ep in (1, 2, 3):
            order = np.random.default_rng(s + ep).permutation(len(tm))
            ids_ = [tm[i]["utterance_id"] for i in order]
            cnt = [sum(len(zh[u]) for u in ids_[j:j + 8]) for j in range(0, len(ids_), 8)]
            per_batch[f"{s}_{ep}"] = {"min": int(min(cnt)), "median": float(np.median(cnt)), "max": int(max(cnt)),
                                      "zero_batches": int(sum(c == 0 for c in cnt)), "n_batches": len(cnt)}
    return {"utterances": len(recs), "mappable_utterances": len(mp), "unmappable_utterances": len(recs) - len(mp),
            "unmappable_fraction": (len(recs) - len(mp)) / len(recs),
            "supervised_positions": sum(r["n_target"] for r in recs), "eos_positions": len(recs),
            "units": sum(r["n_units"] for r in recs), "substantive_en_units": sum(r["n_sub_en"] for r in recs),
            "substantive_zh_units": sum(r["n_sub_zh"] for r in recs),
            "zh_tokens": sum(len(r["zh_tokens"]) for r in recs), "zh_sb_tokens": sum(len(r["zh_sb_tokens"]) for r in recs),
            "en_tokens": sum(len(r["en_tokens"]) for r in recs),
            "ambiguous_tokens": sum(amb.values()), "ambiguous_by_reason": dict(amb), "token_classes": dict(cls),
            "zh_units_without_token": sum(r["zh_units_without_token"] for r in recs),
            "utterances_with_zh": sum(1 for r in recs if r["zh_tokens"]),
            "dialogues_with_zh": len({r["dialogue_id"] for r in recs if r["zh_tokens"]}),
            "dialogues_total": len({r["dialogue_id"] for r in recs}), "eos_in_mask": 0, "per_batch": per_batch}


# ====================================================================== freeze chain

def control_runs() -> dict:
    res = {}
    for s, run in cd()["control"]["runs"].items():
        rd = art(run)
        man = read_json(rd / "manifest.json")
        res[int(s)] = {"run": run, "dir": rd, "selected_epoch": read_json(rd / "selection.json")["selected"]["epoch"],
                       "adapter_sha": {h["epoch"]: h["adapter_sha256"] for h in man["history"][1:]}, "history": man["history"]}
    return res


def frozen_inputs() -> list[str]:
    L = ["p2_0/train_manifest.json", "p2_0/dev_manifest.json", "p2_0/data_audit.json", "p2_0/frozen_baseline_dev150_beam5.jsonl",
         "p2_0/frozen_baseline_dev150_greedy.jsonl", "p1/targets.json", "p2_1b/evaluation.json", "p2_1b/report.json", "p2_1b/verify.json",
         "p2_2/report.json", "arch_s0s1/report.json", "d1_regionkl/report.json", "d1_regionkl/evaluation.json", "d1_regionkl/freeze.json"]
    for c in control_runs().values():
        L += [f"{c['run']}/selection.json", f"{c['run']}/manifest.json"]
        L += [f"{c['run']}/epoch{e}/adapter.pt" for e in (1, 2, 3)]
        L += [f"{c['run']}/epoch{e}/dev_{r}.jsonl" for e in (0, 1, 2, 3) for r in ("beam5", "greedy")]
    return L


def upstream_chains() -> dict:
    """D1 freeze chain, whose checker verifies S0/S1, which verifies P2-0, P2-1b, P2-1c and P2-2."""
    from reft.d1_regionkl_train import check_freeze as d1
    return {"d1_regionkl": d1()}


def freeze_path(rev: int) -> Path:
    return out("freeze.json") if rev == 0 else out(f"freeze_r{rev}.json")


def chain_head() -> int:
    rev = 0
    while freeze_path(rev + 1).exists():
        rev += 1
        if read_json(freeze_path(rev))["prev_freeze_sha256"] != sha(freeze_path(rev - 1)):
            raise SystemExit(f"D2 freeze chain broken at {freeze_path(rev).name}")
    return rev


def run_freeze():
    if DRY:
        raise SystemExit("no freeze in dry-run mode")
    p = out("freeze.json")
    if p.exists():
        raise SystemExit(f"{p} exists; write-once")
    g = git_state()
    if g["branch"] != "cs-asr-reft" or g["dirty_paths"]:
        raise SystemExit(f"branch/working tree not clean: {g}")
    if not all(out(f).exists() for f in MASK_FILES):
        raise SystemExit("mask metadata missing (run `mask` first)")
    st = read_json(out("mask_static.json"))
    if st["mask_sha256"] != sha(out("mask_private.jsonl")):
        raise SystemExit("mask metadata hash mismatch")
    ctl = {s: {e: sha(c["dir"] / f"epoch{e}" / "adapter.pt") == h for e, h in c["adapter_sha"].items()} for s, c in control_runs().items()}
    if not all(v for x in ctl.values() for v in x.values()):
        raise SystemExit(f"R2-GoldKL control adapter hash mismatch: {ctl}")
    rec = {"schema": "reft_d2_edit_locality_freeze_v1", "utc": now(), "git": g, "sources": {s: sha(ROOT / s) for s in FREEZE_SOURCES},
           "reused_sources": {s: sha(ROOT / s) for s in REUSED_SOURCES}, "inputs": {i: sha(art(i)) for i in frozen_inputs()},
           "mask_metadata": {f: sha(out(f)) for f in MASK_FILES}, "controls_match_recorded_sha": {str(k): v for k, v in ctl.items()},
           "calibration": cd()["calibration"], "upstream": upstream_chains()}
    write_json(p, rec)
    write_json(out("AUTHORIZATION.json"), {"utc": now(), "stage": "D2 representation edit-locality pilot (R2-LoReFT + GoldKL + mu L_edit)",
                                           "authorized_by": "project owner (2026-10-15 prompt)", "runs": cd()["new_runs"],
                                           "max_new_training_runs": cd()["max_new_training_runs"],
                                           "conditions": "one mu from the frozen training-only calibration rule; formal training only if the "
                                                         "preflight is D2_READY_FOR_TRAINING; no D1 reweighting, mu sweeps, D1+D2, R3/M1, "
                                                         "layer/rank changes, prefix-aware objectives, LoRA, locked roles or independent test "
                                                         "evaluation; historical controls are not retrained"})
    print(p)


def run_freeze_amend(doc: str):
    if DRY:
        raise SystemExit("no freeze in dry-run mode")
    k = chain_head() + 1
    prev = freeze_path(k - 1)
    rec0 = read_json(prev)
    if not (ROOT / doc).exists():
        raise SystemExit(f"{doc} missing")
    inputs = {i: sha(art(i)) for i in rec0["inputs"]}
    if inputs != rec0["inputs"] or {f: sha(out(f)) for f in MASK_FILES} != rec0["mask_metadata"]:
        raise SystemExit(f"frozen inputs or mask metadata changed since {prev.name}")
    srcs = list(rec0["sources"]) + ([doc] if doc not in rec0["sources"] else [])
    rec = {"schema": "reft_d2_edit_locality_freeze_amendment_v1", "amendment": f"r{k}", "utc": now(), "git": git_state(),
           "prev_freeze": prev.name, "prev_freeze_sha256": sha(prev), "amendment_doc": doc,
           "changed_sources": sorted(x for x in srcs if rec0["sources"].get(x) != sha(ROOT / x)),
           "sources": {x: sha(ROOT / x) for x in srcs}, "reused_sources": {x: sha(ROOT / x) for x in rec0["reused_sources"]},
           "inputs": inputs, "mask_metadata": rec0["mask_metadata"], "controls_match_recorded_sha": rec0["controls_match_recorded_sha"],
           "calibration": rec0["calibration"], "upstream": upstream_chains(),
           "outputs_at_amendment": {q.name: sha(q) for q in sorted(out().glob("*.json"))}}
    print(write_json(freeze_path(k), rec))


def frozen_mu(required: bool = True) -> dict | None:
    """The write-once D2.0 coefficient record, verified against its calibration file and freeze record."""
    p = out("mu_freeze.json")
    if not p.exists():
        if required:
            raise SystemExit("mu is not frozen (run `calibrate`)")
        return None
    m = read_json(p)
    if m["calibration_sha256"] != sha(out("calibration_private.json")):
        raise SystemExit("calibration record changed after mu was frozen")
    if not DRY and sha(out(m["freeze_record"])) != m["freeze_sha256"]:
        raise SystemExit("freeze record under mu_freeze.json changed")
    if m["verdict"] != "D2_CALIBRATION_OK":
        raise SystemExit(f"calibration not OK: {m['verdict']}")
    return {"mu": m["mu"], "mu_freeze_sha256": sha(p), "calibration_sha256": m["calibration_sha256"]}


def check_freeze(require_mu: bool = True) -> dict:
    if DRY:
        mu = frozen_mu(required=False)
        return {"dry_run": True, **(mu or {})}
    head = chain_head()
    p = freeze_path(head)
    rec = read_json(p)
    bad = [s for s, h in {**rec["sources"], **rec["reused_sources"]}.items() if sha(ROOT / s) != h]
    bad += [i for i, h in rec["inputs"].items() if sha(art(i)) != h]
    bad += [f for f, h in rec["mask_metadata"].items() if sha(out(f)) != h]
    if bad:
        raise SystemExit(f"D2 frozen sources/inputs/mask metadata changed: {bad}")
    if upstream_chains() != rec["upstream"]:
        raise SystemExit("an upstream freeze chain changed (P2-0/P2-1b/P2-1c/P2-2/S0S1/D1)")
    res = {"d2_freeze_sha256": sha(p), "d2_freeze_record": p.name, "d2_freeze_chain_len": head + 1,
           "mask_sha256": rec["mask_metadata"]["mask_private.jsonl"], "d1_freeze_sha256": rec["upstream"]["d1_regionkl"]["d1_freeze_sha256"]}
    mu = frozen_mu(required=require_mu)
    return {**res, **(mu or {})}


# ====================================================================== training step / epoch (GPU)

def d2_step(b, ad, obs, rows, lam, mu, mask: dict, grad=True):
    """reft.p2_1b_train.kl_step (same CE, teacher, kl_masked) with the edit observer attached and + mu * L_edit."""
    import torch
    from reft.auto_policy import build_tf_batch, tf_loss
    from reft.p2_1b_train import kl_masked, teacher_mask
    from reft.train_supervised import encode, tf_forward
    eos = b.processor.tokenizer.eos_token_id
    ids = [r["target_ids"] for r in rows]
    X, Y = build_tf_batch([r["prompt"] for r in rows], ids, eos, eos)
    Z = token_mask_tensor(Y, ids, [mask[r["utterance_id"]]["zh_tokens"] if r["utterance_id"] in mask else [] for r in rows])
    ZE = token_mask_tensor(Y, ids, [mask[r["utterance_id"]]["en_tokens"] if r["utterance_id"] in mask else [] for r in rows])
    _, enc = encode(b, [r["audio_path"] for r in rows])
    X, Y = X.to(b.device), Y.to(b.device)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        zt = b.model(encoder_outputs=(enc,), decoder_input_ids=X, use_cache=False).logits
    torch.cuda.synchronize()
    t_teacher = time.time() - t0
    with obs.attached():
        c0 = ad.calls
        zs = tf_forward(b, ad, enc, X, grad=grad)
        rec = obs.single()
        calls = ad.calls - c0
    if calls != 1 or not torch.equal(rec["pos"], torch.arange(X.shape[1], device=rec["pos"].device)):
        raise RuntimeError(f"edit observer: {calls} adapter calls or non-absolute positions")
    ce, n = tf_loss(zs, Y)
    kl, m = kl_masked(zt, zs, Y)
    M = teacher_mask(zt, Y)
    Mz = M & Z.to(M.device)
    E = rec["energy"]
    le, nz = edit_loss(E, Mz)
    loss = ce + lam * kl + mu * le
    with torch.no_grad():
        Ed = E.detach() / D_MODEL
        ZEd, Zd = ZE.to(E.device), Z.to(E.device)
        lab = Y != -100

        def mean_(sel):
            k = int(sel.sum())
            return float(Ed[sel].mean()) if k else None
        st = {"ce": float(ce), "kl": float(kl), "edit": float(le), "mu_edit": float(mu * le), "total": float(loss), "n_tok": n, "n_mask": m,
              "n_edit": nz, "n_zh_tokens": int(Z.sum()), "n_en_tokens": int(ZE.sum()), "teacher_s": t_teacher,
              "e_zh_tc": mean_(Mz), "e_zh_all": mean_(Zd), "e_en_tc": mean_(M & ZEd), "e_en_all": mean_(ZEd), "e_labelled": mean_(lab),
              "n_en_tc": int((M & ZEd).sum())}
    return loss, st, (zt, zs, X, Y, Z, Mz, E)


def d2_epoch(b, ad, obs, opt, tm, tr, seed, ep, lam, mu, mask, steps0, log):
    """One epoch of reft.p2_1b_train.train_loop (arm KL) step for step with d2_step, plus read-only diagnostics in `log`."""
    import torch
    from reft.arch_s0s1_adapters import named_trainable, orth_error
    mb, acc = tr["micro_batch"], tr["grad_accum"]
    steps = steps0
    order = np.random.default_rng(seed + ep).permutation(len(tm))
    batches = [[tm[i] for i in order[j:j + mb]] for j in range(0, len(order), mb)]
    torch.cuda.synchronize()
    t0 = time.time()
    sums = collections.Counter()
    pending, gn = 0, []
    params = named_trainable(ad)
    for k, rows in enumerate(batches):
        loss, st, _ = d2_step(b, ad, obs, rows, lam, mu, mask)
        sums["kl"] += st["kl"] * st["n_mask"]
        sums["edit"] += st["edit"] * st["n_edit"]
        for key in ("n_mask", "n_edit", "n_zh_tokens", "teacher_s"):
            sums[key] += st[key]
        if st["e_en_tc"] is not None:
            sums["e_en_tc"] += st["e_en_tc"] * st["n_en_tc"]
            sums["n_en_tc"] += st["n_en_tc"]
        sums["empty_mask_batches"] += st["n_mask"] == 0
        sums["empty_edit_batches"] += st["n_edit"] == 0
        (loss / acc).backward()
        sums["ce"] += st["ce"] * st["n_tok"]
        sums["tot"] += float(loss) * st["n_tok"]
        sums["n_tok"] += st["n_tok"]
        log["micro"].append({"epoch": ep, "k": k, **{x: st[x] for x in ("ce", "kl", "edit", "mu_edit", "total", "n_tok", "n_mask", "n_edit",
                                                                         "n_zh_tokens", "n_en_tokens", "n_en_tc", "e_zh_tc", "e_zh_all",
                                                                         "e_en_tc", "e_en_all", "e_labelled")}})
        pending += 1
        if pending == acc or k == len(batches) - 1:
            with torch.no_grad():
                gpre = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in params.items()}
                before = {n: p.detach().clone() for n, p in params.items()}
            gn.append(float(torch.nn.utils.clip_grad_norm_(ad.parameters(), tr["clip_norm"])))
            opt.step()
            opt.zero_grad(set_to_none=True)
            steps += 1
            pending = 0
            with torch.no_grad():
                log["steps"].append({"epoch": ep, "step": steps, "grad_norm_total_preclip": gn[-1], "grad_norm_preclip": gpre,
                                     "lr": opt.param_groups[0]["lr"],
                                     "update_norm": {n: float((p.detach() - before[n]).norm()) for n, p in params.items()},
                                     "param_norm": {n: float(p.detach().norm()) for n, p in params.items()}, "orth_err": orth_error(ad)})
    torch.cuda.synchronize()
    return {"epoch": ep, "train_ce": sums["ce"] / sums["n_tok"], "train_total_tokmean": sums["tot"] / sums["n_tok"],
            "train_kl": sums["kl"] / sums["n_mask"] if sums["n_mask"] else None,
            "train_edit": sums["edit"] / sums["n_edit"] if sums["n_edit"] else None,
            "train_mu_edit": mu * sums["edit"] / sums["n_edit"] if sums["n_edit"] else None,
            "train_e_en_tc": sums["e_en_tc"] / sums["n_en_tc"] if sums["n_en_tc"] else None,
            "mask_positions": sums["n_mask"], "edit_mask_positions": sums["n_edit"], "zh_tokens": sums["n_zh_tokens"],
            "supervised_tokens": sums["n_tok"], "mask_fraction": sums["n_mask"] / sums["n_tok"],
            "empty_mask_batches": sums["empty_mask_batches"], "empty_edit_batches": sums["empty_edit_batches"], "updates_total": steps,
            "grad_norm_mean": float(np.mean(gn)), "grad_norm_max": float(np.max(gn)), "train_seconds": time.time() - t0,
            "teacher_forward_seconds": sums["teacher_s"]}


def make_r2(b, seed):
    import torch
    from reft.arch_s0s1_adapters import make_arch_adapter
    torch.manual_seed(seed)
    return make_arch_adapter("R2", b.model, seed, b.device)


def load_control_epoch1(b, ctl):
    """The historical R2-GoldKL seed-42 epoch-1 adapter after its sha256 is checked against the run manifest."""
    import torch
    p = ctl[42]["dir"] / "epoch1" / "adapter.pt"
    if sha(p) != ctl[42]["adapter_sha"][1]:
        raise SystemExit("historical KL_seed42 epoch-1 adapter hash mismatch")
    a = make_r2(b, 42)
    a.load_state_dict(torch.load(p))
    return a


def grad_pieces(b, ad, rows, lam, mask):
    """Raw (pre-clipping, unscaled) gradients of CE + lam * GoldKL and of L_edit for one micro-batch, one forward."""
    import torch
    from reft.arch_s0s1_adapters import named_trainable
    params = named_trainable(ad)
    assert tuple(params) == R2_TRAINABLE, tuple(params)
    obs = EditObserver(ad)
    loss0, st, (zt, zs, X, Y, Z, Mz, E) = d2_step(b, ad, obs, rows, lam, 0.0, mask)
    base = loss0                     # mu = 0: CE + lam * GoldKL (+ 0)
    le, nz = edit_loss(E, Mz)
    gb = torch.autograd.grad(base, list(params.values()), retain_graph=True)
    ge = torch.autograd.grad(le, list(params.values()))
    return gb, ge, {"base_loss": float(base), "edit": float(le), "n_edit": nz, "n_mask": st["n_mask"], "ce": st["ce"], "kl": st["kl"],
                    "per_tensor_base": {n: float(g.norm()) for n, g in zip(params, gb)},
                    "per_tensor_edit": {n: float(g.norm()) for n, g in zip(params, ge)}}


# ====================================================================== D2.0 calibration (GPU)

def run_calibrate():
    import torch
    from reft.arch_s0s1_train import Runtime
    fz = check_freeze(require_mu=False)
    if out("mu_freeze.json").exists() or out("calibration_private.json").exists():
        raise SystemExit("calibration outputs exist; write-once")
    if not DRY and not out("AUTHORIZATION.json").exists():
        raise SystemExit("D2 not authorized")
    C = cd()
    K, lam = C["calibration"], C["objective"]["lambda_kl"]
    mask = load_mask()
    t_all = time.time()
    rt = Runtime()
    b = rt.b
    ctl = control_runs()
    byid = {r["utterance_id"]: r for r in rt.tm}
    cal = calibration_rows([r["utterance_id"] for r in rt.tm], K["n_batches"])
    batches = [[byid[u] for u in bb] for bb in cal]
    # identity: L_edit and its gradient are exactly zero (calibration never uses this state)
    z = make_r2(b, 42)
    gbz, gez, iz = grad_pieces(b, z, batches[0], lam, mask)
    identity = {"edit": iz["edit"], "edit_exact_zero": iz["edit"] == 0.0, "edit_grads_exact_zero": all(bool((g == 0).all()) for g in gez),
                "base_grad_norm": flat_norm(gbz)}
    del z
    src = load_control_epoch1(b, ctl)
    passes = []
    for p_ in range(K["passes"]):
        rows_ = []
        for j, rows in enumerate(batches):
            gb, ge, info = grad_pieces(b, src, rows, lam, mask)
            rows_.append({"batch": j, "g_base": flat_norm(gb), "g_edit": flat_norm(ge), "cos_base_edit": flat_cos(gb, ge), **info})
        passes.append(rows_)
    gb1, ge1 = [r["g_base"] for r in passes[0]], [r["g_edit"] for r in passes[0]]
    mu = mu_rule(gb1, ge1, K["rho"])
    mu2 = mu_rule([r["g_base"] for r in passes[-1]], [r["g_edit"] for r in passes[-1]], K["rho"])
    chk = calibration_checks(gb1, ge1, [r["n_edit"] for r in passes[0]], mu, mu2, K["stop"])
    ok = chk["ok"] and identity["edit_exact_zero"] and identity["edit_grads_exact_zero"]
    verdict = "D2_CALIBRATION_OK" if ok else "D2_CALIBRATION_BLOCKED_" + "_".join(chk["failed"] + ([] if identity["edit_grads_exact_zero"] else ["identity"]))
    for r in passes[0]:
        r["mu_edit_over_base_loss"] = mu * r["edit"] / r["base_loss"]
    rec = {"schema": "reft_d2_calibration_v1", "utc": now(), "dry_run": bool(DRY), **fz, "git": git_state(), "slurm": slurm(),
           "rule": K["rule"], "rho": K["rho"], "source": K["source"], "source_sha256": ctl[42]["adapter_sha"][1],
           "batch_rule": K["batch_rule"], "batches_private": cal, "identity": identity, "passes": passes,
           "median_g_base": float(np.median(gb1)), "median_g_edit": float(np.median(ge1)), "mu": mu, "mu_repeat": mu2,
           "checks": chk, "verdict": verdict, "env": rt.env, "cuda_libs": rt.libs, "weights_unchanged": weights_hash(b.model) == rt.w0,
           "wall_seconds": time.time() - t_all}
    write_json(out("calibration_private.json"), rec)
    if ok:
        fr = "freeze.json" if DRY else freeze_path(chain_head()).name
        write_json(out("mu_freeze.json"), {"schema": "reft_d2_mu_freeze_v1", "utc": now(), "mu": mu, "rho": K["rho"], "verdict": verdict,
                                           "calibration_sha256": sha(out("calibration_private.json")), "freeze_record": fr,
                                           "freeze_sha256": None if DRY else sha(out(fr)), "git": git_state(), "slurm": slurm()})
    print(json.dumps({k: rec[k] for k in ("identity", "median_g_base", "median_g_edit", "mu", "mu_repeat", "checks", "verdict")}, indent=1))
    print(json.dumps([{k: r[k] for k in ("batch", "g_base", "g_edit", "cos_base_edit", "n_edit", "base_loss", "edit", "mu_edit_over_base_loss")}
                      for r in passes[0]], indent=1))


# ====================================================================== preflight (GPU)

def run_preflight():
    import torch
    import torch.nn.functional as F
    from reft.adapters import site_hook_count
    from reft.arch_s0s1_adapters import init_record, named_trainable, orth_error, perturb
    from reft.arch_s0s1_train import Runtime, cached_vs_full, greedy_vs_tf, make_opt, manual_delta, spy_positions
    from reft.auto_policy import build_tf_batch, frozen_lid
    from reft.common import fillers
    from reft.p2_1b_train import kl_step, seed_config, teacher_mask
    from reft.text import tok_bytes_fn
    from reft.train_supervised import batch_loss, encode
    from csasr.models.whisper import batch_model_inputs
    fz = check_freeze(require_mu=True)
    mu = fz["mu"]
    t_all = time.time()
    C = cd()
    P, T = C["preflight"], C["preflight"]["tolerances"]
    lam, K = C["objective"]["lambda_kl"], C["evaluation"]["K"]
    jid = str(slurm()["SLURM_JOB_ID"] or "local")
    pdir = out("preflight_files", jid)
    # training references for the G3 re-derivation are read before the GPU firewall is raised (training data only)
    tg = read_json(art("p1/targets.json"))["targets"]
    mrecs = read_jsonl(out("mask_private.jsonl"))
    static = read_json(out("mask_static.json"))
    mask = {r["utterance_id"]: r for r in mrecs}
    rt = Runtime()
    b = rt.b
    dev = b.device
    torch.cuda.reset_peak_memory_stats()
    res = {"freeze": fz}
    ctl = control_runs()
    pb = rt.C1["frozen_baseline"]
    sh = sorted(rt.tm, key=lambda r: r["utterance_id"])
    zrows = [r for r in sh if mask[r["utterance_id"]]["zh_tokens"]]

    # ---- G1 provenance and data firewall
    audit = read_json(art("p2_0/data_audit.json"))
    try:
        sq = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j %T"], capture_output=True, text=True, timeout=20).stdout
    except Exception as e:  # noqa: BLE001
        sq = f"unavailable: {e}"
    me = str(slurm()["SLURM_JOB_ID"])
    others = [x for x in sq.splitlines() if x.strip() and not x.startswith(me + " ") and ("reft" in x and "RUNNING" in x and "d2" not in x)]
    g = git_state()
    res["G1"] = {"branch": g["branch"] == "cs-asr-reft", "git_head": g["head"], "git_clean": not g["dirty_paths"],
                 "freeze_verified": DRY is None and "d2_freeze_sha256" in fz, "mu_frozen": "mu_freeze_sha256" in fz,
                 "controls_sha": all(sha(c["dir"] / f"epoch{e}" / "adapter.pt") == h for c in ctl.values() for e, h in c["adapter_sha"].items()),
                 "control_selection_and_evaluation_present": all((c["dir"] / "selection.json").exists() for c in ctl.values())
                 and art("p2_1b/evaluation.json").exists(),
                 "pinned_baseline": sha(art(pb["beam5"])) == pb["beam5_sha256"] and sha(art(pb["greedy"])) == pb["greedy_sha256"],
                 "manifests": sha(art("p2_0/train_manifest.json")) == rt.C1["data"]["train_manifest_sha256"]
                 and sha(art("p2_0/dev_manifest.json")) == rt.C1["data"]["dev_manifest_sha256"],
                 "weights_expected": rt.w0 == rt.expected_w, "canonical_cuda": cublas_consistent(rt.libs),
                 "data_audit_valid_no_locked": bool(audit["valid"] and audit["checks"]["no_locked_dialogues"]),
                 "no_conflicting_running_reft_jobs": not others, "no_training_outputs_yet": not out("training").exists()}
    res["G1_squeue"] = sq

    # ---- G2 site and edit correctness (perturbed adapter, training rows with Mandarin-positive tokens)
    g2 = {}
    tiny = zrows[:P["tf_rows"]]
    X, Y, enc = rt.tf_batch(tiny)
    q, u, zf = rt.site_record(enc, X)
    pa = make_r2(b, 42)
    perturb(pa, P["perturb"]["scale_w"], P["perturb"]["scale_b"], P["perturb"]["seed"])
    z_off = rt.logits(pa, enc, X)
    obs = EditObserver(pa, keep_input=True)
    with obs.attached():
        c0 = pa.calls
        z_on = rt.logits(pa, enc, X, grad=True)
        g2["observer_records"] = len(obs.records)
        g2["adapter_calls"] = pa.calls - c0
        rec = obs.single()
        g2["records_after_single"] = len(obs.records)
    g2["module_hooks_after_detach"] = len(pa.module._forward_hooks)
    g2["observer_logits_bitwise_unchanged"] = bool(torch.equal(z_on.detach(), z_off))
    md = manual_delta(pa, q, u)
    g2["single_application_bitwise"] = bool(torch.equal(z_on.detach(), rt.site_delta_logits(enc, X, md)))
    e_man = md.pow(2).sum(-1)
    E = rec["energy"]
    g2["observer_energy_bitwise_manual"] = bool(torch.equal(E, e_man))
    g2["observer_energy_rel_diff_manual"] = float((E - e_man).abs().max() / e_man.abs().max())
    with torch.no_grad():
        m_ = pa.module
        a = F.linear(rec["h"], m_.learned.weight - m_.rot.weight) + m_.learned.bias
        ea = a.pow(2).sum(-1)
        cm = (rec["pos"] >= START)
        g2["rank2_identity_rel_diff"] = float(((ea - E)[:, cm].abs() / ea[:, cm].clamp_min(1e-30)).max())
        g2["prompt_positions_zero_energy"] = bool((E[:, :START] == 0).all())
        app = (md.to(torch.bfloat16).float()).pow(2).sum(-1)
        g2["bf16_cast_energy_rel_diff_descriptive"] = float(((app - e_man)[:, cm].abs() / e_man[:, cm].clamp_min(1e-30)).median())
    g2["rank2_identity_ok"] = g2["rank2_identity_rel_diff"] <= T["rank2_identity_rtol"]
    g2["energy_requires_grad"] = bool(E.requires_grad)
    g2["positions_absolute"] = bool(torch.equal(rec["pos"], torch.arange(X.shape[1], device=rec["pos"].device)))
    # locality gradient reaches every trainable tensor (perturbed state); no backbone gradient
    pb2 = make_r2(b, 42)
    perturb(pb2, P["perturb"]["scale_w"], P["perturb"]["scale_b"], P["perturb"]["seed"])
    _, s2, (zt2, zs2, X2, Y2, Z2, Mz2, E2) = d2_step(b, pb2, EditObserver(pb2), tiny, lam, mu, mask)
    le2, nz2 = edit_loss(E2, Mz2)
    le2.backward()
    gp = named_trainable(pb2)
    g2["edit_grad_norms"] = {k: float(v.grad.norm()) for k, v in gp.items()}
    g2["edit_grads_all_nonzero_finite"] = all(np.isfinite(v) and v > 0 for v in g2["edit_grad_norms"].values())
    g2["backbone_grads_none"] = all(p.grad is None for p in b.model.parameters())
    g2["orth_after_perturb"] = orth_error(pb2)
    # edit-loss arithmetic: explicit masked mean of the same float32 energies; teacher-incorrect and EOS excluded; empty mask
    with torch.no_grad():
        Mt = teacher_mask(zt2, Y2)
        manual_le = float((E2.detach().double()[Mz2] / D_MODEL).sum() / int(Mz2.sum()))
        eos_pos = torch.zeros_like(Mz2)
        for i, r in enumerate(tiny):
            eos_pos[i, START + len(r["target_ids"]) - 1] = True
    g2["edit_loss_value"] = float(le2)
    g2["edit_loss_matches_manual"] = abs(manual_le - float(le2)) <= T["edit_loss_rtol"] * max(1e-30, abs(manual_le))
    g2["edit_mask_subset_teacher_correct"] = bool((Mz2 & ~Mt).sum() == 0)
    g2["edit_mask_excludes_eos_and_prompt"] = bool((Mz2 & eos_pos).sum() == 0) and bool((Mz2[:, :START]).sum() == 0)
    g2["teacher_incorrect_zh_excluded"] = int((Z2.to(dev) & ~Mt).sum())
    g2["n_edit"] = nz2
    le0, n0 = edit_loss(E2.detach().requires_grad_(True), torch.zeros_like(Mz2))
    g2["empty_mask_zero"] = float(le0) == 0.0 and n0 == 0 and le0.requires_grad
    del pa, pb2
    res["G2"] = g2

    # ---- G3 Mandarin mask correctness and coverage (whole training set)
    tb = tok_bytes_fn(b.processor.tokenizer)
    FILL = fillers()
    rederived = {r["utterance_id"]: utterance_mask(tg[r["utterance_id"]]["target_text"], r["target_ids"], tb, FILL, K) for r in rt.tm}
    keys_ = ("zh_tokens", "en_tokens", "zh_sb_tokens", "ambiguous", "mappable")
    bad_tok = 0
    for r in rt.tm:
        ids = r["target_ids"]
        for t in mask[r["utterance_id"]]["zh_tokens"]:
            s = tb(ids[t]).decode("utf-8", errors="ignore")
            if t >= len(ids) - 1 or ids[t] == rt.eos or any(c.isascii() and c.isalnum() for c in s):
                bad_tok += 1
    shift_rows = zrows[:8]
    Xs, Ys = build_tf_batch([r["prompt"] for r in shift_rows], [r["target_ids"] for r in shift_rows], rt.eos, rt.eos)
    Zs = token_mask_tensor(Ys, [r["target_ids"] for r in shift_rows], [mask[r["utterance_id"]]["zh_tokens"] for r in shift_rows])
    shift_ok = all(int(Ys[i, START + t]) == r["target_ids"][t] and int(Xs[i, START + t]) == (r["target_ids"][t - 1] if t else r["prompt"][-1])
                   for i, r in enumerate(shift_rows) for t in mask[r["utterance_id"]]["zh_tokens"])
    cov = collections.Counter()
    per_dlg = collections.Counter()
    per_utt = {}
    per_batch = []
    t_cov = time.time()
    for j in range(0, len(rt.tm), 8):
        rows = rt.tm[j:j + 8]
        Xc, Yc, ec = rt.tf_batch(rows)
        with torch.no_grad():
            zc = b.model(encoder_outputs=(ec,), decoder_input_ids=Xc, use_cache=False).logits
        M = teacher_mask(zc, Yc).cpu()
        cov["supervised"] += int((Yc != -100).sum())
        cov["teacher_correct"] += int(M.sum())
        nb = 0
        for i, r in enumerate(rows):
            mr = mask[r["utterance_id"]]
            n = len(r["target_ids"])
            cov["eos_correct"] += int(M[i, START + n - 1])
            sb = set(mr["zh_sb_tokens"])
            for t in mr["zh_tokens"]:
                c_ = int(M[i, START + t])
                cov["zh_tc"] += c_
                cov["zh_all"] += 1
                cov["zh_sb_tc" if t in sb else "zh_nonsb_tc"] += c_
                cov["zh_sb_all" if t in sb else "zh_nonsb_all"] += 1
                per_dlg[r["dialogue_id"]] += c_
                per_utt[r["utterance_id"]] = per_utt.get(r["utterance_id"], 0) + c_
                nb += c_
            for t in mr["en_tokens"]:
                cov["en_tc"] += int(M[i, START + t])
                cov["en_all"] += 1
            for t, _ in mr["ambiguous"]:
                cov["ambiguous_tc"] += int(M[i, START + t])
        per_batch.append(nb)
    coverage = {"supervised_positions": cov["supervised"], "teacher_correct_positions": cov["teacher_correct"],
                "teacher_correct_fraction": cov["teacher_correct"] / cov["supervised"],
                "zh_tokens": cov["zh_all"], "zh_tokens_teacher_correct": cov["zh_tc"],
                "zh_coverage": cov["zh_tc"] / cov["zh_all"] if cov["zh_all"] else None,
                "edit_mask_share_of_teacher_correct": cov["zh_tc"] / cov["teacher_correct"],
                "zh_sb": [cov["zh_sb_tc"], cov["zh_sb_all"]], "zh_non_sb": [cov["zh_nonsb_tc"], cov["zh_nonsb_all"]],
                "en_tokens_teacher_correct": [cov["en_tc"], cov["en_all"]], "ambiguous_tokens_teacher_correct": cov["ambiguous_tc"],
                "eos_teacher_correct": [cov["eos_correct"], len(rt.tm)],
                "dialogues_with_edit_mask": sum(1 for v in per_dlg.values() if v > 0), "dialogues_total": len({r["dialogue_id"] for r in rt.tm}),
                "top3_dialogue_share": (sum(sorted(per_dlg.values(), reverse=True)[:3]) / cov["zh_tc"]) if cov["zh_tc"] else None,
                "per_microbatch_edit_positions": {"min": int(min(per_batch)), "median": float(np.median(per_batch)), "max": int(max(per_batch)),
                                                  "zero_batches": int(sum(x == 0 for x in per_batch)), "n_batches": len(per_batch)},
                "per_training_order_edit_positions": training_order_counts(rt.tm, per_utt),
                "seconds": time.time() - t_cov,
                "note": "teacher-correct edit-mask positions per micro-batch in the nine seed x epoch training orders (gate) and in "
                        "manifest order (reported); training-time masks are recomputed per micro-batch and logged per run"}
    res["coverage"] = coverage
    cg = C["coverage_gate"]
    res["G3"] = {"mask_hash_frozen": DRY is not None or sha(out("mask_private.jsonl")) == fz.get("mask_sha256"),
                 "static_hash_consistent": static["mask_sha256"] == sha(out("mask_private.jsonl")),
                 "rederived_equal": set(rederived) == set(mask) and all(rederived[u_][k] == mask[u_][k] for u_ in mask for k in keys_),
                 "zh_tokens_han_only": bad_tok == 0, "n_bad_zh_tokens": bad_tok, "decoder_shift_ok": shift_ok,
                 "mask_only_at_listed_positions": int(Zs.sum()) == sum(len(mask[r["utterance_id"]]["zh_tokens"]) for r in shift_rows),
                 "eos_never_in_mask": all(not bool(Zs[i, START + len(r["target_ids"]) - 1]) for i, r in enumerate(shift_rows)),
                 "prompt_positions_excluded": not bool(Zs[:, :START].any()),
                 "unmappable_fraction": static["unmappable_fraction"],
                 "unmappable_ok": static["unmappable_fraction"] <= C["mask"]["integrity_block"]["max_unmappable_fraction"],
                 "targets_match_manifest": static["targets_mismatch"] == 0,
                 "coverage_ok": (coverage["zh_coverage"] or 0) >= cg["min_teacher_correct_zh_coverage"]
                 and coverage["dialogues_with_edit_mask"] >= cg["min_dialogues_with_edit_mask"]
                 and all(v["zero_batches"] <= cg["max_empty_microbatch_fraction"] * v["n_batches"]
                         for v in coverage["per_training_order_edit_positions"].values()),
                 "static": {k: v for k, v in static.items() if k not in ("git", "per_batch")}}

    # ---- G4 GoldKL parity (mu = 0, observer installed)
    g4 = {}
    tiny4 = sh[:P["tf_rows"]]
    z0a, z0b = make_r2(b, 42), make_r2(b, 42)
    sd0a, sd0b = z0a.state_dict(), z0b.state_dict()
    g4["init_state_identical_incl_base"] = set(sd0a) == set(sd0b) and all(torch.equal(sd0a[k], sd0b[k]) for k in sd0a)
    for tag, (pa_, pb_) in (("zero_init", (z0a, z0b)), ("perturbed", (make_r2(b, 42), make_r2(b, 42)))):
        if tag == "perturbed":
            perturb(pa_, P["perturb"]["scale_w"], P["perturb"]["scale_b"], P["perturb"]["seed"])
            perturb(pb_, P["perturb"]["scale_w"], P["perturb"]["scale_b"], P["perturb"]["seed"])
        l1, s1, (zt1, zs1, X1, Y1) = kl_step(b, pa_, tiny4, lam)
        l1.backward()
        l2, s2_, (zt2_, zs2_, X2_, Y2_, Z2_, Mz2_, E2_) = d2_step(b, pb_, EditObserver(pb_), tiny4, lam, 0.0, mask)
        l2.backward()
        ga, gb_ = named_trainable(pa_), named_trainable(pb_)
        g4[tag] = {"loss_bitwise": bool(torch.equal(l1.detach(), l2.detach())), "ce_equal": s1["ce"] == s2_["ce"], "kl_equal": s1["kl"] == s2_["kl"],
                   "n_tok_equal": s1["n_tok"] == s2_["n_tok"], "mask_equal": s1["n_mask"] == s2_["n_mask"],
                   "mask_tensor_equal": bool(torch.equal(teacher_mask(zt1, Y1), teacher_mask(zt2_, Y2_))),
                   "teacher_logits_bitwise": bool(torch.equal(zt1, zt2_)), "student_logits_bitwise": bool(torch.equal(zs1, zs2_)),
                   "grads_bitwise": all(torch.equal(ga[k].grad, gb_[k].grad) for k in ga),
                   "grad_max_abs_diff": max(float((ga[k].grad - gb_[k].grad).abs().max()) for k in ga), "kl": s1["kl"], "n_mask": s1["n_mask"],
                   "edit": s2_["edit"], "n_edit": s2_["n_edit"], "ce": s2_["ce"]}
        del pa_, pb_
    lce, _, _, _, _ = batch_loss(b, make_r2(b, 42), tiny4, grad=True)
    g4["ce_equals_p2_1_ce_path"] = float(lce) == g4["zero_init"]["ce"]
    zi = make_r2(b, 42)
    _, szi, (ztz, zsz, _, _, _, _, _) = d2_step(b, zi, EditObserver(zi), zrows[:8], lam, mu, mask)
    g4["zero_init_with_mu"] = {"initial_kl": szi["kl"], "initial_kl_ok": abs(szi["kl"]) <= T["initial_kl_max"], "edit": szi["edit"],
                               "edit_zero": szi["edit"] == 0.0, "student_equals_teacher": bool(torch.equal(zsz.detach(), ztz))}
    tr = seed_config(rt.C1, 42)["training"]
    a3 = make_r2(b, 42)
    ref1 = torch.load(ctl[42]["dir"] / "epoch1" / "adapter.pt")
    base_key = "rot.parametrizations.weight.0.base"
    base_eq = torch.equal(a3.state_dict()[base_key], ref1[base_key])
    opt = make_opt(a3, rt.C1, tr)
    lg = {"micro": [], "steps": []}
    h = d2_epoch(b, a3, EditObserver(a3), opt, rt.tm, tr, 42, 1, lam, 0.0, mask, 0, lg)
    mine = a3.state_dict()
    hist1 = ctl[42]["history"][1]
    g4["replay_KL42_epoch1_mu0"] = {"init_base_equal": base_eq,
                                    "bitwise_equal": set(mine) == set(ref1) and all(torch.equal(mine[k], ref1[k]) for k in ref1),
                                    "max_abs_diff": max(float((mine[k] - ref1[k]).abs().max()) for k in ref1),
                                    "train_kl_equal": h["train_kl"] == hist1["train_kl"], "train_ce_equal": h["train_ce"] == hist1["train_ce"],
                                    "mask_positions_equal": h["mask_positions"] == hist1["mask_positions"],
                                    "supervised_tokens_equal": h["supervised_tokens"] == hist1["supervised_tokens"],
                                    "updates": h["updates_total"], "updates_equal": h["updates_total"] == hist1["updates_total"],
                                    "edit_mask_positions": h["edit_mask_positions"], "train_edit_mu0": h["train_edit"],
                                    "epoch_seconds": h["train_seconds"], "max_orth_err": max(s["orth_err"] for s in lg["steps"])}
    del a3, opt
    res["G4"] = g4

    # ---- G5 locality gradient: identity zero; calibration state finite and nonzero; weighted contribution under mu
    g5l = {}
    zi2 = make_r2(b, 42)
    gbz, gez, iz = grad_pieces(b, zi2, zrows[:8], lam, mask)
    g5l["identity_edit_zero"] = iz["edit"] == 0.0
    g5l["identity_edit_grads_zero"] = all(bool((x == 0).all()) for x in gez)
    src = load_control_epoch1(b, ctl)
    gbs, ges, info = grad_pieces(b, src, zrows[:8], lam, mask)
    nb_, ne_ = flat_norm(gbs), flat_norm(ges)
    g5l.update({"calib_state_edit": info["edit"], "calib_state_edit_positive_finite": bool(info["edit"] > 0 and np.isfinite(info["edit"])),
                "calib_state_edit_grad_norm": ne_, "calib_state_base_grad_norm": nb_,
                "edit_grads_finite_nonzero": all(bool(torch.isfinite(x).all()) for x in ges) and ne_ > 0,
                "per_tensor_edit": info["per_tensor_edit"], "mu": mu, "scaled_grad_ratio": mu * ne_ / nb_,
                "scaled_grad_ratio_ok": mu * ne_ / nb_ <= C["calibration"]["stop"]["max_scaled_ratio"],
                "mu_edit_over_base_loss": mu * info["edit"] / info["base_loss"], "cos_base_edit": flat_cos(gbs, ges)})
    cal = read_json(out("calibration_private.json"))
    g5l["calibration_verdict_ok"] = cal["verdict"] == "D2_CALIBRATION_OK" and cal["mu"] == mu
    del zi2, src
    res["G5"] = g5l

    # ---- G6 decoding integrity and resources
    g6 = {}
    tiny4 = sh[:P["tf_rows"]]
    X, Y, enc = rt.tf_batch(tiny4)
    q, u, zf = rt.site_record(enc, X)
    ad = make_r2(b, 42)
    g6["n_trainable"] = ad.n_trainable()
    g6["init"] = init_record(ad)
    g6["orth_err_init"] = orth_error(ad)
    g6["W_equals_R_b_zero"] = bool(torch.equal(ad.module.learned.weight, ad.module.rot.weight)) and bool((ad.module.learned.bias == 0).all())
    g6["zero_init_tf_logits_bitwise_frozen"] = bool(torch.equal(rt.logits(ad, enc, X), zf))
    pa = make_r2(b, 42)
    perturb(pa, P["perturb"]["scale_w"], P["perturb"]["scale_b"], P["perturb"]["seed"])
    zp = rt.logits(pa, enc, X)
    with pa.disabled():
        zd = rt.logits(pa, enc, X)
    g6["disabled_reproduces_frozen"] = bool(torch.equal(zd, zf))
    g6["prefix_0_2_unchanged"] = bool(torch.equal(zp[:, :START], zf[:, :START]))
    g6["content_positions_changed"] = not bool(torch.equal(zp[:, START:], zf[:, START:]))
    ck = pdir / "perturbed_adapter.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pa.state_dict(), ck)
    rl = make_r2(b, 42)
    rl.load_state_dict(torch.load(ck))
    g6["reload_logits_bitwise"] = bool(torch.equal(rt.logits(rl, enc, X), zp))
    g6["backbone_trainable"] = sum(p.requires_grad for p in b.model.parameters())
    base5 = {x["id"]: x for x in read_jsonl(art(pb["beam5"]))}
    base1 = {x["id"]: x for x in read_jsonl(art(pb["greedy"]))}
    dev20 = rt.dm[:P["dev_rows"]]
    e5 = e1 = 0
    ts = []
    for row in dev20:
        t0 = time.time()
        a5, _ = rt.decode(row, 5, ad)
        ts.append(time.time() - t0)
        a1, _ = rt.decode(row, 1, ad)
        e5 += a5["tokens"] == base5[row["utterance_id"]]["tokens"]
        e1 += a1["tokens"] == base1[row["utterance_id"]]["tokens"]
    g6["zero_init_dev_beam5_equal"], g6["zero_init_dev_greedy_equal"] = [e5, len(dev20)], [e1, len(dev20)]
    g6["zero_init_beam5_s_per_utt"] = float(np.mean(ts))
    lid_eq, changed = 0, []
    for row in dev20:
        inp = batch_model_inputs(b, [row["audio_path"]])
        with pa.active(b.model):
            lid_eq += frozen_lid(b, inp["input_features"], pa)["token"] == row["frozen_lid_token"]
        d5, _ = rt.decode(row, 5, pa)
        changed.append(d5["tokens"] != base5[row["utterance_id"]]["tokens"])
    rx = dev20[changed.index(True)] if any(changed) else rt.dm[0]
    ry = next(r for r in dev20 if r is not rx)
    sg, sb_ = spy_positions(rt, pa, rx, 1), spy_positions(rt, pa, rx, 5)
    x1, _ = rt.decode(rx, 5, pa)
    rt.decode(ry, 5, pa)
    x2, _ = rt.decode(rx, 5, pa)
    x3, _ = rt.decode(rx, 5, pa)
    fz_ = rt.decode(rx, 5, None)[0]
    g6.update({"lid_unchanged_with_adapter": [lid_eq, len(dev20)], "perturbed_changed_dev_decodes": [sum(changed), len(dev20)],
               "greedy_cache_positions": {k: v for k, v in sg.items() if k != "seq"}, "beam5_cache_positions": {k: v for k, v in sb_.items() if k != "seq"},
               "cached_greedy_vs_tf_mismatch": [greedy_vs_tf(rt, pa, row) for row in rt.dm[:P["cache_rows"]]],
               "xyx_no_state_leakage": x1["tokens"] == x2["tokens"], "deterministic_repeat": x2["tokens"] == x3["tokens"],
               "perturbed_changes_decode": x1["tokens"] != fz_["tokens"], "hooks_after": site_hook_count(b.model, 31),
               "module_hooks_after": len(pa.module._forward_hooks)})
    del ad, rl
    mb8 = zrows[:8]
    tim = {}
    for arm in ("CE", "GoldKL", "D2"):
        a_ = make_r2(b, 42)
        perturb(a_, P["perturb"]["scale_w"], P["perturb"]["scale_b"], P["perturb"]["seed"])
        ob_ = EditObserver(a_)
        tt = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            if arm == "CE":
                l_, _, _, _, _ = batch_loss(b, a_, mb8, grad=True)
            elif arm == "GoldKL":
                l_, _, _ = kl_step(b, a_, mb8, lam)
            else:
                l_, _, _ = d2_step(b, a_, ob_, mb8, lam, mu, mask)
            l_.backward()
            torch.cuda.synchronize()
            tt.append(time.time() - t0)
            for p in a_.parameters():
                p.grad = None
        tim[arm] = tt
        del a_
    mean = {a_: float(np.mean(v[1:])) for a_, v in tim.items()}
    n_mb = -(-len(rt.tm) // rt.C1["training"]["micro_batch"])
    dev_per_utt = g6["zero_init_beam5_s_per_utt"] * 1.65 + 0.15
    proj = 3 * n_mb * mean["D2"] + 4 * len(rt.dm) * dev_per_utt + 300
    g6.update({"timing": tim, "mean_micro_batch_seconds": mean, "d2_over_goldkl": mean["D2"] / mean["GoldKL"] - 1,
               "projected_job_seconds": proj, "adapter_file_bytes": ck.stat().st_size, "trainable_parameter_bytes_fp32": 4 * g6["n_trainable"],
               "mask_mapping_seconds_cpu": static["mapping_seconds"],
               "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())})
    fp_rows = rt.dm[:P["fp32_rows"]]
    cvf = {"bf16": [], "fp32": []}
    for row in fp_rows:
        _, e1_ = encode(b, [row["audio_path"]])
        cvf["bf16"].append(cached_vs_full(b, pa, e1_, row["prompt"] + base5[row["utterance_id"]]["tokens"][:-1], T["fp32_argmax_margin"]))
    w_before_cast = weights_hash(b.model)
    b.model.float()  # in-memory only, last step of this job
    for row in fp_rows:
        inp = batch_model_inputs(b, [row["audio_path"]])
        with torch.inference_mode():
            e1_ = b.model.model.encoder(inp["input_features"].float()).last_hidden_state
        cvf["fp32"].append(cached_vs_full(b, pa, e1_, row["prompt"] + base5[row["utterance_id"]]["tokens"][:-1], T["fp32_argmax_margin"]))
    g6["cached_vs_full"] = cvf
    g6["hooks_after_all"] = site_hook_count(b.model, 31)
    res["G6"] = g6
    res["weights_unchanged_before_fp32_cast"] = w_before_cast == rt.w0
    g7 = {"freeze_verified": res["G1"]["freeze_verified"], "mu_frozen": res["G1"]["mu_frozen"], "authorization_present": out("AUTHORIZATION.json").exists(),
          "git_clean": res["G1"]["git_clean"], "git_head_matches_freeze": DRY is None and read_json(freeze_path(chain_head()))["git"]["head"] == g["head"]}
    res["G7"] = g7

    gates = {
        "G1_provenance": all(v for k, v in res["G1"].items() if k != "git_head"),
        "G2_site_edit": all(g2[k] for k in ("observer_logits_bitwise_unchanged", "single_application_bitwise", "observer_energy_bitwise_manual",
                                            "rank2_identity_ok", "prompt_positions_zero_energy", "energy_requires_grad", "positions_absolute",
                                            "edit_grads_all_nonzero_finite", "backbone_grads_none", "edit_loss_matches_manual",
                                            "edit_mask_subset_teacher_correct", "edit_mask_excludes_eos_and_prompt", "empty_mask_zero"))
        and g2["observer_records"] == 1 and g2["adapter_calls"] == 1 and g2["records_after_single"] == 0 and g2["module_hooks_after_detach"] == 0
        and g2["n_edit"] > 0 and g2["orth_after_perturb"] <= T["orth_max_abs"],
        "G3_mandarin_mask": all(res["G3"][k] for k in ("mask_hash_frozen", "static_hash_consistent", "rederived_equal", "zh_tokens_han_only",
                                                        "decoder_shift_ok", "mask_only_at_listed_positions", "eos_never_in_mask",
                                                        "prompt_positions_excluded", "unmappable_ok", "targets_match_manifest", "coverage_ok")),
        "G4_goldkl_parity": g4["init_state_identical_incl_base"] and g4["ce_equals_p2_1_ce_path"]
        and all(all(g4[t][k] for k in ("loss_bitwise", "ce_equal", "kl_equal", "n_tok_equal", "mask_equal", "mask_tensor_equal",
                                        "teacher_logits_bitwise", "student_logits_bitwise", "grads_bitwise")) for t in ("zero_init", "perturbed"))
        and g4["zero_init_with_mu"]["initial_kl_ok"] and g4["zero_init_with_mu"]["edit_zero"] and g4["zero_init_with_mu"]["student_equals_teacher"]
        and all(g4["replay_KL42_epoch1_mu0"][k] for k in ("init_base_equal", "bitwise_equal", "train_kl_equal", "train_ce_equal",
                                                           "mask_positions_equal", "supervised_tokens_equal", "updates_equal")),
        "G5_locality_gradient": all(g5l[k] for k in ("identity_edit_zero", "identity_edit_grads_zero", "calib_state_edit_positive_finite",
                                                      "edit_grads_finite_nonzero", "scaled_grad_ratio_ok", "calibration_verdict_ok")),
        "G6_decoding_resources": g6["n_trainable"] == C["method"]["expected_trainable_parameters"] and g6["orth_err_init"] <= T["orth_max_abs"]
        and g6["W_equals_R_b_zero"] and g6["zero_init_tf_logits_bitwise_frozen"] and g6["disabled_reproduces_frozen"] and g6["prefix_0_2_unchanged"]
        and g6["content_positions_changed"] and g6["reload_logits_bitwise"] and g6["backbone_trainable"] == 0
        and g6["zero_init_dev_beam5_equal"][0] == len(dev20) and g6["zero_init_dev_greedy_equal"][0] == len(dev20)
        and g6["lid_unchanged_with_adapter"][0] == len(dev20) and g6["greedy_cache_positions"]["first"] == [0, 1, 2, 3]
        and g6["greedy_cache_positions"]["incremental_ok"] and g6["beam5_cache_positions"]["first"] == [0, 1, 2, 3]
        and g6["beam5_cache_positions"]["incremental_ok"] and all(x <= T["cached_greedy_vs_tf_max_mismatch"] for x in g6["cached_greedy_vs_tf_mismatch"])
        and g6["xyx_no_state_leakage"] and g6["deterministic_repeat"] and g6["perturbed_changes_decode"] and g6["hooks_after"] == 0
        and g6["module_hooks_after"] == 0 and g6["hooks_after_all"] == 0 and res["weights_unchanged_before_fp32_cast"]
        and all(x["max_abs_logp_observed"] <= T["fp32_cached_vs_full_max_abs_logp"] and x["argmax_agree_where_sure"] and x["positions_ok"] and x["finite"]
                for x in cvf["fp32"])
        and proj <= P["max_projected_job_seconds"] and g6["peak_reserved_bytes"] < 38e9,
        "G7_protocol_freeze": all(g7.values()),
    }
    verdict = "D2_READY_FOR_TRAINING" if all(gates.values()) else "D2_BLOCKED_" + "_".join(k.split("_")[0] for k, v in gates.items() if not v)
    rec = {"schema": "reft_d2_preflight_v1", "utc": now(), "dry_run": bool(DRY), "git": g, "slurm": slurm(), "env": rt.env, "cuda_libs": rt.libs,
           "results": res, "gates": gates, "verdict": verdict, "wall_seconds": time.time() - t_all}
    write_json(out(f"preflight_{jid}.json"), rec)
    write_json(out(f"mask_coverage_{jid}.json"), {"utc": now(), "static": {k: v for k, v in static.items() if k != "git"}, "teacher": coverage})
    print(json.dumps({"verdict": verdict, "gates": gates, "G1": res["G1"], "G2": g2, "coverage": coverage,
                      "G3": {k: v for k, v in res["G3"].items() if k != "static"}, "G4": g4, "G5": g5l,
                      "G6": {k: v for k, v in g6.items() if k != "init"}, "G7": g7}, indent=1, default=str)[:60000])


def newest_preflight():
    ps = sorted(out().glob("preflight_*.json"), key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else -1)
    return ps[-1] if ps else None


# ====================================================================== train (GPU)

def run_train(seed: int):
    import torch
    from reft.arch_s0s1_adapters import init_record, orth_error
    from reft.arch_s0s1_train import Runtime, make_opt
    from reft.auto_policy import frozen_lid
    from reft.p2_1b_train import seed_config
    if DRY:
        raise SystemExit("D2 training never runs in dry-run mode")
    C = cd()
    assert seed in C["seeds"]
    fz = check_freeze(require_mu=True)
    mu = fz["mu"]
    if not out("AUTHORIZATION.json").exists():
        raise SystemExit("D2 not authorized")
    pf = newest_preflight()
    if pf is None or read_json(pf)["verdict"] != "D2_READY_FOR_TRAINING" or read_json(pf)["dry_run"]:
        raise SystemExit(f"D2 preflight not READY: {pf}")
    od = out("training", f"D2_seed{seed}")
    if od.exists():
        raise SystemExit(f"{od} exists; runs are write-once")
    mask = load_mask()
    if sha(out("mask_private.jsonl")) != fz["mask_sha256"]:
        raise SystemExit("mask metadata differs from the frozen hash")
    lam = C["objective"]["lambda_kl"]
    ctl = control_runs()[seed]
    torch.manual_seed(seed)                       # P2-1b order: seed -> backbone -> adapter
    rt = Runtime()
    b = rt.b
    tr = seed_config(rt.C1, seed)["training"]
    from reft.arch_s0s1_adapters import make_arch_adapter
    ad = make_arch_adapter("R2", b.model, seed, b.device)   # seed-specific identity init, never the calibration state
    if ad.n_trainable() != C["method"]["expected_trainable_parameters"]:
        raise SystemExit(f"R2 has {ad.n_trainable()} trainable parameters")
    init = init_record(ad)
    ref1 = torch.load(ctl["dir"] / "epoch1" / "adapter.pt")
    base_key = "rot.parametrizations.weight.0.base"
    pairing = {"orth_err_init": orth_error(ad), "base_equal_GoldKL_seed": bool(torch.equal(ad.state_dict()[base_key], ref1[base_key])),
               "identity_init": bool(torch.equal(ad.module.learned.weight, ad.module.rot.weight)) and bool((ad.module.learned.bias == 0).all())}
    if not (pairing["base_equal_GoldKL_seed"] and pairing["identity_init"]):
        raise SystemExit(f"initial state not the seed-specific R2 identity paired with GoldKL: {pairing}")
    opt = make_opt(ad, rt.C1, tr)
    obs = EditObserver(ad)
    base = {x["id"]: x for x in read_jsonl(art(rt.C1["frozen_baseline"]["beam5"]))}
    torch.cuda.reset_peak_memory_stats()
    t_all = time.time()

    def dev_decode(ep):
        res = {"beam5": [], "greedy": []}
        t0 = time.time()
        for r in rt.dm:
            for reg, nb in (("beam5", 5), ("greedy", 1)):
                rec, inp = rt.decode(r, nb, ad)
                res[reg].append({"id": r["utterance_id"], **rec})
            with ad.active(b.model):
                assert frozen_lid(b, inp["input_features"], ad)["token"] == r["frozen_lid_token"], "LID changed under adapter"
        for reg, v in res.items():
            write_jsonl(od / f"epoch{ep}" / f"dev_{reg}.jsonl", v)
        return time.time() - t0, res

    sec, res0 = dev_decode(0)
    ident = sum(x["tokens"] == base[x["id"]]["tokens"] for x in res0["beam5"])
    if ident != len(rt.dm):
        raise SystemExit(f"zero-init adapter does not reproduce the frozen baseline ({ident}/{len(rt.dm)})")
    hist = [{"epoch": 0, "zero_init_identity_beam5": ident, "dev_decode_seconds": sec, "init_hashes": init, **pairing}]
    steps = 0
    for ep in range(1, tr["epochs"] + 1):
        lg = {"micro": [], "steps": []}
        h = d2_epoch(b, ad, obs, opt, rt.tm, tr, seed, ep, lam, mu, mask, steps, lg)
        steps = h["updates_total"]
        write_jsonl(od / "diagnostics" / f"epoch{ep}_steps.jsonl", lg["steps"])
        write_jsonl(od / "diagnostics" / f"epoch{ep}_microbatches.jsonl", lg["micro"])
        (od / f"epoch{ep}").mkdir(parents=True, exist_ok=True)
        torch.save(ad.state_dict(), od / f"epoch{ep}" / "adapter.pt")
        s_, _ = dev_decode(ep)
        hk = ctl["history"][ep]
        st = lg["steps"]
        mi = lg["micro"]

        def mmean(k):
            v = [x[k] for x in mi if x[k] is not None]
            return float(np.mean(v)) if v else None
        diag = {"orth_err_max": max(x["orth_err"] for x in st),
                "grad_norm_preclip_mean": {n: float(np.mean([x["grad_norm_preclip"][n] for x in st])) for n in st[0]["grad_norm_preclip"]},
                "update_norm_mean": {n: float(np.mean([x["update_norm"][n] for x in st])) for n in st[0]["update_norm"]},
                "param_norm_end": st[-1]["param_norm"], "lr": sorted({x["lr"] for x in st}),
                "edit_mask_per_microbatch": {"min": min(x["n_edit"] for x in mi), "median": float(np.median([x["n_edit"] for x in mi])),
                                             "max": max(x["n_edit"] for x in mi), "zero": sum(x["n_edit"] == 0 for x in mi)},
                **{f"{k}_mean": mmean(k) for k in ("e_zh_tc", "e_zh_all", "e_en_tc", "e_en_all", "e_labelled", "mu_edit", "edit", "kl", "ce")},
                "mu": mu,
                "paired_with_GoldKL": {"mask_positions_equal": h["mask_positions"] == hk["mask_positions"],
                                       "supervised_tokens_equal": h["supervised_tokens"] == hk["supervised_tokens"],
                                       "updates_equal": h["updates_total"] == hk["updates_total"]},
                "goldkl_train_kl_historical": hk["train_kl"]}
        hist.append({**h, "diagnostics": diag, "dev_decode_seconds": s_, "adapter_sha256": sha(od / f"epoch{ep}" / "adapter.pt"),
                     "adapter_file_bytes": (od / f"epoch{ep}" / "adapter.pt").stat().st_size})
        print(json.dumps({k: v for k, v in hist[-1].items() if k != "diagnostics"} | {"orth_err_max": diag["orth_err_max"],
                                                                                      "paired": diag["paired_with_GoldKL"]}), flush=True)
    w1 = weights_hash(b.model)
    lat = {reg: float(np.mean([x["seconds"] for x in read_jsonl(od / f"epoch{tr['epochs']}" / f"dev_{reg}.jsonl")])) for reg in ("beam5", "greedy")}
    write_json(od / "manifest.json", {
        "schema": "reft_d2_run_v1", "utc": now(), **fz, "git": git_state(), "env": rt.env, "slurm": slurm(), "seed": seed, "mu": mu,
        "objective": C["objective"], "mask": C["mask"], "mask_sha256": sha(out("mask_private.jsonl")), "training": tr, "lr": rt.C1["methods"]["R2"]["lr"],
        "history": hist, "weights_before": rt.w0, "weights_after": w1, "weights_unchanged": rt.w0 == w1 == rt.expected_w, "cuda_libs": rt.libs,
        "preflight": pf.name, "preflight_sha256": sha(pf),
        "efficiency": {"trainable_parameters": ad.n_trainable(), "adapter_bytes_fp32": 4 * ad.n_trainable(),
                       "adapter_file_bytes": hist[-1]["adapter_file_bytes"], "optimizer_updates": steps,
                       "supervised_tokens": sum(x.get("supervised_tokens", 0) for x in hist),
                       "train_seconds": sum(x.get("train_seconds", 0) for x in hist),
                       "teacher_forward_seconds": sum(x.get("teacher_forward_seconds", 0) for x in hist),
                       "dev_decode_seconds": sum(x.get("dev_decode_seconds", 0) for x in hist), "wall_seconds": time.time() - t_all,
                       "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                       "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "mean_decode_seconds": lat}})


# ====================================================================== verify (GPU)

def run_verify():
    import torch
    from reft.arch_s0s1_train import Runtime
    fz = check_freeze()
    if out("verify.json").exists():
        raise SystemExit("verify.json exists; write-once")
    rt = Runtime()
    res = {}
    for s in SEEDS:
        rd = out("training", f"D2_seed{s}")
        man = read_json(rd / "manifest.json")
        res[rd.name] = {}
        for h in man["history"][1:]:
            p = rd / f"epoch{h['epoch']}" / "adapter.pt"
            ad = make_r2(rt.b, s)
            ad.load_state_dict(torch.load(p))
            rec = {x["id"]: x for x in read_jsonl(rd / f"epoch{h['epoch']}" / "dev_beam5.jsonl")}
            eq = sum(rt.decode(r, 5, ad)[0]["tokens"] == rec[r["utterance_id"]]["tokens"] for r in rt.dm[:20])
            res[rd.name][h["epoch"]] = {"sha_ok": sha(p) == h["adapter_sha256"], "reload_equal": [eq, 20],
                                        "n_trainable_ok": ad.n_trainable() == cd()["method"]["expected_trainable_parameters"]}
    w1 = weights_hash(rt.b.model)
    ok = rt.w0 == w1 == rt.expected_w and all(v["sha_ok"] and v["reload_equal"][0] == 20 and v["n_trainable_ok"] for x in res.values() for v in x.values())
    write_json(out("verify.json"), {"schema": "reft_d2_verify_v1", "utc": now(), **fz, "slurm": slurm(), "cuda_libs": rt.libs,
                                    "results": res, "weights_unchanged": rt.w0 == w1, "all_ok": ok,
                                    "historical_R2_GoldKL": "verified in P2-1b (p2_1b/verify.json all_ok)"
                                    if read_json(art("p2_1b/verify.json"))["all_ok"] else "NOT OK"})
    print(json.dumps({"all_ok": ok, "results": res}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mask", "freeze", "freeze-amend", "calibrate", "preflight", "train", "verify"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--doc")
    a = ap.parse_args(argv)
    if a.cmd == "train":
        run_train(a.seed)
    elif a.cmd == "freeze-amend":
        run_freeze_amend(a.doc)
    else:
        {"mask": run_mask, "freeze": run_freeze, "calibrate": run_calibrate, "preflight": run_preflight, "verify": run_verify}[a.cmd]()


if __name__ == "__main__":
    main()
