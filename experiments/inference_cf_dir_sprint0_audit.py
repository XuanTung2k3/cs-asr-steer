#!/usr/bin/env python
"""Independent DIR-SPRINT0 auditor (CPU). Stages: ``pre`` -> PASS_TO_DIR_SPRINT0, ``a`` -> DIR_SPRINT0_AUDIT_A,
``primary`` -> DIR_SPRINT0_AUDIT_PRIMARY (no references), ``full`` -> DIR_SPRINT0_AUDIT_FULL + terminal label;
``concepts-check`` (provider venv only) recomputes the frozen concept table from panphon.

It never imports the DIR-SPRINT0 runner, its mechanics module (``csasr.inference_cf.dir_sprint0``) or the evaluator.
Recomputed here from raw sealed artifacts with this file's own code: population selection and exposure exclusion, the
D3 legal set and candidate rule, the R2 gate window / float32 script masses / LID support / fallbacks, compatibility
statistics, D0 / D1 / D4 / random / D5 / shuffled-D5 / v_AC direction geometry, D5 evidence and calibration prototypes,
the planned arm targets, realized energies, top-1 decisions, the severe-EOS proxy, coverage, correction / corruption
counts, the paired bootstrap, every family predicate and the first-match terminal precedence. Shared code is limited to
mechanical hashing/JSON helpers, the pinned tokenizer partition, the frozen historical R0/S1 region primitives (for the
historical v_AC control), the SRD2-G0 auditor's independent reference mapping, and library numerics (numpy / torch).
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, digest, file_hash

CONFIG = "configs/inference_cf/dir_sprint0.json"
CONFIG_SHA = "sha256:b264460beedd09ae4b0c005ef5f8178494c49a52aa9ba2bec4bc116c984d27db"
FREEZE = "docs/inference_cf/DIR_SPRINT0_FREEZE.json"
POPULATION = "docs/inference_cf/DIR_SPRINT0_POPULATION.json"
RUN = "results/inference_cf/dir_sprint0/run1"
REMOTE = "origin/cs-asr-steer-inf"
MODEL = Path("/mnt/data/tungnx/whisper-large-v3")
VENV_PY = "/mnt/data/tungnx/cs-asr-steer/envs/dir_sprint0_d5/bin/python"
ARCHIVE = "/mnt/data/tungnx/cs-asr-steer/archives/dir_sprint0/run1"
CB = [50258, 50260, 50360, 50364]
EOS = 50257
DIM = 1280
PULSE_ARMS = ("D0", "D1", "D2", "VAC", "RND", "D3", "D4", "D5", "D5SH", "D2G", "D3G", "D4G", "D5G", "D5SHG", "RNDG")
ARMS = PULSE_ARMS + ("D3CD",)
GATED_BASE = {"D2G": "D2", "D3G": "D3", "D4G": "D4", "D5G": "D5", "D5SHG": "D5SH", "RNDG": "RND"}
FAMILIES = ("D3", "D4", "D5")
VARIANTS = {"D3": ("D3", "D3G"), "D4": ("D4", "D4G"), "D5": ("D5", "D5G")}
STRATA = ("EN-confusion", "EN-correct", "ZH-correct")
PRIORITY = ("mid_character", "localizer_fail", "local_support_fail", "baseline_provider_fail", "nonfinite_signal")
PROVIDER = PRIORITY[1:]
RUNNER = "experiments/inference_cf_dir_sprint0.py"
MECH = "src/csasr/inference_cf/dir_sprint0.py"
EVALUATOR = "experiments/inference_cf_dir_sprint0_evaluate.py"
AUDITOR = "experiments/inference_cf_dir_sprint0_audit.py"
TESTS = ("tests/test_dir_sprint0_impl.py", "tests/test_dir_sprint0_freeze.py")
SBATCH = "slurm/inference_cf_dir_sprint0.sbatch"
CONCEPT_TABLE = "results/inference_cf/dir_sprint0/design/d5_concept_table.json"
FEATURE_TABLE = "results/inference_cf/dir_sprint0/d5_provider/feature_table.json"
PROVIDER_MANIFEST = "results/inference_cf/dir_sprint0/d5_provider/provider_manifest.json"
FORBIDDEN_OPEN = (".parquet", "transcript", "evaluation_panel", "evaluation_units", "/ctc", "ctc_", "mms", "D-dev-confirm",
                  "D-test", "router-calib", "router_calib", "p2r_population", "positions.json", "/roles/", "evaluation.json",
                  "_eval.json", "_eval.npz", "phonemizer", "espeak")


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def pushed(rel: str) -> bool:
    if git("ls-files", rel) != rel:
        return False
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    if "sha256:" + hashlib.sha256(blob).hexdigest() != file_hash(ROOT / rel):
        return False
    c = git("log", "-n1", "--format=%H", "--", rel)
    return bool(c) and subprocess.run(["git", "merge-base", "--is-ancestor", c, REMOTE], cwd=ROOT).returncode == 0


def cfg_load() -> dict:
    if file_hash(ROOT / CONFIG) != CONFIG_SHA:
        raise SystemExit("config bytes changed")
    return json.loads((ROOT / CONFIG).read_text())


def unbf16(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.int16)
    return (a.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)


def fbits(x: float) -> str:
    return np.float64(x).view(np.uint64).item().to_bytes(8, "big").hex()


def quant(xs, p):
    return float(np.quantile(np.asarray(xs, dtype=np.float64), p)) if len(xs) else None


def close(a, b, tol=1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def population() -> dict:
    return json.loads((ROOT / POPULATION).read_text())


class Tok:
    """Tokenizer-derived facts: partition, suppression, language IDs, byte decoding, float32 script masses."""

    def __init__(self, cfg: dict):
        from transformers import WhisperTokenizer
        from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
        from csasr.inference_cf.core_r2 import tokenizer_partition
        self.tok = WhisperTokenizer.from_pretrained(str(MODEL), local_files_only=True)
        part = tokenizer_partition(self.tok)
        if part["hash"] != cfg["gate"]["partition_hash"]:
            raise SystemExit("partition hash")
        self.E = np.asarray(part["embedded_ids"], dtype=np.int64)
        self.M = np.asarray(part["matrix_ids"], dtype=np.int64)
        self.Eset, self.Mset = set(self.E.tolist()), set(self.M.tolist())
        gen = json.loads((MODEL / "generation_config.json").read_text())
        self.gen = gen
        self.suppress = list(gen["suppress_tokens"])
        self.begin = list(gen["begin_suppress_tokens"])
        if digest({"suppress": self.suppress, "begin": self.begin}) != cfg["model"]["suppression_hash"]:
            raise SystemExit("suppression hash")
        self.lang = sorted({int(v) for v in gen["lang_to_id"].values()})
        self.en, self.zh = int(gen["lang_to_id"]["<|en|>"]), int(gen["lang_to_id"]["<|zh|>"])
        self.bdec = {v: k for k, v in bytes_to_unicode().items()}
        self.special = set(self.tok.all_special_ids)
        self.eos = EOS

    def tbytes(self, i: int) -> bytes:
        return bytes(self.bdec[c] for c in self.tok.convert_ids_to_tokens(int(i)))

    def argmax(self, z: np.ndarray, t: int) -> int:
        x = np.asarray(z, dtype=np.float64).copy()
        x[self.suppress] = -np.inf
        if t == 0:
            x[self.begin] = -np.inf
        return int(np.argmax(x))

    def utf8_complete(self, ids) -> bool:
        try:
            b = b""
            for i in ids:
                if int(i) in self.special:
                    return False
                b += self.tbytes(i)
            b.decode("utf-8", errors="strict")
            return True
        except (UnicodeDecodeError, KeyError, ValueError):
            return False

    def masses32(self, z: np.ndarray) -> tuple[float, float]:
        """The frozen R2 definition: float32 full-vocabulary log-softmax, logsumexp over the partition (SRD2 amendment A1)."""
        import torch
        lp = torch.log_softmax(torch.as_tensor(np.asarray(z, dtype=np.float32)), dim=-1)
        pe = float(torch.exp(torch.logsumexp(lp[torch.as_tensor(self.E)], 0)))
        pm = float(torch.exp(torch.logsumexp(lp[torch.as_tensor(self.M)], 0)))
        return pe, pm

    def legal_static(self) -> list[int]:
        out = []
        sup = set(self.suppress)
        for i in range(EOS):
            if i in sup or i in self.special:
                continue
            b = self.tbytes(i)
            if not b or 0x80 <= b[0] <= 0xBF:
                continue
            try:
                b.decode("utf-8")
                out.append(i)
            except UnicodeDecodeError as e:
                if e.reason == "unexpected end of data" and e.end == len(b):
                    out.append(i)
        return out


def window(heads_q: np.ndarray, heard: int) -> dict:
    a = np.asarray(heads_q, dtype=np.float64)
    valid = min(1500, (int(heard) + 319) // 320)
    if heard <= 0 or a.ndim != 2 or a.shape[1] < valid:
        raise ValueError("localizer_fail")
    a = a[:, :valid]
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("localizer_fail")
    mean = a.mean(axis=0)
    tot = float(mean.sum())
    if not math.isfinite(tot) or tot <= 0:
        raise ValueError("localizer_fail")
    mean /= tot
    width = min(50, valid)
    masses = np.convolve(mean, np.ones(width, dtype=np.float64), mode="valid")
    start = int(np.argmax(masses))
    h = min(int(heard), 1500 * 320)
    size = min(50 * 320, h)
    s0 = min(start * 320, h - size)
    return {"start_frame": start, "end_frame": start + width, "start_sample": s0, "end_sample": s0 + size}


def support_E(pe, pm, ne, nm) -> float:
    eps = 1e-12
    A = math.log((pe + eps) / (pm + eps)) - math.log((ne + eps) / (nm + eps))
    return max(0.0, math.tanh(A / 2))


def softmax(z):
    x = np.asarray(z, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def heard_mean(h, heard):
    a = np.asarray(h, dtype=np.float64)
    valid = min(1500, (int(heard) + 319) // 320)
    m = a[:, :valid].mean(axis=0)
    return m / m.sum()


def tangent64(u, h) -> tuple[np.ndarray | None, str]:
    """Own float64 tangent normalization (readout rule): g - hhat <hhat,g>, / (||.|| + 1e-12), float32, guards."""
    g = np.asarray(u, dtype=np.float64).reshape(-1)
    hh = np.asarray(h, dtype=np.float64).reshape(-1)
    if not (np.isfinite(g).all() and np.isfinite(hh).all()):
        return None, "nonfinite_state_or_gradient"
    hn = float(np.linalg.norm(hh))
    if not hn > 1e-8:
        return None, "tiny_state"
    hhat = hh / hn
    gp = g - hhat * float(hhat @ g)
    tn = float(np.linalg.norm(gp))
    if not math.isfinite(tn) or tn < 1e-8:
        return None, "tiny_tangent"
    d = (gp / (tn + 1e-12)).astype(np.float32)
    dd = d.astype(np.float64)
    if abs(np.linalg.norm(dd) - 1.0) > 2e-6 or abs(hhat @ dd) > 1e-6:
        return None, "normalization_fail"
    return d, "ok"


def own_logsoftmax(z: np.ndarray, suppress: list[int]) -> np.ndarray:
    import torch
    x = torch.as_tensor(np.asarray(z, dtype=np.float32)).clone()
    x[suppress] = -float("inf")
    return torch.log_softmax(x, dim=-1).numpy().astype(np.float64)


def own_d3(zb: np.ndarray, zn: np.ndarray, legal: np.ndarray, suppress: list[int], b: int) -> dict:
    lb, ln = own_logsoftmax(zb, suppress), own_logsoftmax(zn, suppress)
    if np.isnan(lb).any() or np.isnan(ln).any() or not np.isfinite(lb).any():
        return {"status": "nonfinite_branch", "c_AP": None}
    thr = float(lb.max()) + math.log(1e-3)
    ok = legal & np.isfinite(lb) & np.isfinite(ln) & (lb >= thr)
    ids = np.flatnonzero(ok)
    if not ids.size:
        return {"status": "no_plausible_legal_candidate", "c_AP": None, "plausible": 0}
    s = 2 * lb[ids] - np.maximum(ln[ids], math.log(1e-12))
    best = s.max()
    c = int(ids[s == best].min())
    srt = np.sort(s)
    gap2 = float(srt[-1] - srt[-2]) if s.size > 1 else float("inf")
    return {"status": "baseline_is_candidate" if c == int(b) else "candidate", "c_AP": c, "plausible": int(ids.size),
            "top2_gap": gap2}


def own_weights(s: int, e: int, n: int) -> np.ndarray:
    j = np.arange(n, dtype=np.int64)
    ov = np.clip(np.minimum(j * 320 + 400, e) - np.maximum(j * 320, s), 0, None)
    return ov.astype(np.float64) / 400.0


def own_evidence(lp: np.ndarray, s0: int, s1: int, valid: np.ndarray, excluded: np.ndarray, C: np.ndarray, d5: dict) -> dict:
    w = own_weights(s0, s1, lp.shape[0])
    if int(np.count_nonzero(w)) < d5["validity"]["window_frames_min"]:
        return {"status": "too_few_window_frames"}
    if not np.isfinite(lp).all():
        return {"status": "nonfinite_posteriors"}
    p = np.exp(lp.astype(np.float64))
    sj = p[:, valid].sum(axis=1)
    we = w * (sj >= 0.5)
    E = float(we.sum())
    M = float(we @ sj)
    Xm = float(we @ p[:, excluded].sum(axis=1))
    if E < d5["validity"]["emitting_weight_min"]:
        return {"status": "low_emitting_weight", "E": E}
    if (M + Xm) <= 0 or Xm / (M + Xm) > d5["validity"]["excluded_share_max"]:
        return {"status": "excluded_mass", "E": E}
    if not M > 0:
        return {"status": "zero_valid_mass"}
    q = (we @ (p * valid[None, :]) @ C) / M
    return {"status": "ok", "q": q[:6], "E": E, "M": M}


def own_prototypes(Q: np.ndarray, H: np.ndarray, dl: list[str], d5: dict) -> dict:
    b = d5["bank"]
    U = H.astype(np.float64) / np.linalg.norm(H.astype(np.float64), axis=1, keepdims=True)
    dl = np.asarray(dl)
    names = sorted(set(dl.tolist()))
    V, mu, sd, st = np.zeros((6, U.shape[1])), np.zeros(6), np.zeros(6), []
    for k in range(6):
        q = Q[:, k]
        lo, hi = np.quantile(q, b["quantiles"][0]), np.quantile(q, b["quantiles"][1])
        mu[k] = np.mean([q[dl == d].mean() for d in names if (dl == d).any()])
        sd[k] = q.std()
        if not hi > lo:
            st.append("degenerate_quantiles")
            continue
        H_, L_ = q >= hi, q <= lo
        deltas = [U[H_ & (dl == d)].mean(0) - U[L_ & (dl == d)].mean(0) for d in names
                  if (H_ & (dl == d)).sum() >= b["dialogue_side_min"] and (L_ & (dl == d)).sum() >= b["dialogue_side_min"]]
        if len(deltas) < b["dialogues_min"] or H_.sum() < b["side_total_min"] or L_.sum() < b["side_total_min"]:
            st.append("insufficient_construction_coverage")
            continue
        if not sd[k] > 1e-6:
            st.append("degenerate_sd")
            continue
        m = np.mean(np.stack(deltas), axis=0)
        V[k] = m / np.linalg.norm(m)
        st.append("ok")
    return {"V": V, "mu": mu, "sd": sd, "status": st, "ok": all(s == "ok" for s in st)}


def own_shuffle(uid: str, ts: list[int]) -> dict:
    order = sorted(ts, key=lambda t: (hashlib.sha256(f"DIR-SPRINT0-d5-shuffle-v1|240924|{uid}|{t}".encode()).hexdigest(), t))
    n = len(order)
    return {order[i]: order[(i + 1) % n] for i in range(n)}


def own_random(uid: str, t: int) -> np.ndarray:
    seed = int(hashlib.sha256(f"DIR-SPRINT0-random-v1|240924|{uid}|{int(t)}|16".encode()).hexdigest()[:16], 16)
    v = np.random.Generator(np.random.PCG64(seed)).standard_normal(DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


def own_d0(he, hb) -> np.ndarray | None:
    d = np.asarray(he, dtype=np.float64) - np.asarray(hb, dtype=np.float64)
    n = float(np.linalg.norm(d))
    if not math.isfinite(n) or n < 1e-4:
        return None
    return (d / (n + 1e-6)).astype(np.float32)


def own_vac(hc, hm) -> np.ndarray | None:
    d = np.asarray(hc, dtype=np.float64) - np.asarray(hm, dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-4:
        return None
    return (d / (n + 1e-6)).astype(np.float32)


def static_scan(rel: str) -> dict:
    src = (ROOT / rel).read_text()
    tree = ast.parse(src)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            d = ast.get_docstring(node, clean=False)
            if d:
                docs.add(d)
    return {"imports": sorted(imports), "constants": [c for c in consts if c not in docs], "tree": tree, "src": src}


def own_exposed() -> tuple[set, dict]:
    """Exposure set from the raw addenda + SRD2 roster (own reading, own hash checks)."""
    add = json.loads((ROOT / "docs/current/DATA_EXPOSURE_ADDENDA.json").read_text())
    ok = all(file_hash(ROOT / add[k]["path"]) == add[k]["sha256"] for k in ("base_ledger", "base_machine_registry"))
    reg = json.loads((ROOT / add["base_machine_registry"]["path"]).read_text())
    ids = {r["utterance_id"] for r in reg["roster"] if r["exclusion_reasons"]}
    doc = set(ids)
    for e in add["entries"]:
        lst = list(e["utterance_ids"])
        ok &= len(lst) == e["count"] and "sha256:" + hashlib.sha256(json.dumps(lst, separators=(",", ":")).encode()).hexdigest() == e["utterance_ids_hash"]
        ids |= set(lst)
    return ids, {"hashes_ok": ok, "documented": sorted(doc), "addenda_sha256": file_hash(ROOT / "docs/current/DATA_EXPOSURE_ADDENDA.json")}


# ======================================================================================================
# concepts-check (provider venv; panphon)
# ======================================================================================================

def cmd_concepts_check(args) -> dict:
    import panphon
    ft = panphon.FeatureTable()
    t = json.loads((ROOT / CONCEPT_TABLE).read_text())
    rules = {"NAS": lambda f: f["nas"] == 1, "STOP": lambda f: f["son"] == -1 and f["cont"] == -1,
             "FRIC": lambda f: f["son"] == -1 and f["cont"] == 1, "LAB": lambda f: f["cons"] == 1 and f["lab"] == 1,
             "COR": lambda f: f["cons"] == 1 and f["cor"] == 1,
             "DOR": lambda f: f["cons"] == 1 and f["cor"] == -1 and f["lab"] == -1 and f["hi"] == 1,
             "VOI_OBS": lambda f: f["son"] == -1 and f["voi"] == 1, "ASP": lambda f: f["sg"] == 1}
    bad = []
    for r in t["rows"]:
        if r["values"] is None:
            continue
        fs = [dict(zip(ft.names, ft.fts(s).numeric())) for s in r["segments"]]
        vals = [sum(rules[k](f) for f in fs) / len(fs) for k in t["classes"]]
        if any(abs(a - b) > 1e-12 for a, b in zip(vals, r["values"])):
            bad.append(r["symbol"])
    ftab = json.loads((ROOT / FEATURE_TABLE).read_text())
    seg_match = all((r["values"] is None) == (fr["status"] != "segmental") and r["segments"] == fr["segments"]
                    for r, fr in zip(t["rows"], ftab["rows"]))
    dig = "sha256:" + hashlib.sha256(json.dumps({k: t[k] for k in ("classes", "rules", "rows", "feature_table_digest")},
                                                sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    out = {"mismatches": bad, "segments_match_feature_table": seg_match, "digest_recomputed": dig,
           "digest_ok": dig == t["concept_table_digest"], "classes": t["classes"]}
    print(json.dumps(out))
    return out


# ======================================================================================================
# pre
# ======================================================================================================

def cmd_pre(args) -> dict:
    cfg = cfg_load()
    checks, notes = {}, {}
    fr = json.loads((ROOT / FREEZE).read_text())
    checks["freeze_files"] = all(file_hash(ROOT / p) == h for p, h in fr["files"].items())
    P = population()
    pc = cfg["population"]
    sel = P["selected"]
    checks["population_hashes"] = (file_hash(ROOT / POPULATION) == pc["file_sha256"]
                                   and digest({k: v for k, v in P.items() if k != "manifest_hash"}) == P["manifest_hash"] == pc["manifest_hash"]
                                   and digest(sel) == pc["selected_hash"] and digest([r["utterance_id"] for r in sel]) == pc["selected_ids_hash"]
                                   and digest(P["roster"]) == pc["roster_hash"]
                                   and digest(P["calibration_bank"]["rows"]) == pc["calibration_bank"]["rows_hash"])
    checks["selected_240_12x20"] = len(sel) == 240 and set(Counter(r["dialogue_id"] for r in sel).values()) == {12} \
        and len({r["dialogue_id"] for r in sel}) == 20
    exposed, info = own_exposed()
    checks["exposure_hashes"] = info["hashes_ok"] and info["addenda_sha256"] == pc["exclusion_registry"]["addenda_sha256"]
    checks["exposure_excluded"] = len(exposed) == 700 and not exposed & {r["utterance_id"] for r in sel} \
        and {r["utterance_id"] for r in P["roster"] if r["exclusion_reasons"]} >= exposed
    # own selection recomputation from the identity roster
    elig = defaultdict(list)
    for r in P["roster"]:
        if not r["exclusion_reasons"]:
            elig[r["dialogue_id"]].append(r["utterance_id"])
    own = []
    for d in sorted(elig):
        own += sorted(elig[d], key=lambda u: (hashlib.sha256(f"DIR-SPRINT0-population-v1|240924|{u}".encode()).hexdigest(), u))[:12]
    checks["selection_recomputed"] = own == [r["utterance_id"] for r in sel]
    bank = P["calibration_bank"]["rows"]
    checks["bank_is_FULL300"] = sorted(r["utterance_id"] for r in bank) == info["documented"] and len(bank) == 300 \
        and not {r["utterance_id"] for r in bank} & {r["utterance_id"] for r in sel}
    checks["audio_mono16k"] = all(r["sample_rate"] == 16000 and r["channels"] == 1 for r in sel + bank)
    panel = json.loads((ROOT / RUN / "runtime_panel.json").read_text())
    ownp = {"schema": "dir_sprint0_runtime_panel_v1",
            "rows": [{"canonical_index": i, "utterance_id": r["utterance_id"], "audio_path": r["audio_path"],
                      "audio_full_sha256": r["audio_full_sha256"]} for i, r in enumerate(sel)],
            "bank_rows": [{"bank_index": i, "utterance_id": r["utterance_id"], "audio_path": r["audio_path"],
                           "audio_full_sha256": r["audio_full_sha256"]} for i, r in enumerate(bank)],
            "population_selected_ids_hash": digest([r["utterance_id"] for r in sel]),
            "bank_ids_hash": digest(sorted(r["utterance_id"] for r in bank))}
    ownp["runtime_hash"] = digest(ownp)
    checks["runtime_panel_is_exact_projection"] = panel == ownp
    m = json.loads((ROOT / RUN / "manifest.json").read_text())
    checks["manifest_self_hash"] = digest({k: v for k, v in m.items() if k != "manifest_hash"}) == m["manifest_hash"]
    checks["manifest_sources_match"] = all(file_hash(ROOT / p) == h for p, h in m["sources"].items())
    checks["manifest_pushed"] = pushed(f"{RUN}/manifest.json") and pushed(f"{RUN}/runtime_panel.json")
    checks["manifest_commit_ancestor"] = subprocess.run(["git", "merge-base", "--is-ancestor", m["git_commit"], "HEAD"], cwd=ROOT).returncode == 0
    changed = git("diff", "--name-only", m["git_commit"], "HEAD").splitlines()
    checks["only_run_dir_changed_since_manifest"] = all(c.startswith(RUN + "/") for c in changed)
    checks["design_freeze_commit_ancestor"] = subprocess.run(["git", "merge-base", "--is-ancestor", m["design_freeze_commit"], "HEAD"], cwd=ROOT).returncode == 0
    checks["historical_pins"] = all(file_hash(ROOT / p) == h for p, h in cfg["source_sha256"].items())
    checks["new_files_pinned"] = all(p in m["sources"] for p in (RUNNER, MECH, EVALUATOR, AUDITOR, SBATCH, *TESTS, CONCEPT_TABLE))
    checks["config_hash_in_manifest"] = m["config_sha256"] == CONFIG_SHA and m["runtime_hash"] == panel["runtime_hash"]
    model_now = {f: file_hash(MODEL / f) for f in cfg["model"]["files"]}
    checks["model_files"] = model_now == cfg["model"]["files"] == m["model"]["files"]
    checks["transformers_forward"] = file_hash(cfg["environment"]["transformers_forward_source"]) == cfg["environment"]["transformers_forward_sha256"]
    T = Tok(cfg)
    checks["partition_suppression_heads"] = (m["partition_hash"] == cfg["gate"]["partition_hash"]
                                             and m["suppression_hash"] == cfg["model"]["suppression_hash"]
                                             and T.gen["alignment_heads"] == cfg["gate"]["alignment_heads"] and len(T.lang) == 100)
    br = {k: v for k, v in cfg["branches"].items() if isinstance(v, dict)}
    checks["prompts_and_tasks"] = (all(T.tok.convert_ids_to_tokens(v["prompt"]) == v["tokens"] for v in br.values() if "tokens" in v)
                                   and T.gen["task_to_id"] == {"transcribe": 50360, "translate": 50359}
                                   and br["B0"]["prompt"] == CB and br["NULL"]["prompt"] == CB
                                   and br["E"]["prompt"] == [50258, 50259, 50360, 50364] and br["TL"]["prompt"] == [50258, 50260, 50359, 50364])
    legal = T.legal_static()
    checks["D3_legal_static_recomputed"] = len(legal) == cfg["families"]["D3"]["legal_static"]["count"] == 49695 \
        and digest(legal) == cfg["families"]["D3"]["legal_static"]["hash"] == m["D3_legal_static_hash"]
    # provider / tables
    pm = json.loads((ROOT / PROVIDER_MANIFEST).read_text())
    body = {k: v for k, v in pm.items() if k != "manifest_digest"}
    dig = "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    d5 = cfg["families"]["D5"]
    checks["provider_manifest"] = dig == pm["manifest_digest"] == d5["provider_manifest_digest"] == m["provider_manifest_digest"]
    checks["provider_model_files"] = all(file_hash(Path(pm["model"]["dir"]) / n) == h for n, h in pm["model"]["files"].items())
    ft = json.loads((ROOT / FEATURE_TABLE).read_text())
    ftd = "sha256:" + hashlib.sha256(json.dumps({k: ft[k] for k in ("feature_names", "valid_mask", "rows")}, sort_keys=True,
                                                ensure_ascii=False).encode()).hexdigest()
    checks["feature_table"] = ftd == ft["table_digest"] == d5["feature_table_digest"]
    venv = subprocess.run([VENV_PY, "-I", str(ROOT / AUDITOR), "concepts-check"], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": ""})
    try:
        cc = json.loads(venv.stdout.strip().splitlines()[-1])
    except Exception:
        cc = {"error": venv.stderr[-2000:]}
    notes["concepts_check"] = cc
    checks["concept_table_recomputed_from_panphon"] = venv.returncode == 0 and not cc.get("mismatches") and cc.get("digest_ok") is True \
        and cc.get("segments_match_feature_table") is True and cc.get("digest_recomputed") == d5["concept_table_digest"]
    ref = json.loads((ROOT / "results/inference_cf/dir_sprint0/design/d5_provider_reference.json").read_text())
    eng = json.loads((ROOT / "results/inference_cf/dir_sprint0/d5_provider/engineering_checks.json").read_text())
    with np.load(ROOT / "results/inference_cf/dir_sprint0/design/d5_provider_reference.npz") as z:
        refh = {c: "sha256:" + hashlib.sha256(str(z[k].dtype).encode() + str(z[k].shape).encode() + np.ascontiguousarray(z[k]).tobytes()).hexdigest()
                for c, k in ref["npz_keys"].items()}
    checks["provider_reference_is_sealed_engineering"] = all(refh[c] == eng["C"]["files"][c]["log_probs_sha256"] for c in refh) \
        and sorted(refh) == sorted(d5["provider_apparatus_clips"])
    checks["threshold_engineering_pinned"] = file_hash(ROOT / d5["threshold_engineering"]) == d5["threshold_engineering_sha256"]
    folds = json.loads((ROOT / cfg["families"]["D1"]["folds"]).read_text())
    checks["D1_folds"] = (file_hash(ROOT / cfg["families"]["D1"]["folds"]) == cfg["families"]["D1"]["folds_sha256"]
                          and sorted(folds["folds"]) == sorted({r["dialogue_id"] for r in sel})
                          and all(f["status"] == "ok" and file_hash(ROOT / f["vector_path"]) == cfg["families"]["D1"]["vector_file_sha256"][d]
                                  for d, f in folds["folds"].items()))
    # ---- static firewall / semantics ---------------------------------------------------------------
    run_s, mech_s = static_scan(RUNNER), static_scan(MECH)
    bad_imports = ("pyarrow", "csasr.evaluation", "csasr.data.normalize", "experiments.inference_cf_p0_r2_evaluate",
                   "experiments.inference_cf_p2r_population", "experiments.inference_cf_srd2_g0_evaluate",
                   "experiments.inference_cf_srd2_g0_audit", "experiments.inference_cf_dir_sprint0_evaluate",
                   "experiments.inference_cf_dir_sprint0_audit", "experiments.inference_cf_p2rj", "phonemizer", "panphon")
    checks["runner_imports_no_reference_code"] = not any(i.startswith(b) for i in run_s["imports"] + mech_s["imports"] for b in bad_imports)
    bad_const = (".parquet", "transcript", "/roles/", "evaluation_units", "target_ids", "target_set", "stratum", "ctc", "Y_ref",
                 "EN-confusion", "ZH-correct", "EN-correct")
    hits = sorted({c for c in run_s["constants"] + mech_s["constants"] for b in bad_const if b in c})
    notes["runner_suspicious_constants"] = hits
    checks["runner_has_no_reference_constants"] = not hits
    rsrc, msrc = run_s["src"], mech_s["src"]
    seg = {n.name: ast.get_source_segment(rsrc, n) for n in run_s["tree"].body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    gpu_fns = ("load_run", "Ctx", "load_model", "encode_path", "encode_wave", "cached_baseline", "branch_replay", "capture_utterance",
               "capture_bank", "cmd_capture", "cmd_phones", "Engine", "apparatus_old30", "pulse_utterance", "cmd_pulses")
    checks["gpu_phases_never_read_population_or_dialogues"] = all(
        f in seg and "dialogue_of" not in seg[f] and "dialogue_id" not in seg[f] and "POPULATION" not in seg[f] for f in gpu_fns)
    checks["gate_uses_raw_full_logits"] = ("raw_logits=full_logits[q]" in rsrc and "r2.full_replay(ctx.bundle, enc, list(ctx.prompt), content, attention=True)" in rsrc
                                           and "S.r2_gate(" in rsrc)
    checks["d2_is_original_readout_provider"] = "ReadoutDirection(" in rsrc and "DirectionContext(b_cache_pre_step=B.cache" in rsrc
    checks["d3_objective_c_minus_b"] = ("j = z[int(c_token)] - z[int(b_token)]" in msrc and "ZeroProbe(bundle, layer, query)" in msrc
                                        and 'c_token=int(qa["D3"]["c_AP"]), b_token=int(qa["expected_action"])' in rsrc)
    checks["d3_rule_constants"] = "D3_LOG_ALPHA = math.log(1e-3)" in msrc and "D3_LOG_NULL_FLOOR = math.log(1e-12)" in msrc
    checks["d4_transcribe_minus_translate"] = "X.d4_direction(h, hT[i])" in rsrc and "diff = a.double() - b.double()" in msrc
    checks["null_branch_zero_audio"] = "zeros = np.zeros(30 * bundle.sample_rate, dtype=np.float32)" in rsrc and \
        'ctx.null_enc, nlin = encode_wave(bundle, zeros)' in rsrc
    checks["phones_full_waveform"] = 'x, sr = sf.read(r["audio_path"], dtype="float32", always_2d=False)' in rsrc and "prov.posteriors(x, sr)" in rsrc
    checks["dose_targets_are_chords"] = ("target = X.arm_target(a, ctx.e_star, g_old)" in rsrc and "self.p2r.solve_scale(r_nat, v, float(target))" in rsrc
                                         and 'sol["s"] *' not in rsrc and "S.CachedSolver(" in rsrc
                                         and "return 0.0 if float(g) == 0.0 else float(e_star) * float(g)" in msrc)
    checks["zero_target_bypass"] = 'if target == 0.0:\n            rec["status"] = "zero_target"' in rsrc
    checks["baseline_explicit_argmax"] = "processed_argmax(lg, t, ctx.suppress, ctx.begin)" in rsrc and "topk" not in rsrc
    checks["advance_with_baseline_token_only"] = "S.feed_tokens(content, t, ctx.prompt)" in rsrc and "new = [nxt_fed]" in rsrc
    ev = (ROOT / EVALUATOR).read_text()
    checks["evaluator_barrier"] = ('"DIR_SPRINT0_AUDIT_PRIMARY: PASS"' in ev and 'ds.field("utterance_id").isin(ids)' in ev
                                   and ev.index("DIR_SPRINT0_AUDIT_PRIMARY: PASS required") < ev.index("refs, access = load_references(ids)"))
    me = static_scan(AUDITOR)
    checks["auditor_independent_imports"] = not any(i in ("experiments.inference_cf_dir_sprint0", "experiments.inference_cf_dir_sprint0_evaluate",
                                                          "csasr.inference_cf.dir_sprint0", "csasr.inference_cf.srd2_g0",
                                                          "experiments.inference_cf_srd2_g0") for i in me["imports"])
    sb = (ROOT / SBATCH).read_text()
    checks["slurm_contract"] = ("#SBATCH --partition=mig" in sb and "#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1" in sb
                                and "#SBATCH --time=03:00:00" in sb and "capture)" in sb and "pulses)" in sb
                                and "sbatch " not in sb.split("set -euo pipefail", 1)[1] and "--cpus-per-task=12" in sb and "--mem=96G" in sb)
    checks["terminal_precedence"] = cfg["terminal_precedence"] == [
        "DIR_SPRINT0_INVALID", "DIR_SPRINT0_COMPUTE_BLOCKED", "DIR_SPRINT0_OPPORTUNITY_INSUFFICIENT",
        "DIR_SPRINT0_STEERING_FEASIBILITY_SIGNAL", "DIR_SPRINT0_CAUSAL_POWER_WITH_DAMAGE", "DIR_SPRINT0_MECHANISTIC_EFFECT_ONLY",
        "DIR_SPRINT0_ALL_DIRECTIONS_INEFFECTIVE"]
    st = cfg["statistics"]
    checks["statistics_contract"] = st["family_size"] == 30 and st["bootstrap_replicates"] == 10000 and \
        close(st["family_quantiles"][0], 0.05 / 60) and cfg["mechanistic"]["family_size"] == 6
    c = cfg["compute"]
    checks["pre_job_forecast"] = c["job_A_conservative_seconds"] <= c["seconds_max_each"] and c["max_scientific_jobs"] == 2
    free = {p: shutil.disk_usage(p).free for p in (str(ROOT), "/mnt/data/tungnx")}
    notes["free_bytes"] = free
    checks["storage_ge_50GB"] = all(v >= 50e9 for v in free.values())
    q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h"], capture_output=True, text=True)
    notes["squeue"] = q.stdout.strip()
    checks["no_conflicting_jobs"] = q.returncode == 0 and not q.stdout.strip()
    checks["no_prior_outputs"] = not any((ROOT / RUN / d).exists() for d in ("capture", "bank", "phones", "construct", "pulses"))
    env = {**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}", "CUDA_VISIBLE_DEVICES": ""}
    suites = [*TESTS, "tests/test_dir_sprint0_d5_provider.py", "tests/test_srd2_g0_freeze.py", "tests/test_inference_cf_p2dir_directions.py"]
    res = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", "-p", "no:cacheprovider", *suites],
                         cwd=ROOT, env=env, capture_output=True, text=True)
    notes["pytest_tail"] = res.stdout.strip().splitlines()[-3:]
    checks["tests_pass"] = res.returncode == 0
    verdict = "PASS_TO_DIR_SPRINT0" if all(checks.values()) else "BLOCK_BEFORE_DIR_SPRINT0"
    return {"schema": "dir_sprint0_audit_pre_v1", "verdict": verdict, "manifest_hash": m["manifest_hash"], "checks": checks,
            "notes": notes, "git_head": git("rev-parse", "HEAD"), "created_unix": time.time()}


# ======================================================================================================
# common runtime checks
# ======================================================================================================

def opened_paths_check(paths: list[str], audio_rows: list[dict], m: dict, apparatus: bool, provider: bool) -> dict:
    audio = {os.path.abspath(r["audio_path"]) for r in audio_rows}
    pm = json.loads((ROOT / PROVIDER_MANIFEST).read_text())
    prov_prefixes = (pm["model"]["dir"], pm["engineering_audio"]["dir"])
    repo_allowed = {os.path.abspath(ROOT / p) for p in m["sources"]} | {os.path.abspath(ROOT / RUN / "manifest.json"),
                                                                      os.path.abspath(ROOT / "configs/model/whisper_large_v3.yaml")}
    app_prefixes = tuple(os.path.abspath(ROOT / p) for p in ("results/inference_cf/p2dir/exp1_run1", "results/inference_cf/p2_A_r1_L16",
                                                              "results/inference_cf/p0_r2/inference_panel.json"))
    fold_prefix = os.path.abspath(ROOT / "results/inference_cf/p2dir/folds_run1")
    cats = defaultdict(list)
    for p in paths:
        low = p.lower()
        if any(f.lower() in low for f in FORBIDDEN_OPEN) and not p.startswith(os.path.abspath(ROOT / RUN)):
            cats["forbidden"].append(p)
        elif p in audio:
            cats["audio"].append(p)
        elif p.startswith(str(MODEL)):
            cats["model"].append(p)
        elif provider and p.startswith(prov_prefixes):
            cats["provider"].append(p)
        elif p in repo_allowed:
            cats["pinned"].append(p)
        elif p.startswith(os.path.abspath(ROOT / RUN)):
            cats["run"].append(p)
        elif apparatus and p.startswith(app_prefixes):
            cats["apparatus"].append(p)
        elif p.startswith(fold_prefix):
            cats["folds"].append(p)
        elif p.startswith(str(ROOT)):
            cats["repo_other"].append(p)
        elif p.startswith("/mnt/"):
            cats["mnt_other"].append(p)
        else:
            cats["other"].append(p)
    ok = not cats["forbidden"] and not cats["mnt_other"] and not [p for p in cats["repo_other"] if not p.endswith(".py")]
    return {"ok": ok, "counts": {k: len(v) for k, v in cats.items()}, "forbidden": cats["forbidden"],
            "mnt_other": cats["mnt_other"], "repo_other": [p for p in cats["repo_other"] if not p.endswith(".py")], "other": cats["other"][:50]}


def base_checks() -> tuple[dict, dict, list, list, dict]:
    cfg = cfg_load()
    m = json.loads((ROOT / RUN / "manifest.json").read_text())
    panel = json.loads((ROOT / RUN / "runtime_panel.json").read_text())
    pre = json.loads((ROOT / RUN / "audit_PRE.json").read_text())
    checks = {"manifest_self_hash": digest({k: v for k, v in m.items() if k != "manifest_hash"}) == m["manifest_hash"],
              "pre_audit_pass": pre["verdict"] == "PASS_TO_DIR_SPRINT0" and pre["manifest_hash"] == m["manifest_hash"] and pushed(f"{RUN}/audit_PRE.json"),
              "sources_unchanged": all(file_hash(ROOT / p) == h for p, h in m["sources"].items()),
              "panel_hash": panel["runtime_hash"] == m["runtime_hash"]}
    return cfg, m, panel["rows"], panel["bank_rows"], checks


def load_seal(rel: str) -> tuple[dict, dict]:
    seal = json.loads((ROOT / RUN / rel).read_text())
    checks = {"seal_pushed": pushed(f"{RUN}/{rel}"), "seal_status": seal.get("status") == "SEALED",
              "seal_self_hash": digest({k: v for k, v in seal.items() if k != "seal_hash"}) == seal["seal_hash"]}
    bad = [f for f, h in seal["files"].items() if file_hash(ROOT / RUN / f) != h]
    checks["seal_files_match"] = not bad
    arch = seal["archive"]["content_addressed"]
    checks["archive_copies"] = all(Path(p).exists() and file_hash(p) == seal["files"][f] for f, p in arch.items())
    checks["archive_complete"] = all(f in arch for f in seal["files"] if f.endswith(".npz"))
    return seal, checks


def rows_of(seal: dict, sub: str, ids: list[str]) -> dict:
    out = {}
    for uid in ids:
        rel = f"{sub}/{uid}.json"
        out[uid] = json.loads((ROOT / RUN / rel).read_text()) if rel in seal["files"] else None
    return out


# ======================================================================================================
# audit A
# ======================================================================================================

def recompute_gate(T: Tok, g: dict, arr: dict, null: dict) -> tuple[list[dict], dict]:
    content, term = g["baseline"]["content_ids"], g["baseline"]["terminated"]
    inv = list(range(len(content) + 1)) if term == "eos" else list(range(len(content)))
    errs = Counter()
    cl, fl = unbf16(arr["cached_logits"]), unbf16(arr["full_logits"])
    cs, fs = unbf16(arr["cached_site"]), unbf16(arr["full_site"])
    heads = unbf16(arr["heads"])
    lid = {(int(k[0]), int(k[1])): p for k, p in zip(arr["lid_keys"], arr["lid_probs"])}
    li = {v: i for i, v in enumerate(T.lang)}
    heard = int(g["audio"]["samples"])
    flags = {"inventory_ok": g["inventory"] == inv and [x["t"] for x in g["queries"]] == inv,
             "query_index_ok": [x["query"] for x in g["queries"]] == [3 + t for t in g["inventory"]]}
    out = []
    for i, (t, x) in enumerate(zip(inv, g["queries"])):
        exp = content[t] if t < len(content) else EOS
        if x["query"] != 3 + t or x["expected_action"] != exp or T.argmax(cl[i], t) != exp or x["cached_argmax"] != exp:
            errs["query_identity_or_argmax"] += 1
        prefix = content[:t]
        utf8 = T.utf8_complete(prefix)
        if utf8 != x["utf8_complete"]:
            errs["utf8"] += 1
        fin = bool(np.isfinite(cs[i]).all() and np.isfinite(cl[i]).all())
        elig = t >= 1 and utf8 and all(int(p) < EOS for p in prefix) and fin
        if elig != x["structural"]["eligible"]:
            errs["structural"] += 1
        reasons, E, R, win = [], None, None, None
        if not utf8:
            reasons.append("mid_character")
        try:
            win = window(heads[i], heard)
        except ValueError:
            reasons.append("localizer_fail")
        pe = pm = None
        if np.isfinite(fl[i]).all():
            pe, pm = T.masses32(fl[i])
            R = max(0.0, pm - pe)
        else:
            reasons.append("nonfinite_signal")
        if not reasons:
            key = (win["start_sample"], win["end_sample"])
            if key not in lid:
                errs["lid_missing"] += 1
                reasons.append("local_support_fail")
            else:
                p = lid[key]
                E = support_E(float(p[li[T.en]]), float(p[li[T.zh]]), null["EN"], null["ZH"])
        fb = next((r for r in PRIORITY if r in reasons), None)
        gv = 0.0 if fb else E * R
        rg = x["gate"]
        if win is not None and rg["window"] is not None:
            if any(win[k] != rg["window"][k] for k in win):
                errs["window"] += 1
        elif (win is None) != (rg["window"] is None):
            errs["window_presence"] += 1
        if fb != rg["fallback_reason"]:
            errs["fallback"] += 1
        if pe is not None and rg["baseline"] is not None and not (abs(pe - rg["baseline"]["P_E"]) <= 1e-6 and abs(pm - rg["baseline"]["P_M"]) <= 1e-6):
            errs["script_mass"] += 1
        if E is not None and (rg["E"] is None or abs(E - rg["E"]) > 1e-9):
            errs["E"] += 1
        if abs(gv - x["g"]) > 1e-6 or fbits(x["g"]) != x["g_bits"]:
            errs["g"] += 1
        rec = {"t": t, "eligible": elig, "g": x["g"], "fallback": fb, "utf8": utf8, "window": win}
        if elig:
            tv = 0.5 * np.abs(softmax(fl[i]) - softmax(cl[i])).sum()
            pec, pmc = T.masses32(cl[i])
            a, b = fs[i].astype(np.float64), cs[i].astype(np.float64)
            nb = np.linalg.norm(b)
            rel = float(np.linalg.norm(a - b) / nb) if nb > 0 else None
            cosv = float(a @ b / (np.linalg.norm(a) * nb)) if nb > 0 and np.linalg.norm(a) > 0 else None
            agree = T.argmax(fl[i], t) == T.argmax(cl[i], t)
            c = x["compat"]
            if not (close(tv, c["TV"], 1e-6) and abs(abs(pe - pec) - c["PE_abs_diff"]) <= 1e-6 and abs(abs(pm - pmc) - c["PM_abs_diff"]) <= 1e-6
                    and agree == c["argmax_agree"] and close(rel, c["site_rel_L2"], 1e-6) and close(cosv, c["site_cos"], 1e-6)):
                errs["compat"] += 1
            rec.update(TV=float(tv), dPE=abs(pe - pec), dPM=abs(pm - pmc), agree=agree, rel=rel, cos=cosv)
        out.append(rec)
    probes = []
    for j, p in enumerate(g["probes"]):
        i = inv.index(p["t"])
        tv = 0.5 * np.abs(softmax(unbf16(arr["probe_logits"][j])) - softmax(fl[i])).sum()
        l1 = float(np.abs(heard_mean(heads[i], heard) - heard_mean(unbf16(arr["probe_heads"][j]), heard)).sum())
        if not (close(tv, p["TV"], 1e-6) and close(l1, p["attention_L1"], 1e-6)):
            errs["probe"] += 1
        probes.append({"TV": float(tv), "L1": l1, "future": p["future_mass_max"]})
    st = [x["t"] for x in g["queries"] if x["structural"]["eligible"]]
    if st and [p["t"] for p in g["probes"]] != sorted({st[0], st[(len(st) - 1) // 2], st[-1]}):
        errs["probe_selection"] += 1
    return out, {"errors": dict(errs), "probes": probes, **flags}


def provider_spot(cfg: dict, rows: list[dict], bank: list[dict], run: Path, n_each: int = 5) -> dict:
    """Own posterior recomputation (transformers Wav2Vec2ForCTC + own normalization) for a hash-ordered sample."""
    import soundfile as sf
    import torch
    from transformers import Wav2Vec2ForCTC
    torch.set_num_threads(4)
    pm = json.loads((ROOT / PROVIDER_MANIFEST).read_text())
    model = Wav2Vec2ForCTC.from_pretrained(pm["model"]["dir"], dtype=torch.float32).eval()

    def pick(rs):
        return sorted(rs, key=lambda r: hashlib.sha256(f"DIR-SPRINT0-audit-provider|{r['utterance_id']}".encode()).hexdigest())[:n_each]
    out = []
    for r in pick(rows) + pick(bank):
        uid = r["utterance_id"]
        rec = json.loads((run / "phones" / f"{uid}.json").read_text())
        x, sr = sf.read(r["audio_path"], dtype="float32", always_2d=False)
        xn = ((x - x.mean()) / np.sqrt(x.var() + 1e-7)).astype(np.float32)
        tf = torch.backends.cuda.matmul.allow_tf32
        with torch.inference_mode():
            lp = torch.log_softmax(model(torch.from_numpy(xn)[None], attention_mask=torch.ones((1, xn.size), dtype=torch.long)).logits[0].float(), -1).numpy()
        torch.backends.cuda.matmul.allow_tf32 = tf
        with np.load(run / "phones" / rec["arrays"]["file"]) as z:
            sealed = z["log_probs"]
        same = sealed.shape == lp.shape
        out.append({"utterance_id": uid, "shape_equal": same, "bitwise": bool(same and np.array_equal(sealed, lp)),
                    "max_abs": float(np.abs(sealed.astype(np.float64) - lp).max()) if same else None,
                    "argmax_agreement": float((sealed.argmax(1) == lp.argmax(1)).mean()) if same else None})
    tol = cfg["families"]["D5"]["provider_apparatus_tolerance"]
    ok = all(o["shape_equal"] and (o["bitwise"] or (o["max_abs"] <= tol["max_abs_logprob"] and o["argmax_agreement"] >= tol["argmax_agreement"]))
             for o in out)
    return {"ok": ok, "rows": out}


def audit_construction(cfg: dict, T: Tok, P: dict, rows: list[dict], bank: list[dict], cap: dict, brow: dict, cons: dict,
                       run: Path, own_gate: dict) -> dict:
    """Own D3 / D0 / D1 / D4 / random / v_AC / D5 (evidence, prototypes, directions, shuffle) recomputation."""
    import torch  # noqa: F401  (log-softmax numerics)
    from csasr.inference_cf import r0_regions as R0
    from csasr.inference_cf.s1_evidence import heard_attention, select_target
    errs, notes = Counter(), {}
    d5 = cfg["families"]["D5"]
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in P["selected"]}
    bdlg = {r["utterance_id"]: r["dialogue_id"] for r in P["calibration_bank"]["rows"]}
    legal = np.zeros(51866, dtype=bool)
    legal[T.legal_static()] = True
    ft = json.loads((ROOT / FEATURE_TABLE).read_text())
    status = np.array([r["status"] for r in ft["rows"]])
    valid, excluded = status == "segmental", status == "excluded"
    ct = json.loads((ROOT / CONCEPT_TABLE).read_text())
    C = np.zeros((len(ft["rows"]), 8))
    for r in ct["rows"]:
        if r["values"] is not None:
            C[r["id"]] = r["values"]
    phones_rt = {w: json.loads((run / f"phones_runtime_{w}.json").read_text()) for w in ("prospective", "bank")}
    provider_ok = all(r.get("apparatus_pass") is True and r.get("status") in ("completed", "failed") for r in phones_rt.values())

    def phones(uid):
        p = run / "phones" / f"{uid}.json"
        if not p.exists():
            return None
        rec = json.loads(p.read_text())
        if rec.get("status") != "ok":
            return None
        with np.load(run / "phones" / rec["arrays"]["file"]) as z:
            return z["log_probs"]
    # ---- bank evidence + prototypes ----
    BQ, BH, BD = [], [], []
    bank_ok = 0
    for r in bank:
        b = brow[r["utterance_id"]]
        if b is None or b["status"] != "ok":
            continue
        bank_ok += 1
        with np.load(run / "bank" / b["arrays"]["file"]) as z:
            sites = unbf16(z["cached_site"])
        lp = phones(r["utterance_id"])
        content = b["baseline"]["content_ids"]
        for i, q in enumerate(b["queries"]):
            t = q["t"]
            prefix = content[:t]
            elig = t >= 1 and T.utf8_complete(prefix) and all(int(p) < EOS for p in prefix) and bool(np.isfinite(sites[i]).all())
            if elig != q["structural"]["eligible"]:
                errs["bank_structural"] += 1
            if not elig or q["window"] is None or lp is None:
                continue
            ev = own_evidence(lp, q["window"]["start_sample"], q["window"]["end_sample"], valid, excluded, C, d5)
            if ev["status"] == "ok":
                BQ.append(ev["q"])
                BH.append(sites[i])
                BD.append(bdlg[r["utterance_id"]])
    proto_rec = json.loads((run / "construct" / "prototypes.json").read_text())
    blocked = []
    if not provider_ok:
        blocked.append("provider_apparatus_or_child_failure")
    if bank_ok != len(bank):
        blocked.append("bank_capture_incomplete")
    own_p = None
    if BQ:
        own_p = own_prototypes(np.asarray(BQ), np.asarray(BH), BD, d5)
        if not own_p["ok"]:
            blocked.append("prototype_construction_failed")
    else:
        blocked.append("no_valid_bank_evidence")
    notes["D5_blocked_own"] = blocked
    if sorted(blocked) != sorted(proto_rec["blocked_reasons"]):
        errs["D5_block_status"] += 1
    D5_ok = not blocked
    if D5_ok:
        with np.load(run / "construct" / "prototypes.npz") as z:
            Vs, mus, sds = z["V"], z["mu"], z["sd"]
        if not (np.abs(Vs - own_p["V"]).max() <= 1e-9 and np.abs(mus - own_p["mu"]).max() <= 1e-12 and np.abs(sds - own_p["sd"]).max() <= 1e-12):
            errs["prototypes"] += 1
        notes["prototype_max_abs_diff"] = float(np.abs(Vs - own_p["V"]).max())
        notes["n_valid_bank_queries"] = len(BQ)
    # ---- D1 vectors ----
    folds = json.loads((ROOT / cfg["families"]["D1"]["folds"]).read_text())
    fv = {d: np.load(ROOT / f["vector_path"]).astype(np.float32) for d, f in folds["folds"].items()}
    # ---- prospective ----
    maxerr = Counter()
    d3_counts, near_ties = Counter(), 0
    cover = Counter()
    for r in rows:
        uid = r["utterance_id"]
        g, cn = cap[uid], cons[uid]
        if g is None or g["status"] != "ok" or cn is None:
            errs["row_missing"] += 1
            continue
        with np.load(run / "capture" / g["arrays"]["file"]) as z:
            arr = {k: z[k] for k in z.files}
        with np.load(run / "construct" / cn["arrays"]["file"]) as z:
            dirs = {k: z[k] for k in z.files}
        cl, nl = unbf16(arr["cached_logits"]), unbf16(arr["NULL_logits"])
        hB, hE, hT = unbf16(arr["cached_site"]), unbf16(arr["E_site"]), unbf16(arr["TL_site"])
        mkeys = arr["mask_keys"].tolist()
        msite = unbf16(arr["mask_site"]) if len(mkeys) else None
        mask_of = {int(j): k for k, (j, a, b) in enumerate(mkeys)}
        lp = phones(uid)
        sts = [i for i, q in enumerate(g["queries"]) if q["structural"]["eligible"]]
        if cn["structural_ts"] != [g["queries"][i]["t"] for i in sts] or list(dirs["structural_ts"]) != cn["structural_ts"]:
            errs["construct_structural"] += 1
            continue
        # R0 track (historical primitives on sealed window probabilities + audio crops)
        import soundfile as sf
        x, _ = sf.read(r["audio_path"], dtype="float32", always_2d=False)
        rc = cfg["families"]["VAC"]["regions"]
        grid = R0.window_grid(len(x), rc["window_samples"], rc["stride_samples"])
        if [list(w) for w in grid] != arr["r0_grid"].tolist():
            errs["r0_grid"] += 1
        codes = [R0.classify_window(arr["r0_window_probs"][k], rc["language_ids"], x[s:e], rc)["code"] for k, (s, e) in enumerate(grid)]
        track = R0.intervals_to_track(R0.sample_intervals(grid, codes, len(x), rc["sample_vote_fraction_min"]), len(x))
        heard = min(len(x), 480000)
        th = np.array([(s, e, k) for s, e, k in R0.track_to_intervals(track[:heard])], dtype=np.int64).reshape(-1, 3)
        if not np.array_equal(th, arr["r0_track_heard_intervals"]):
            errs["r0_track"] += 1
        zrec = {}
        for k, i in enumerate(sts):
            q = g["queries"][i]
            t = q["t"]
            st = cn["status"][k]
            # D3
            od = own_d3(cl[i], nl[i], legal, T.suppress, q["expected_action"])
            d3_counts[od["status"]] += 1
            if od["status"] != q["D3"]["status"] or od["c_AP"] != q["D3"].get("c_AP"):
                if od.get("top2_gap", 1.0) <= 1e-9:
                    near_ties += 1
                else:
                    errs["D3_candidate"] += 1
            exp_cd = q["D3"]["c_AP"] if q["D3"]["status"] == "candidate" else q["expected_action"]
            if q["D3CD_top1"] != exp_cd:
                errs["D3CD"] += 1
            # D0
            d0 = own_d0(hE[i], hB[i])
            if (d0 is None) != np.isnan(dirs["D0"][k]).any() or (d0 is not None and np.abs(d0 - dirs["D0"][k]).max() > 1e-6):
                errs["D0"] += 1
            # D1
            if np.abs(fv[dlg[uid]] - dirs["D1"][k]).max() != 0:
                errs["D1"] += 1
            # D4
            d4, s4 = tangent64(hB[i].astype(np.float64) - hT[i].astype(np.float64), hB[i])
            if (d4 is None) != np.isnan(dirs["D4"][k]).any() or (d4 is not None and np.abs(d4 - dirs["D4"][k]).max() > 1e-6):
                errs["D4"] += 1
            maxerr["D4"] = max(maxerr["D4"], 0 if d4 is None else float(np.abs(d4 - dirs["D4"][k]).max()))
            cover["D4"] += int(d4 is not None)
            # random
            if not np.array_equal(own_random(uid, t), dirs["RND"][k]):
                errs["RND"] += 1
            # v_AC
            tg = select_target(heard_attention(unbf16(arr["cached_heads"][i]), heard, cfg["families"]["VAC"]["heard_mass_min"]),
                               th, x, heard, {"regions": cfg["families"]["VAC"]["s1_regions"]})
            if tg["status"] != q["VAC_target"]["status"] or tg.get("bounds") != q["VAC_target"].get("bounds"):
                errs["VAC_target"] += 1
            vac = None
            if tg["status"] == "OK" and i in mask_of:
                vac = own_vac(hB[i], msite[mask_of[i]])
            if (vac is None) != np.isnan(dirs["VAC"][k]).any() or (vac is not None and np.abs(vac - dirs["VAC"][k]).max() > 1e-6):
                errs["VAC"] += 1
            cover["VAC"] += int(vac is not None)
            # D5
            win = own_gate[uid][g["inventory"].index(t)]["window"]
            d5d = None
            if D5_ok and win is not None and lp is not None:
                ev = own_evidence(lp, win["start_sample"], win["end_sample"], valid, excluded, C, d5)
                if ev["status"] == "ok":
                    z = (ev["q"] - own_p["mu"]) / own_p["sd"]
                    if np.linalg.norm(z) >= 1e-6:
                        d5d, _ = tangent64(z @ own_p["V"], hB[i])
                        if d5d is not None:
                            zrec[t] = z
            if (d5d is None) != np.isnan(dirs["D5"][k]).any() or (d5d is not None and np.abs(d5d - dirs["D5"][k]).max() > 1e-6):
                errs["D5"] += 1
            maxerr["D5"] = max(maxerr["D5"], 0 if d5d is None else float(np.abs(d5d - dirs["D5"][k]).max()))
            cover["D5"] += int(d5d is not None)
            if st["D3"]["status"] != q["D3"]["status"]:
                errs["status_D3"] += 1
        # shuffled D5
        don = own_shuffle(uid, sorted(zrec))
        for k, i in enumerate(sts):
            t = g["queries"][i]["t"]
            if t in don:
                sh, _ = tangent64(zrec[don[t]] @ own_p["V"], hB[i])
            else:
                sh = None
            if (sh is None) != np.isnan(dirs["D5SH"][k]).any() or (sh is not None and np.abs(sh - dirs["D5SH"][k]).max() > 1e-6):
                errs["D5SH"] += 1
    return {"errors": dict(errs), "notes": notes, "D3_status_counts": dict(d3_counts), "D3_near_ties": near_ties,
            "max_direction_abs_error": dict(maxerr), "coverage_counts": dict(cover), "D5_blocked": blocked, "provider_ok": provider_ok}


def own_job_a_metrics(cfg: dict, rows: list, recs: dict, row_flags: dict, probes: list, null: dict, dlg: dict, cap: dict) -> dict:
    dialogues = sorted(set(dlg.values()))
    struct = [(u, x) for u, rr in recs.items() for x in rr if x["eligible"]]
    nz = [(u, x) for u, x in struct if x["g"] > 0]
    cp = [x for rr in recs.values() for x in rr if x["utf8"]]
    fp = [x for x in cp if x["fallback"] not in PROVIDER]
    tv = [x["TV"] for _, x in struct]
    rel = [x["rel"] for _, x in struct]
    allq = [x for u in recs for x in cap[u]["queries"]]
    bi = [x["branches"][b]["identity"] for x in allq for b in ("E", "TL", "NULL")]
    bf = [x["branches"][b]["site_finite"] for x in allq for b in ("E", "TL", "NULL")]
    nf = [cap[u]["queries"][i]["branches"]["NULL"]["logits_finite"] for u, rr in recs.items() for i, x in enumerate(rr) if x["eligible"]]
    return {"rows_complete": len(recs), "dialogues_complete": sum(1 for d in dialogues if all(u in recs for u in dlg if dlg[u] == d)),
            "inventory_complete_fraction": sum(1 for u in recs if row_flags[u]["inventory_ok"]) / len(rows),
            "structural_queries": len(struct), "structural_dialogues": len({dlg[u] for u, _ in struct}),
            "nonzero_gates": len(nz), "nonzero_gate_dialogues": len({dlg[u] for u, _ in nz}),
            "finite_provider_fraction": len(fp) / len(cp) if cp else None,
            "identity_fraction": sum(1 for u in recs if cap[u]["full_replay"]["input_ids_sha256"] == digest(CB + cap[u]["baseline"]["content_ids"])
                                     and cap[u]["baseline"]["lineage_ok"] and cap[u]["full_replay"]["shared_encoder"] is True
                                     and row_flags[u]["query_index_ok"]) / len(rows),
            "cached_baseline_identity_bitwise": bool(recs) and all(cap[u]["baseline"]["replay_bitwise"] and cap[u]["baseline"]["lineage_ok"] for u in recs),
            "raw_TV_median": quant(tv, .5), "raw_TV_p99": quant(tv, .99), "raw_TV_max": max(tv) if tv else None,
            "PE_abs_diff_p99": quant([x["dPE"] for _, x in struct], .99), "PM_abs_diff_p99": quant([x["dPM"] for _, x in struct], .99),
            "processed_argmax_agreement": sum(x["agree"] for _, x in struct) / len(struct) if struct else None,
            "site_rel_L2_p99": quant(rel, .99) if None not in rel else None, "site_rel_L2_max": max(rel) if rel and None not in rel else None,
            "site_cosine_min": min(x["cos"] for _, x in struct) if struct and all(x["cos"] is not None for _, x in struct) else None,
            "causal_probe_TV_max": max(p["TV"] for p in probes) if probes else None,
            "causal_probe_attention_L1_max": max(p["L1"] for p in probes) if probes else None,
            "future_self_attention_mass_max": max([cap[u]["full_replay"]["future_mass_max"] for u in recs] + [p["future"] for p in probes]) if recs else None,
            "null_EN_abs_error": abs(null["EN"] - cfg["gate"]["null_EN_historical"]),
            "null_ZH_abs_error": abs(null["ZH"] - cfg["gate"]["null_ZH_historical"]),
            "branch_identity_fraction": sum(bi) / len(bi) if bi else None, "branch_finite_fraction": sum(bf) / len(bf) if bf else None,
            "D3_null_finite_fraction": sum(nf) / len(nf) if nf else None}


def own_a_predicates(cfg: dict, M: dict) -> dict:
    g, c, nul = cfg["job_A_gate"], cfg["compatibility"], cfg["gate"]["null_probability_abs_tolerance"]

    def ok(k, op, v):
        x = M.get(k)
        return x is not None and {"==": x == v, ">=": x >= v if x is not None else False, "<=": x <= v if x is not None else False}[op]
    cov = all([ok("rows_complete", "==", g["rows_complete"]), ok("dialogues_complete", "==", g["dialogues_complete"]),
               ok("inventory_complete_fraction", "==", 1.0), ok("structural_queries", ">=", g["pulse_structural_queries_min"]),
               ok("structural_dialogues", ">=", g["pulse_structural_dialogues_min"]), ok("nonzero_gates", ">=", g["nonzero_old_gate_queries_min"]),
               ok("nonzero_gate_dialogues", ">=", g["nonzero_old_gate_dialogues_min"]),
               ok("finite_provider_fraction", ">=", g["finite_provider_fraction_on_complete_prefix_min"]),
               ok("branch_identity_fraction", "==", 1.0), ok("branch_finite_fraction", "==", 1.0)])
    comp = all([ok("identity_fraction", "==", 1.0), M.get("cached_baseline_identity_bitwise") is True,
                ok("raw_TV_median", "<=", c["full_vs_cached_raw_TV_median_max"]), ok("raw_TV_p99", "<=", c["full_vs_cached_raw_TV_p99_max"]),
                ok("raw_TV_max", "<=", c["full_vs_cached_raw_TV_max"]),
                ok("PE_abs_diff_p99", "<=", c["full_vs_cached_raw_script_mass_abs_diff_p99_max"]),
                ok("PM_abs_diff_p99", "<=", c["full_vs_cached_raw_script_mass_abs_diff_p99_max"]),
                ok("processed_argmax_agreement", ">=", c["full_vs_cached_processed_argmax_agreement_min"]),
                ok("site_rel_L2_p99", "<=", c["full_vs_cached_site_relative_L2_p99_max"]),
                ok("site_rel_L2_max", "<=", c["full_vs_cached_site_relative_L2_max"]),
                ok("site_cosine_min", ">=", c["full_vs_cached_site_cosine_min"]),
                ok("causal_probe_TV_max", "<=", c["prefix_only_vs_full_TV_max"]),
                ok("causal_probe_attention_L1_max", "<=", c["prefix_only_vs_full_attention_L1_max"]),
                ok("future_self_attention_mass_max", "<=", c["decoder_self_attention_future_mass_max"]),
                ok("null_EN_abs_error", "<=", nul), ok("null_ZH_abs_error", "<=", nul)])
    return {"job_A_coverage_pass": cov, "compatibility_pass": comp}


def cmd_a(args) -> dict:
    cfg, m, rows, bank, checks = base_checks()
    seal, sc = load_seal("job_A_seal.json")
    checks.update(sc)
    run = ROOT / RUN
    ids = [r["utterance_id"] for r in rows]
    cap, cons = rows_of(seal, "capture", ids), rows_of(seal, "construct", ids)
    brow = rows_of(seal, "bank", [r["utterance_id"] for r in bank])
    checks["all_capture_rows_sealed"] = all(v is not None for v in cap.values()) and all(v is not None for v in cons.values())
    rt = json.loads((run / "capture_runtime.json").read_text())
    checks["runtime_completed"] = rt["status"] == "completed" and rt["manifest_hash"] == m["manifest_hash"]
    checks["weights_unchanged"] = rt["weights_unchanged"] is True and rt["weights_digest_start"] == rt["weights_digest_end"]
    checks["no_parameter_grads"] = rt["parameters_with_grad"] == 0 and rt["parameters_requiring_grad"] == 0
    op = opened_paths_check(rt["opened_paths"], rows + bank, m, apparatus=False, provider=False)
    ph = {w: json.loads((run / f"phones_runtime_{w}.json").read_text()) for w in ("prospective", "bank")}
    oph = {w: opened_paths_check(ph[w].get("opened_paths", []), rows + bank, m, apparatus=False, provider=True) for w in ph}
    checks["opened_paths_allowlist"] = op["ok"] and all(v["ok"] for v in oph.values())
    null = {"EN": rt["null"]["EN"], "ZH": rt["null"]["ZH"]}
    checks["null_pins"] = abs(null["EN"] - cfg["gate"]["null_EN_historical"]) <= 1e-6 and abs(null["ZH"] - cfg["gate"]["null_ZH_historical"]) <= 1e-6
    T = Tok(cfg)
    P = population()
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in P["selected"]}
    errs, recs, probes, flags, own_gate = Counter(), {}, [], {}, {}
    for r in rows:
        uid = r["utterance_id"]
        g = cap[uid]
        if g is None or g["status"] != "ok":
            errs["row_not_ok"] += 1
            continue
        if g["audio"]["full_sha256"] != r["audio_full_sha256"]:
            errs["audio_hash"] += 1
        with np.load(run / "capture" / g["arrays"]["file"]) as z:
            arr = {k: z[k] for k in ("cached_logits", "full_logits", "cached_site", "full_site", "heads", "lid_keys", "lid_probs",
                                     "probe_logits", "probe_heads")}
        rr, info = recompute_gate(T, g, arr, null)
        recs[uid], flags[uid] = rr, info
        own_gate[uid] = rr
        probes += info["probes"]
        errs.update(info["errors"])
    checks["independent_gate_recomputation"] = not errs
    M = own_job_a_metrics(cfg, rows, recs, flags, probes, null, dlg, cap)
    agree = {k: (v == seal["metrics"].get(k)) if isinstance(v, bool) else close(v, seal["metrics"].get(k), 1e-9) for k, v in M.items()}
    checks["metrics_agree_with_seal"] = all(agree.values())
    preds = own_a_predicates(cfg, M)
    checks["predicates_agree_with_seal"] = all(preds[k] == seal["predicates"][k] for k in preds)
    con = audit_construction(cfg, T, P, rows, bank, cap, brow, cons, run, own_gate)
    checks["independent_construction"] = not con["errors"]
    fam_own = {"D3": not (M["D3_null_finite_fraction"] is not None and M["D3_null_finite_fraction"] >= cfg["job_A_gate"]["D3_null_finite_fraction_min"]),
               "D4": False, "D5": bool(con["D5_blocked"])}
    checks["family_construction_agrees"] = all(fam_own[f] == seal["family_construction"][f]["blocked"] for f in FAMILIES)
    spot = provider_spot(cfg, rows, bank, run)
    checks["provider_spot_recomputation"] = spot["ok"]
    verdict = "DIR_SPRINT0_AUDIT_A: PASS" if all(checks.values()) else "DIR_SPRINT0_AUDIT_A: BLOCK"
    return {"schema": "dir_sprint0_audit_A_v1", "verdict": verdict, "checks": checks, "errors": dict(errs), "construction": con,
            "metrics": M, "metric_agreement": agree, "predicates": preds, "family_blocked": fam_own, "provider_spot": spot,
            "opened_paths": {"main": op, "phones": oph}, "job_A_seal_sha256": file_hash(run / "job_A_seal.json"),
            "manifest_hash": m["manifest_hash"], "references_used": False, "git_head": git("rev-parse", "HEAD"), "created_unix": time.time()}


# ======================================================================================================
# audit PRIMARY (no references)
# ======================================================================================================

def primary_recompute(cfg: dict, T: Tok, rows: list[dict], cap: dict, cons: dict, pulses: dict, blocked: dict, run: Path) -> dict:
    e = float(cfg["dose"]["e_star"])
    dose = cfg["dose"]
    errs = Counter()
    status = Counter()
    per_arm = {a: {"req": 0, "exe": 0, "planned": [], "lost": [], "real": []} for a in PULSE_ARMS}
    sq, ch, cp = [], [], []
    ok_flags = {"restore": True, "clean": True, "scratch2": True, "scratch3": True, "integrity": True}
    executed = 0
    auto = {"D2": 0, "D3": 0}
    eos = {a: 0 for a in ARMS}
    for r in rows:
        uid = r["utterance_id"]
        g, cn, p = cap[uid], cons[uid], pulses[uid]
        if p is None or p["status"] != "ok":
            errs["pulse_row_not_ok"] += 1
            continue
        if p["inventory"] != g["inventory"] or [x["t"] for x in p["queries"]] != g["inventory"]:
            errs["inventory"] += 1
        with np.load(run / "pulses" / p["arrays"]["file"]) as z:
            arr = {k: z[k] for k in z.files}
        with np.load(run / "pulses" / p["logits"]["file"]) as z:
            lg = {k: z[k] for k in z.files}
        with np.load(run / "capture" / g["arrays"]["file"]) as z:
            cl = unbf16(z["cached_logits"])
            csite = unbf16(z["cached_site"])
        with np.load(run / "construct" / cn["arrays"]["file"]) as z:
            dirs = {k: z[k] for k in z.files}
        if not np.array_equal(arr["exec_keys"], lg["exec_keys"]):
            errs["exec_keys"] += 1
        ex_logits = unbf16(lg["exec_logits"])
        ex_ffn = unbf16(arr["exec_ffn"]).astype(np.float64)
        T_ = len(g["baseline"]["content_ids"])
        si, k = -1, 0
        for i, x in enumerate(p["queries"]):
            gx = g["queries"][i]
            t = x["t"]
            b0 = T.argmax(cl[i], t)
            ok_flags["clean"] &= bool(x["clean"]["logits_bitwise_vs_A"] and x["clean"]["site_bitwise_vs_A"])
            if x["arms"]["B0"]["top1"] != b0 or b0 != gx["expected_action"]:
                errs["B0_top1"] += 1
            if x["structural"] != gx["structural"]["eligible"]:
                errs["structural"] += 1
            cdtop = gx["D3"]["c_AP"] if x["structural"] and gx["D3"]["status"] == "candidate" else b0
            for a in ARMS:
                top = cdtop if a == "D3CD" else x["arms"][a]["top1"]
                if top == EOS and b0 != EOS and T_ - t >= 10:
                    eos[a] += 1
            if not x["structural"]:
                if any(x["arms"][a]["executed"] or x["arms"][a]["top1"] != b0 for a in PULSE_ARMS):
                    errs["ineligible_not_noop"] += 1
                continue
            si += 1
            kk = cn["structural_ts"].index(t)
            rnat = unbf16(arr["native_r"][si]).astype(np.float64)
            if not np.array_equal(unbf16(arr["native_r"][si]), csite[i]):
                errs["native_r_vs_A_site"] += 1
            d2 = x["d2"]
            ok_flags["scratch2"] &= bool(d2["scratch_logits_bitwise"] and d2["scratch_site_bitwise"])
            auto["D2"] += int(d2["counters"].get("autograd_calls", 0))
            own2 = None
            if d2["status"] == "ok":
                own2, _ = tangent64(arr["d2_gradient"][si], rnat)
                if own2 is None or np.abs(own2.astype(np.float64) - arr["d2_direction"][si].astype(np.float64)).max() > 1e-6:
                    errs["d2_geometry"] += 1
            elif not np.isnan(arr["d2_direction"][si]).all():
                errs["d2_invalid_stored"] += 1
            d3 = x["d3"]
            want3 = gx["D3"]["status"] == "candidate" and not blocked["D3"]
            if want3 != ("scratch_logits_bitwise" in d3):
                errs["d3_presence"] += 1
            d3ok = False
            if want3 and "scratch_logits_bitwise" in d3:
                ok_flags["scratch3"] &= bool(d3["scratch_logits_bitwise"] and d3["scratch_site_bitwise"])
                auto["D3"] += int(d3["counters"].get("autograd_calls", 0))
                if d3["c_token"] != gx["D3"]["c_AP"] or d3["b_token"] != gx["expected_action"]:
                    errs["d3_tokens"] += 1
                if d3["status"] == "ok":
                    own3, _ = tangent64(arr["d3_gradient"][si], rnat)
                    if own3 is None or np.abs(own3.astype(np.float64) - arr["d3_direction"][si].astype(np.float64)).max() > 1e-6:
                        errs["d3_geometry"] += 1
                    d3ok = True
            valid = {"D2": d2["status"] == "ok", "D3": d3ok}
            for fam in ("D0", "D1", "D4", "D5", "D5SH", "VAC", "RND"):
                valid[fam] = not np.isnan(dirs[fam][kk]).any()
                if fam in ("D4",) and blocked["D4"] or fam in ("D5", "D5SH") and blocked["D5"]:
                    valid[fam] = False
            gv = float(gx["g"])
            for a in PULSE_ARMS:
                ar = x["arms"][a]
                fam = GATED_BASE.get(a, a)
                tg = e if a not in GATED_BASE else (0.0 if gv == 0 else e * gv)
                status[(a, ar["status"])] += 1
                if fbits(ar["target"]) != fbits(tg):
                    errs["target"] += 1
                if tg == 0 and ar["status"] != "zero_target":
                    errs["zero_target_status"] += 1
                if tg > 0 and not valid[fam] and ar["status"] != "invalid_direction":
                    errs["invalid_status"] += 1
                if tg > 0 and valid[fam] and ar["status"] == "invalid_direction":
                    errs["valid_marked_invalid"] += 1
                pa = per_arm[a]
                if tg > 0 and valid[fam]:
                    pa["req"] += 1
                    pa["planned"].append(tg)
                if ar["executed"]:
                    executed += 1
                    pa["exe"] += 1
                    key = lg["exec_keys"][k]
                    if int(key[0]) != i or int(key[1]) != PULSE_ARMS.index(a) or ar["logits"] != f"exec:{k}":
                        errs["exec_index"] += 1
                    top = T.argmax(ex_logits[k], t)
                    cons_c = float(np.linalg.norm(ex_ffn[k] - rnat))
                    gd = ar["guards"]
                    if top != ar["top1"]:
                        errs["exec_top1"] += 1
                    if not close(cons_c, ar["consumed_chord_actual"], 1e-9) or not close(cons_c, gd["consumed_chord"], 1e-9):
                        errs["consumed_chord"] += 1
                    t2 = ar["target"] ** 2
                    sq += [abs(gd["proposed_chord"] ** 2 / t2 - 1), abs(cons_c * cons_c / t2 - 1)]
                    ch += [abs(gd["proposed_chord"] / ar["target"] - 1), abs(cons_c / ar["target"] - 1)]
                    cp.append(abs(cons_c / gd["proposed_chord"] - 1))
                    ok_flags["integrity"] &= bool(ar["integrity_ok"] and ar["solver_calls"] == 1 and ar["ffn_equals_preview_bitwise"]
                                                  and ar["solver"]["rel_sq_err"] <= dose["relative_squared_energy_error_max"])
                    pa["real"].append(cons_c * cons_c)
                    k += 1
                else:
                    if ar["top1"] != b0:
                        errs["noop_top1"] += 1
                    if tg > 0 and valid[fam]:
                        pa["lost"].append(tg * tg)
            if x["restore_bitwise"] is not None:
                ok_flags["restore"] &= bool(x["restore_bitwise"])
            elif any(x["arms"][a]["executed"] for a in PULSE_ARMS):
                errs["restore_missing"] += 1
        if k != len(lg["exec_keys"]):
            errs["exec_count"] += 1
    energy = {}
    for a, pa in per_arm.items():
        tot = math.fsum(sorted(v * v for v in pa["planned"]))
        energy[a] = {"matched_fraction": pa["exe"] / pa["req"] if pa["req"] else None,
                     "lost_fraction": math.fsum(sorted(pa["lost"])) / tot if tot > 0 else None,
                     "requests": pa["req"], "executed": pa["exe"], "realized_sq": math.fsum(sorted(pa["real"]))}
    return {"errors": dict(errs), "flags": ok_flags, "executed_cells": executed, "autograd": auto, "EOS_severe": eos,
            "energy": energy, "max_rel_sq_err": max(sq) if sq else None, "max_chord_err": max(ch) if ch else None,
            "max_consumed_vs_proposed": max(cp) if cp else None,
            "arm_status": {f"{a}|{s}": n for (a, s), n in sorted(status.items())}}


def cmd_primary(args) -> dict:
    cfg, m, rows, bank, checks = base_checks()
    aseal, ac = load_seal("job_A_seal.json")
    pseal, pc = load_seal("pulse_seal.json")
    checks.update({f"A_{k}": v for k, v in ac.items()})
    checks.update({f"B_{k}": v for k, v in pc.items()})
    run = ROOT / RUN
    ids = [r["utterance_id"] for r in rows]
    cap, cons, pulses = rows_of(aseal, "capture", ids), rows_of(aseal, "construct", ids), rows_of(pseal, "pulses", ids)
    checks["all_rows_sealed"] = all(v is not None for v in pulses.values())
    a = json.loads((run / "audit_A.json").read_text())
    checks["audit_A_pass_pushed"] = a["verdict"] == "DIR_SPRINT0_AUDIT_A: PASS" and pushed(f"{RUN}/audit_A.json")
    auth = json.loads((run / "authorization_B.json").read_text())
    checks["authorization_minimal_and_true"] = list(auth) == cfg["firewall"]["authorization_B_fields"] and auth["authorized"] is True \
        and auth["job_A_seal_hash"] == file_hash(run / "job_A_seal.json") and pseal["authorization_sha256"] == file_hash(run / "authorization_B.json")
    rt = json.loads((run / "pulse_runtime.json").read_text())
    checks["runtime_completed"] = rt["status"] == "completed" and rt["manifest_hash"] == m["manifest_hash"]
    checks["weights_unchanged"] = rt["weights_unchanged"] is True
    checks["no_parameter_grads"] = rt["parameters_with_grad"] == 0 and rt["parameters_requiring_grad"] == 0
    op = opened_paths_check(rt["opened_paths"], rows, m, apparatus=True, provider=False)
    checks["opened_paths_allowlist"] = op["ok"]
    app = json.loads((run / "apparatus_old30.json").read_text())
    hq = cfg["historical_apparatus"]["queries"]
    checks["apparatus_old30"] = (app["ok"] is True and app["n"] == 30 and [(x["utterance_id"], x["t"]) for x in app["positions"]]
                                 == sorted([(x["utterance_id"], x["t"]) for x in hq], key=lambda y: (y[0], y[1]))
                                 and all(x["clean_logits_bitwise"] and x["clean_site_bitwise"] and x["pulse_logits_bitwise"] and x["zero_dose_bitwise"]
                                         and x["d2_max_abs_error"] == 0.0 and x["J_abs_diff"] <= 1e-6 and x["restore_bitwise"] for x in app["positions"])
                                 and not ({x["utterance_id"] for x in hq} & set(ids)))
    blocked = {f: bool(aseal["family_construction"][f]["blocked"]) for f in FAMILIES}
    T = Tok(cfg)
    rec = primary_recompute(cfg, T, rows, cap, cons, pulses, blocked, run)
    checks["independent_recomputation"] = not rec["errors"]
    checks["integrity_flags"] = all(rec["flags"].values())
    checks["autograd_count"] = rec["autograd"] == rt["autograd_calls"]
    d = cfg["dose"]
    checks["energy_bounds"] = all(v is None or v <= lim for v, lim in ((rec["max_rel_sq_err"], d["relative_squared_energy_error_max"]),
                                                                      (rec["max_chord_err"], d["relative_chord_error_max"]),
                                                                      (rec["max_consumed_vs_proposed"], d["consumed_vs_proposed_relative_error_max"])))
    mf = d["ungated_nonzero_requests_matched_fraction_min"]
    checks["D2_RND_energy_valid"] = all(rec["energy"][x]["matched_fraction"] is not None and rec["energy"][x]["matched_fraction"] >= mf for x in ("D2", "RND"))
    verdict = "DIR_SPRINT0_AUDIT_PRIMARY: PASS" if all(checks.values()) else "DIR_SPRINT0_AUDIT_PRIMARY: BLOCK"
    return {"schema": "dir_sprint0_audit_PRIMARY_v1", "verdict": verdict, "checks": checks, "recompute": rec, "opened_paths": op,
            "pulse_seal_sha256": file_hash(run / "pulse_seal.json"), "manifest_hash": m["manifest_hash"], "references_used": False,
            "git_head": git("rev-parse", "HEAD"), "created_unix": time.time()}


# ======================================================================================================
# audit FULL (references; independent mapping)
# ======================================================================================================

def own_gap(z: np.ndarray, Y: list[int], suppress: list[int]) -> float:
    x = np.asarray(z, dtype=np.float32).astype(np.float64)
    x[suppress] = -np.inf
    m = np.zeros(x.size, dtype=bool)
    m[Y] = True
    return float(x[~m].max() - x[m].max())


def full_core(cfg: dict, T: Tok, P: dict, cap: dict, cons: dict, pulses: dict, refs: dict, prim: dict, aseal: dict, fc: dict,
              prt: dict, app: dict, run: Path) -> dict:
    from experiments.inference_cf_srd2_g0_audit import own_map
    sel = P["selected"]
    ids = [r["utterance_id"] for r in sel]
    dlg = {r["utterance_id"]: r["dialogue_id"] for r in sel}
    dur = {r["utterance_id"]: float(r["source_duration_sec"]) for r in sel}
    dialogues = sorted(set(dlg.values()))
    per = {d: Counter() for d in dialogues}
    lat_t = lat_m = 0
    gaps = {}
    blocked = {f: bool(aseal["family_construction"][f]["blocked"]) for f in FAMILIES}
    cover = {f: Counter() for f in FAMILIES}
    for uid in ids:
        g, p = cap[uid], pulses[uid]
        qmap, lt, lm = own_map(T.tok, refs[uid], g["baseline"]["content_ids"], g["baseline"]["terminated"], dur[uid], EOS)
        lat_t += lt
        lat_m += lm
        byt = {x["t"]: (i, x) for i, x in enumerate(p["queries"])}
        st = {s["t"]: s for s in cons[uid]["status"]}
        en_struct = set()
        with np.load(run / "capture" / g["arrays"]["file"]) as z:
            cl = unbf16(z["cached_logits"])
        with np.load(run / "pulses" / p["logits"]["file"]) as z:
            keys, exl = z["exec_keys"], z["exec_logits"]
        exec_of = {(int(i), int(a)): k for k, (i, a) in enumerate(keys.tolist())}
        for t, (s, Y) in sorted(qmap.items()):
            i, x = byt[t]
            gq = g["queries"][i]
            b0 = x["arms"]["B0"]["top1"]
            d = per[dlg[uid]]
            d[{"EN-confusion": "n_C", "EN-correct": "n_EN", "ZH-correct": "n_ZH"}[s]] += 1
            for a in ARMS:
                if a == "D3CD":
                    top = gq["D3"]["c_AP"] if x["structural"] and gq["D3"]["status"] == "candidate" else b0
                    exe = bool(x["structural"] and gq["D3"]["status"] == "candidate")
                else:
                    top, exe = x["arms"][a]["top1"], x["arms"][a]["executed"]
                if s == "EN-confusion":
                    d[f"{a}_C"] += int(b0 not in Y and top in Y)
                elif s == "EN-correct":
                    d[f"{a}_H_EN"] += int(b0 in Y and top not in Y)
                else:
                    d[f"{a}_H_ZH"] += int(b0 in Y and top not in Y)
                    d[f"{a}_act_ZH"] += int(exe)
            if s == "EN-confusion" and x["structural"]:
                en_struct.add(t)
                g0 = own_gap(cl[i], Y, T.suppress)
                if math.isfinite(g0):
                    rec = {"B0": g0}
                    for a in PULSE_ARMS:
                        kk = exec_of.get((i, PULSE_ARMS.index(a)))
                        rec[a] = g0 if kk is None else own_gap(unbf16(exl[kk]), Y, T.suppress)
                    if all(math.isfinite(v) for v in rec.values()):
                        gaps[(uid, t)] = rec
        for x in p["queries"]:
            if not x["structural"]:
                continue
            s_ = st[x["t"]]
            okf = {"D3": (not blocked["D3"]) and (x["d3"]["status"] == "ok" or x["d3"]["status"] == "no_edit:baseline_is_candidate"),
                   "D4": (not blocked["D4"]) and s_["D4"]["status"] == "ok", "D5": (not blocked["D5"]) and s_["D5"]["status"] == "ok"}
            for f in FAMILIES:
                cover[f]["n"] += 1
                cover[f]["k"] += int(okf[f])
                if x["t"] in en_struct:
                    cover[f]["n_en"] += 1
                    cover[f]["k_en"] += int(okf[f])

    def S(k):
        return sum(per[d][k] for d in dialogues)
    nC, nEN, nZH = S("n_C"), S("n_EN"), S("n_ZH")
    A = {}
    for a in ARMS:
        C, HZ, HE, act = S(f"{a}_C"), S(f"{a}_H_ZH"), S(f"{a}_H_EN"), S(f"{a}_act_ZH")
        cd = [per[d][f"{a}_C"] for d in dialogues]
        A[a] = {"C": C, "H_ZH": HZ, "H_EN": HE, "U": C - HZ - HE, "active_ZH": act,
                "active_ZH_dialogues": sum(1 for d in dialogues if per[d][f"{a}_act_ZH"] > 0),
                "correction_dialogues": sum(1 for v in cd if v > 0), "max_share": max(cd) / C if C else None,
                "ZH_rate": HZ / nZH if nZH else None, "ZH_active_rate": HZ / act if act else None, "EN_rate": HE / nEN if nEN else None,
                "EOS": prim["EOS_severe"][a],
                "damage_dialogue": any(per[d][f"{a}_H_ZH"] >= 3 and per[d][f"{a}_act_ZH"] > 0 and per[d][f"{a}_H_ZH"] / per[d][f"{a}_act_ZH"] > .2
                                       for d in dialogues)}
    # bootstrap (own)
    st_ = cfg["statistics"]
    rng = np.random.default_rng(st_["seed"])
    draws = rng.integers(0, len(dialogues), size=(st_["bootstrap_replicates"], len(dialogues)))
    mc = cfg["matched_comparators"]

    def colsum(k):
        v = np.array([per[d][k] for d in dialogues], dtype=np.float64)
        return v[draws].sum(axis=1)
    lo, hi = st_["family_quantiles"]
    fam_iv = {}
    for v in ("D3", "D3G", "D4", "D4G", "D5", "D5G"):
        refs_ = [mc[v]["D2"], mc[v]["RND"]] + ([mc[v]["SH"]] if "SH" in mc[v] else [])
        for b in refs_:
            fam_iv[f"C_{v}-C_{b}"] = [float(np.quantile(colsum(f"{v}_C") - colsum(f"{b}_C"), q)) for q in (lo, hi)]
            fam_iv[f"U_{v}-U_{b}"] = [float(np.quantile((colsum(f"{v}_C") - colsum(f"{v}_H_ZH") - colsum(f"{v}_H_EN"))
                                                        - (colsum(f"{b}_C") - colsum(f"{b}_H_ZH") - colsum(f"{b}_H_EN")), q)) for q in (lo, hi)]
        if v in ("D3", "D3G"):
            fam_iv[f"U_{v}-U_D3CD"] = [float(np.quantile((colsum(f"{v}_C") - colsum(f"{v}_H_ZH") - colsum(f"{v}_H_EN"))
                                                         - (colsum("D3CD_C") - colsum("D3CD_H_ZH") - colsum("D3CD_H_EN")), q)) for q in (lo, hi)]
    mq = cfg["mechanistic"]["quantiles"]
    mech = {}
    for v in ("D3", "D3G", "D4", "D4G", "D5", "D5G"):
        rnd = mc[v]["RND"]
        pd_ = {d: [] for d in dialogues}
        for (uid, t), gg in gaps.items():
            pd_[dlg[uid]].append(gg[rnd] - gg[v])
        means = np.array([np.mean(pd_[d]) if pd_[d] else np.nan for d in dialogues])
        has = ~np.isnan(means)
        point = float(np.mean(means[has])) if has.any() else None
        sel_ = np.where(np.isnan(means), 0.0, means)[draws]
        cnt = has[draws].sum(axis=1)
        okb = cnt > 0
        bs = sel_.sum(axis=1)[okb] / cnt[okb]
        lohi = [float(np.quantile(bs, mq[0])), float(np.quantile(bs, mq[1]))] if okb.any() else None
        mech[v] = {"point": point, "bonferroni": lohi}
    # predicates
    o, pw, ad, se, sf, cvc = cfg["opportunities"], cfg["power"], cfg["advantage"], cfg["safety_estimability"], cfg["safety"], cfg["coverage"]
    nd = lambda k: sum(1 for d in dialogues if per[d][k] > 0)  # noqa: E731
    opp = (nC >= o["mapped_EN_confusion_min"] and nd("n_C") >= o["mapped_EN_confusion_dialogues_min"] and nEN >= o["mapped_EN_correct_min"]
           and nd("n_EN") >= o["mapped_EN_correct_dialogues_min"] and nZH >= o["mapped_ZH_correct_min"]
           and nd("n_ZH") >= o["mapped_ZH_correct_dialogues_min"] and lat_t > 0 and lat_m / lat_t >= o["English_unit_mapping_fraction_min"]
           and bool(aseal["predicates"]["job_A_coverage_pass"]))

    def gain_d(a, b):
        return sum(1 for d in dialogues if per[d][f"{a}_C"] - per[d][f"{b}_C"] > 0)

    def energy_ok(a):
        en = prim["energy"][a]
        if a in GATED_BASE:
            return en["lost_fraction"] is not None and en["lost_fraction"] <= cfg["dose"]["gated_planned_squared_energy_lost_max"]
        return en["matched_fraction"] is not None and en["matched_fraction"] >= cfg["dose"]["ungated_nonzero_requests_matched_fraction_min"]
    statuses = {}
    for f in FAMILIES:
        c_ = cover[f]
        fs = c_["k"] / c_["n"] if c_["n"] else None
        fe = c_["k_en"] / c_["n_en"] if c_["n_en"] else None
        if blocked[f]:
            statuses[f] = {"validity": "PROVIDER_OR_CONSTRUCTION_BLOCKED", "outcome": "NOT_TESTED", "variant": None, "mech": False}
            continue
        if fs is None or fe is None or fs < cvc["constructible_fraction_structural_min"] or fe < cvc["constructible_fraction_EN_confusion_min"]:
            statuses[f] = {"validity": "INSUFFICIENT_COVERAGE", "outcome": "NOT_TESTED", "variant": None, "mech": False}
            continue
        res = {}
        for v in VARIANTS[f]:
            a = A[v]
            m_ = mc[v]
            power = a["C"] >= pw["corrections_min"] and a["correction_dialogues"] >= pw["correction_dialogues_min"] \
                and a["max_share"] is not None and a["max_share"] <= pw["max_correction_dialogue_share"]
            adv = (a["C"] - A[m_["D2"]]["C"] >= ad["over_D2_corrections_min"] and a["C"] - A[m_["RND"]]["C"] >= ad["over_RND_corrections_min"]
                   and gain_d(v, m_["D2"]) >= ad["positive_paired_correction_gain_vs_D2_dialogues_min"])
            if f == "D5":
                adv = adv and a["C"] - A[m_["SH"]]["C"] >= ad["D5_over_shuffle_corrections_min"] \
                    and gain_d(v, m_["SH"]) >= ad["D5_positive_paired_gain_vs_shuffle_dialogues_min"]
            if f == "D3":
                adv = adv and a["U"] - A["D3CD"]["U"] >= ad["D3_utility_minus_D3CD_min"]
            est = a["active_ZH"] >= se["active_ZH_correct_min"] and a["active_ZH_dialogues"] >= se["active_ZH_correct_dialogues_min"]
            safe = (a["U"] >= sf["utility_min"] and a["ZH_rate"] is not None and a["ZH_rate"] <= sf["unconditional_ZH_corruption_rate_max"]
                    and a["ZH_active_rate"] is not None and a["ZH_active_rate"] <= sf["active_ZH_corruption_rate_max"]
                    and a["H_EN"] <= sf["EN_corruptions_max"] and a["EN_rate"] is not None and a["EN_rate"] <= sf["unconditional_EN_corruption_rate_max"]
                    and a["EOS"] <= sf["EOS_severe_events_max"] and not a["damage_dialogue"])
            mk = mech[v]
            mechp = mk["point"] is not None and mk["point"] >= cfg["mechanistic"]["point_min_nats"] and mk["bonferroni"] is not None \
                and mk["bonferroni"][0] > 0
            res[v] = {"energy": energy_ok(v), "power": power, "advantage": adv, "estimability": est, "safety": safe, "mechanistic": mechp}
        order = VARIANTS[f]
        core = {v: res[v]["energy"] and res[v]["power"] and res[v]["advantage"] for v in order}
        mech_any = any(res[v]["energy"] and res[v]["mechanistic"] for v in order)
        out = {"validity": "VALID_AND_TESTED", "outcome": "LEXICAL_POWER_INSUFFICIENT", "variant": None, "mech": mech_any, "leaves": res}
        for v in order:
            if core[v] and res[v]["estimability"] and res[v]["safety"]:
                out.update(outcome="PROMISING_FEASIBILITY", variant=v)
                break
        else:
            for v in order:
                if core[v] and res[v]["estimability"]:
                    out.update(outcome="SAFETY_FAILED", variant=v)
                    break
            else:
                for v in order:
                    if core[v]:
                        out.update(outcome="INSUFFICIENT_COVERAGE", variant=v)
                        break
        statuses[f] = out
    crit = (all(aseal["critical"].values()) and aseal["predicates"]["compatibility_pass"] and prt.get("status") == "completed"
            and prt.get("weights_unchanged") is True and prt.get("parameters_with_grad") == 0 and app.get("ok") is True
            and not prim["errors"] and all(prim["flags"].values()) and energy_ok("D2") and energy_ok("RND")
            and all(pulses[u]["status"] == "ok" for u in ids))
    if not crit or all(s["validity"] == "PROVIDER_OR_CONSTRUCTION_BLOCKED" for s in statuses.values()):
        label = "DIR_SPRINT0_INVALID"
    elif not fc.get("resource_pass"):
        label = "DIR_SPRINT0_COMPUTE_BLOCKED"
    elif not opp or not any(s["validity"] == "VALID_AND_TESTED" for s in statuses.values()):
        label = "DIR_SPRINT0_OPPORTUNITY_INSUFFICIENT"
    elif any(s["outcome"] == "PROMISING_FEASIBILITY" for s in statuses.values()):
        label = "DIR_SPRINT0_STEERING_FEASIBILITY_SIGNAL"
    elif any(s["outcome"] == "SAFETY_FAILED" for s in statuses.values()):
        label = "DIR_SPRINT0_CAUSAL_POWER_WITH_DAMAGE"
    elif any(s["validity"] == "VALID_AND_TESTED" and s["mech"] for s in statuses.values()):
        label = "DIR_SPRINT0_MECHANISTIC_EFFECT_ONLY"
    else:
        label = "DIR_SPRINT0_ALL_DIRECTIONS_INEFFECTIVE"
    return {"A": A, "per": {d: dict(c) for d, c in per.items()}, "opportunity_pass": opp, "n": {"C": nC, "EN": nEN, "ZH": nZH},
            "latin": [lat_t, lat_m], "coverage": {f: dict(c) for f, c in cover.items()}, "statuses": statuses, "family_iv": fam_iv,
            "mechanistic": mech, "critical": crit, "label": label}


def cmd_full(args) -> dict:
    import pyarrow.dataset as ds
    cfg, m, rows, bank, checks = base_checks()
    aseal, ac = load_seal("job_A_seal.json")
    pseal, pc = load_seal("pulse_seal.json")
    checks.update({f"A_{k}": v for k, v in ac.items()})
    checks.update({f"B_{k}": v for k, v in pc.items()})
    run = ROOT / RUN
    prim = json.loads((run / "audit_PRIMARY.json").read_text())
    checks["primary_pass_pushed"] = prim["verdict"] == "DIR_SPRINT0_AUDIT_PRIMARY: PASS" and pushed(f"{RUN}/audit_PRIMARY.json")
    ev = json.loads((run / "evaluation.json").read_text())
    checks["evaluation_after_primary"] = ev["audit_PRIMARY_sha256"] == file_hash(run / "audit_PRIMARY.json") \
        and ev["pulse_seal_sha256"] == file_hash(run / "pulse_seal.json")
    P = population()
    ids = [r["utterance_id"] for r in P["selected"]]
    if file_hash(P["source"]) != P["source_sha256"]:
        raise SystemExit("role parquet changed")
    tab = ds.dataset(P["source"], format="parquet").to_table(columns=["utterance_id", "role", "transcript_raw"],
                                                            filter=ds.field("utterance_id").isin(ids)).to_pylist()
    refs = {r["utterance_id"]: r["transcript_raw"] for r in tab}
    checks["references_exact_selection"] = sorted(refs) == sorted(ids) and all(r["role"] == "D-dev-select" for r in tab)
    cap, cons, pulses = rows_of(aseal, "capture", ids), rows_of(aseal, "construct", ids), rows_of(pseal, "pulses", ids)
    T = Tok(cfg)
    blocked = {f: bool(aseal["family_construction"][f]["blocked"]) for f in FAMILIES}
    prim_rec = primary_recompute(cfg, T, P["selected"], cap, cons, pulses, blocked, run)
    fc = json.loads((run / "resource_forecast.json").read_text())
    prt = json.loads((run / "pulse_runtime.json").read_text())
    app = json.loads((run / "apparatus_old30.json").read_text())
    core = full_core(cfg, T, P, cap, cons, pulses, refs, prim_rec, aseal, fc, prt, app, run)
    am = ev["arm_metrics"]
    agree = {}
    for a in ARMS:
        agree[a] = all(core["A"][a][k] == am[a][k2] for k, k2 in (("C", "C"), ("H_ZH", "H_ZH"), ("H_EN", "H_EN"), ("U", "U"),
                                                                    ("active_ZH", "active_ZH"), ("correction_dialogues", "correction_dialogues"),
                                                                    ("EOS", "EOS_severe"), ("damage_dialogue", "damage_dialogue")))
    checks["arm_counts_agree"] = all(agree.values())
    checks["per_dialogue_agree"] = all(core["per"][d].get(k, 0) == v for d, row in ev["per_dialogue"].items() for k, v in row.items())
    checks["opportunity_agree"] = core["n"] == {"C": ev["opportunities"]["mapped_EN_confusion"], "EN": ev["opportunities"]["mapped_EN_correct"],
                                                "ZH": ev["opportunities"]["mapped_ZH_correct"]}
    checks["family_status_agree"] = all(core["statuses"][f]["validity"] == ev["family_status"][f]["validity"]
                                        and core["statuses"][f]["outcome"] == ev["family_status"][f]["outcome"] for f in FAMILIES)
    checks["bootstrap_family_agree"] = all(close(core["family_iv"][k][0], ev["bootstrap"]["family"][k]["bonferroni"][0], 1e-9)
                                           and close(core["family_iv"][k][1], ev["bootstrap"]["family"][k]["bonferroni"][1], 1e-9)
                                           for k in core["family_iv"])
    checks["mechanistic_agree"] = all(close(core["mechanistic"][v]["point"], ev["bootstrap"]["mechanistic"][v]["point"], 1e-9)
                                      for v in core["mechanistic"])
    checks["label_agree"] = core["label"] == ev["provisional_terminal_label"]
    verdict = "DIR_SPRINT0_AUDIT_FULL: PASS" if all(checks.values()) else "DIR_SPRINT0_AUDIT_FULL: FAIL"
    final_label = core["label"] if verdict.endswith("PASS") else "DIR_SPRINT0_INVALID"
    return {"schema": "dir_sprint0_audit_FULL_v1", "verdict": verdict, "checks": checks, "independent": {k: v for k, v in core.items() if k != "per"},
            "arm_agreement": agree, "terminal_label": final_label, "evaluation_sha256": file_hash(run / "evaluation.json"),
            "manifest_hash": m["manifest_hash"], "git_head": git("rev-parse", "HEAD"), "created_unix": time.time()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("pre", "a", "primary", "full", "concepts-check"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.stage == "concepts-check":
        cmd_concepts_check(args)
        return
    res = {"pre": cmd_pre, "a": cmd_a, "primary": cmd_primary, "full": cmd_full}[args.stage](args)
    name = {"pre": "audit_PRE.json", "a": "audit_A.json", "primary": "audit_PRIMARY.json", "full": "audit_FULL.json"}[args.stage]
    out = Path(args.out) if args.out else ROOT / RUN / name
    if out.exists():
        raise FileExistsError(f"{out} exists; failed attempts are preserved, never overwritten")
    atomic_json(out, res)
    print(json.dumps({"verdict": res["verdict"], "failed": [k for k, v in res["checks"].items() if not v]}, indent=1))


if __name__ == "__main__":
    main()
