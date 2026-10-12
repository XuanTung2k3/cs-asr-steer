#!/usr/bin/env python
"""INTERNAL-0A PRIMARY audit: independent reconstruction of the sealed capture and selection.

Deliberately does NOT import csasr.inference_cf.internal0a or the runner's selection code.
Grid, physical views, association, occurrence grouping, support, ranking and the shuffled
control are re-derived from the frozen design/config text. Only frozen historical ABTS
primitives (token bytes, content units, DTW timing, archive unpacking) are shared.
Never opens references, the evaluation panel or retrospective labels.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
from csasr.inference_cf.core import atomic_json, digest, file_hash

BASE = ROOT / "results/research_review/internal0a"
FORBIDDEN = ("role_D-", "evaluation_panel", "evaluation_units", "candidates_existing_ctc", "abts0_k1f/", "/references",
             "adjudication")
OPENED = set()
FAIL: list[str] = []


def audit_open(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        if any(x in p for x in FORBIDDEN):
            raise RuntimeError(f"audit firewall: {p}")
        OPENED.add(p)


def check(ok, msg):
    if not ok:
        FAIL.append(msg)
    return ok


# ---- independent re-derivations ---------------------------------------------------------------------------------------

def grid(n, sr=16000, cap=480000):
    n = min(n, cap)
    starts = [0] if n <= 2 * sr else list(range(0, n, sr))
    out = []
    for s in starts:
        e = min(n, s + 2 * sr)
        c = (s + e) // 2
        out.append(((s, e), (max(0, c - 2 * sr), min(n, c + 2 * sr))))
    return out


def views(n):
    v = {}
    for k, (short, long) in enumerate(grid(n)):
        for br in ("2_EN", "2_AUTO", "4_AUTO", "4_ZH"):
            b = short if br[0] == "2" else long
            v.setdefault((br, *b), []).append(k)
    return v


def ov(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def assoc(iv, units):
    if iv is None or iv[1] <= iv[0]:
        return None, "UNLOCATED"
    idx = [k for k, u in enumerate(units) if u.get("interval") and ov(iv, u["interval"]) > 0]
    if not idx:
        return None, "UNLOCATED"
    if idx[-1] - idx[0] + 1 != len(idx):
        return None, "NONCONTIGUOUS"
    # union length of clipped covers
    segs = sorted((max(units[k]["interval"][0], iv[0]), min(units[k]["interval"][1], iv[1])) for k in idx)
    tot, cur = 0, None
    for a, b in segs:
        if cur is None or a > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    tot += cur[1] - cur[0]
    if tot / (iv[1] - iv[0]) < 0.5:
        return None, "LOW_COVER"
    return (idx[0], idx[-1]), "ASSOCIATED"


def extract(cap, tb, timing, units, content_units, unit_interval):
    s, e = cap["bounds"]
    out = []
    cu = content_units(tb)
    for u in cu["units"]:
        if u.kind != "LATIN":
            continue
        iv = unit_interval(u, timing)
        full = None if iv is None else (s + iv[0], s + iv[1])
        blk, st = assoc(full, units)
        out.append({"w": u.surface.casefold().replace("’", "'"), "surface": u.surface, "view": cap["view_id"],
                    "branch": cap["branch"], "crop": (s, e), "iv": full, "t": u.tok_first,
                    "edge": full is None or full[0] < s + 3200 or full[1] > e - 3200,
                    "cap": cap["terminated"] == "cap", "utf8": all(cu["chars_valid"][u.char_start:u.char_end]),
                    "blk": blk, "st": st})
    return out


def select(words, units):
    order = sorted(words, key=lambda w: (w["branch"] == "4_ZH", w["w"], w["iv"] or (-1, -1), w["view"], w["t"]))
    groups = []
    for w in order:
        dest = None
        if w["iv"] is not None:
            for g in groups:
                if g[0]["w"] != w["w"]:
                    continue
                if all(v["iv"] is not None and min(v["iv"][1] - v["iv"][0], w["iv"][1] - w["iv"][0]) > 0 and
                       ov(v["iv"], w["iv"]) >= 0.5 * min(v["iv"][1] - v["iv"][0], w["iv"][1] - w["iv"][0]) for v in g):
                    dest = g
                    break
        if dest is None:
            groups.append([w])
        else:
            dest.append(w)
    cands = []
    for cid, g in enumerate(groups):
        prim = [w for w in g if w["branch"] != "4_ZH"]
        src = prim or g
        blocks = {w["blk"] for w in src if w["blk"] is not None}
        vids = [w["view"] for w in src]
        ok = [w for w in prim if not w["edge"] and not w["cap"] and w["utf8"] and w["st"] == "ASSOCIATED"]
        sup = any(a["crop"] != b["crop"] for a in ok if a["branch"] in ("2_EN", "2_AUTO") for b in ok if b["branch"] == "4_AUTO")
        blk = next(iter(blocks)) if len(blocks) == 1 else None
        present = blk is not None and any(units[j]["surface"].casefold().replace("’", "'") == g[0]["w"]
                                          for j in range(blk[0], blk[1] + 1))
        ivs = [w["iv"] for w in g if w["iv"] is not None]
        cands.append({"id": cid, "w": g[0]["w"], "blk": blk, "sup": sup, "n_br": len({w["branch"] for w in prim}),
                      "rec": len(set(vids)), "start": min(i[0] for i in ivs) if ivs else None,
                      "elig": sup and len(blocks) == 1 and len(vids) == len(set(vids)) and not present,
                      "members": sorted((w["view"], w["t"]) for w in g)})
    cnt = {}
    for c in cands:
        if c["blk"] is not None:
            cnt[(c["w"], c["blk"])] = cnt.get((c["w"], c["blk"]), 0) + 1
    for c in cands:
        if c["blk"] is not None and cnt[(c["w"], c["blk"])] > 1:
            c["elig"] = False
    el = sorted((c for c in cands if c["elig"]), key=lambda c: (-c["n_br"], -c["rec"], c["start"], c["w"], c["id"]))
    return cands, (el[0]["id"] if el else None)


def shuffle(rows, seed):
    by = {}
    for r in rows:
        by.setdefault(r["dialogue_id"], []).append(r)
    out = {}
    for g in by.values():
        g = sorted(g, key=lambda r: hashlib.sha256(f"{seed}:{r['utterance_id']}".encode()).hexdigest())
        if len(g) == 1:
            out[g[0]["utterance_id"]] = g[0]["utterance_id"]
            continue
        best = min(range(1, len(g)), key=lambda k: (sum(abs(g[i]["heard_samples"] - g[(i + k) % len(g)]["heard_samples"])
                                                        for i in range(len(g))), k))
        for i, r in enumerate(g):
            out[r["utterance_id"]] = g[(i + best) % len(g)]["utterance_id"]
    return out


# ---- audit ----------------------------------------------------------------------------------------------------------

def main():
    sys.addaudithook(audit_open)
    from csasr.inference_cf.abts0_k1 import content_units, unit_interval, token_intervals
    from csasr.inference_cf.core_r2 import token_bytes
    from experiments.acoustic_s1 import unpack
    from transformers import WhisperTokenizer
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
    import soundfile as sf  # noqa: F401  (load_audio below)
    from csasr.models.whisper import load_audio

    cfg = json.loads((ROOT / "configs/research_review/internal0a.json").read_text())
    m = json.loads((BASE / "runtime_manifest.json").read_text())
    check(digest({k: v for k, v in m.items() if k != "manifest_hash"}) == m["manifest_hash"], "manifest hash")
    check(file_hash(ROOT / "configs/research_review/internal0a.json") == m["config_sha256"], "config hash")
    for p, h in m["sources"].items():
        check(file_hash(ROOT / p) == h, f"source {p}")
    check(m["membership_hash"] == digest([r["utterance_id"] for r in m["rows"]]), "membership")
    check(len(m["rows"]) == 45 and len({r["dialogue_id"] for r in m["rows"]}) == 19, "45/19 population")

    seal = json.loads((BASE / "output_seal.json").read_text())
    check(digest({k: v for k, v in seal.items() if k != "seal_hash"}) == seal["seal_hash"], "seal hash")
    check(seal["manifest_hash"] == m["manifest_hash"], "seal manifest")
    for p, h in {**seal["files"], **seal["archives"]}.items():
        check(file_hash(p if Path(p).is_absolute() else ROOT / p) == h, f"sealed file {p}")

    eng = json.loads((BASE / "capture_engineering.json").read_text())
    check(eng["manifest_hash"] == m["manifest_hash"], "engineering manifest")
    check(eng["sources"] == m["sources"] and eng["model_files"] == m["model_files"], "engineering pins")
    check(len(eng["global_identity"]) == 45 and all(g["passed"] for g in eng["global_identity"]), "global B0 identity x45")
    check(eng["weights_unchanged_probe"] and eng["no_parameter_gradients"], "frozen weights")
    check(not any(any(x in p for x in FORBIDDEN) for p in eng["opened_paths"]), "capture firewall")
    check(eng["gpu"].startswith("NVIDIA H100") and "MIG" in eng["gpu"], f"gpu {eng['gpu']}")

    cand = json.loads((BASE / "run1/candidates.json").read_text())
    check(cand["manifest_hash"] == m["manifest_hash"] and cand["references_used"] is False, "candidates provenance")
    check(not any(any(x in p for x in FORBIDDEN) for p in cand["opened_paths"]), "selection firewall")
    blob = json.dumps(cand["rows"])
    for k in ("transcript", "\"reference\"", "ref_index", "present_anywhere", "is_control", "onset"):
        check(k not in blob, f"reference-like field {k} in candidates")

    tok = WhisperTokenizer.from_pretrained(m["model_dir"], local_files_only=True)
    bd = {v: k for k, v in bytes_to_unicode().items()}
    eng_cfg = cfg["engine"]
    stats = {"views": 0, "capped": 0, "auto_lang": {}, "encoder_groups": 0, "heads_bytes": 0}
    recon = {}
    mismatch_rows = []
    for r in m["rows"]:
        uid = r["utterance_id"]
        x = load_audio(r["audio_path"], 16000)
        check("sha256:" + hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() == r["waveform_sha256"], f"waveform {uid}")
        x = x[:r["heard_samples"]]
        check(r["heard_samples"] == min(r["n_samples"], 480000), f"heard horizon {uid}")
        V = views(len(x))
        d = BASE / "run1/capture" / f"{r['index']:03d}"
        found = sorted(p.name for p in d.glob("*.json"))
        want = sorted(f"{b}:{s}:{e}.json" for b, s, e in V)
        check(found == want, f"matrix {uid}: {len(found)} vs {len(want)}")
        enc_by_bounds = {}
        words = []
        for (br, s, e), gi in V.items():
            p = d / f"{br}:{s}:{e}.json"
            if not p.exists():
                continue
            c = json.loads(p.read_text())
            stats["views"] += 1
            check(c["manifest_hash"] == m["manifest_hash"] and c["utterance_id"] == uid, f"view identity {p.name}")
            check(c["branch"] == br and c["bounds"] == [s, e] and c["grid_indices"] == gi, f"view bounds {p.name}")
            check(c["waveform_sha256"] == "sha256:" + hashlib.sha256(np.ascontiguousarray(x[s:e]).tobytes()).hexdigest(),
                  f"crop waveform {uid} {p.name}")
            pr = c["prompt"]
            if br == "2_EN":
                check(pr == eng_cfg["cE"], f"EN prompt {p.name}")
            elif br == "4_ZH":
                check(pr == eng_cfg["cB"], f"ZH prompt {p.name}")
            else:
                check(len(pr) == 4 and pr[0] == 50258 and pr[2:] == [50360, 50364] and 50259 <= pr[1] < 50358, f"AUTO prompt {p.name}")
                lang = tok.convert_ids_to_tokens(pr[1])
                stats["auto_lang"].setdefault(br, {}).setdefault(lang, 0)
                stats["auto_lang"][br][lang] += 1
            n = len(c["tokens"])
            check(c["terminated"] in ("eos", "cap"), f"termination {p.name}")
            check(n <= eng_cfg["max_new_tokens_local"] and (c["terminated"] != "cap" or n == eng_cfg["max_new_tokens_local"]),
                  f"cap semantics {p.name}")
            check(len(c["logits_sha256"]) == n + (c["terminated"] == "eos"), f"query count {p.name}")
            check(eng_cfg["eos"] not in c["tokens"], f"eos inside tokens {p.name}")
            stats["capped"] += c["terminated"] == "cap"
            key = (s, e)
            if key in enc_by_bounds:
                check(enc_by_bounds[key] == c["encoder_lineage"], f"encoder reuse lineage {p.name}")
            else:
                enc_by_bounds[key] = c["encoder_lineage"]
            check(file_hash(c["heads"]["path"]) == c["heads"]["sha256"], f"heads archive {p.name}")
            stats["heads_bytes"] += os.path.getsize(c["heads"]["path"])
            with np.load(c["heads"]["path"]) as z:
                h = unpack(z["heads"])
            F = -(-(e - s) // 320)
            check(h.shape == (len(c["logits_sha256"]), 10, F), f"heads shape {p.name} {h.shape}")
            timing = token_intervals(h, n, c["terminated"], e - s)
            words += extract(c, [token_bytes(tok, i, bd) for i in c["tokens"]], timing, r["baseline_units"],
                             content_units, unit_interval)
        stats["encoder_groups"] += len(enc_by_bounds)
        recon[uid] = words
    # selection reconstruction
    rows = {x["utterance_id"]: x for x in cand["rows"]}
    check([x["utterance_id"] for x in cand["rows"]] == [r["utterance_id"] for r in m["rows"]], "candidate row order")
    smap = shuffle(m["rows"], cfg["seed"])
    check(smap == m["shuffle_map"], "shuffle map")
    meta = {r["utterance_id"]: r for r in m["rows"]}
    n_sel = n_ssel = 0
    for uid, words in recon.items():
        got = rows[uid]
        check(len(got["words"]) == len(words), f"word count {uid}")
        gw = sorted((w["view_id"], w["tok_first"], w["normalized"], tuple(w["interval"]) if w["interval"] else None,
                     tuple(w["block"]) if w["block"] else None, w["association"], w["boundary_truncation_possible"])
                    for w in got["words"])
        mw = sorted((w["view"], w["t"], w["w"], w["iv"], w["blk"], w["st"], w["edge"]) for w in words)
        check(gw == mw, f"word records {uid}")
        cands, sel = select(words, meta[uid]["baseline_units"])
        mine = sorted((c["w"], tuple(c["members"]), c["sup"], c["elig"]) for c in cands)
        theirs = sorted((c["normalized"], tuple(sorted((w["view_id"], w["tok_first"]) for w in c["members"])),
                         c["cross_context_supported"], c["selection_eligible"]) for c in got["candidates"])
        check(mine == theirs, f"candidate groups {uid}")
        msel = None if sel is None else next(c for c in cands if c["id"] == sel)
        gsel = None if got["selected"] is None else next(c for c in got["candidates"] if c["candidate_id"] == got["selected"])
        same = (msel is None and gsel is None) or (msel is not None and gsel is not None and msel["w"] == gsel["normalized"]
                and tuple(msel["members"]) == tuple(sorted((w["view_id"], w["tok_first"]) for w in gsel["members"])))
        if not check(same, f"selection {uid}"):
            mismatch_rows.append(uid)
        n_sel += sel is not None
        # shuffled control: donor words rescaled, re-associated, re-selected
        donor = smap[uid]
        check(got["shuffled"]["donor"] == donor and got["shuffled"]["informative"] == (donor != uid), f"shuffle donor {uid}")
        sc = meta[uid]["heard_samples"] / meta[donor]["heard_samples"]
        sw = []
        for w in recon[donor]:
            iv = None if w["iv"] is None else (round(w["iv"][0] * sc), round(w["iv"][1] * sc))
            cr = (round(w["crop"][0] * sc), round(w["crop"][1] * sc))
            blk, st = assoc(iv, meta[uid]["baseline_units"])
            sw.append(dict(w, iv=iv, crop=cr, blk=blk, st=st,
                           edge=iv is None or iv[0] < cr[0] + 3200 or iv[1] > cr[1] - 3200))
        sc_c, ssel = select(sw, meta[uid]["baseline_units"])
        gs = got["shuffled"]
        gss = None if gs["selected"] is None else next(c for c in gs["candidates"] if c["candidate_id"] == gs["selected"])
        mss = None if ssel is None else next(c for c in sc_c if c["id"] == ssel)
        check((gss is None) == (mss is None) and (gss is None or gss["normalized"] == mss["w"]), f"shuffled selection {uid}")
        n_ssel += ssel is not None
    expected_views = sum(len(views(min(r["n_samples"], 480000))) for r in m["rows"])
    out = {"schema": "internal0a_primary_audit_v1", "status": "PASS" if not FAIL else "FAIL",
           "independent": True,
           "independence_note": "independent re-implementation of grid/views/association/grouping/support/ranking/shuffle; "
                                "does not import csasr.inference_cf.internal0a or experiments/internal0a*.py; same agent session",
           "seal_hash": seal["seal_hash"], "manifest_hash": m["manifest_hash"], "job_id": eng["job_id"],
           "failures": FAIL, "expected_views": expected_views, "stats": stats,
           "selected_reconstructed": n_sel, "shuffled_selected_reconstructed": n_ssel,
           "selection_mismatch_rows": mismatch_rows, "references_opened": False,
           "opened_forbidden": [p for p in OPENED if any(x in p for x in FORBIDDEN)],
           "auditor_sha256": file_hash(Path(__file__))}
    atomic_json(BASE / "primary_audit.json", out)
    print(json.dumps({k: out[k] for k in ("status", "failures", "expected_views", "stats", "selected_reconstructed",
                                          "shuffled_selected_reconstructed")}, default=str)[:4000])


if __name__ == "__main__":
    main()
