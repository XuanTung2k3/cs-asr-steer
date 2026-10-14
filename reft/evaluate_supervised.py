#!/usr/bin/env python
"""P2 evaluation engine (CPU): P0 scorer conventions (canonical units, ACL-0 alignment, reference-unit strata).

  score-baseline        -- score the P2-0 two-stage frozen-LID baseline decode of dev150 (must reproduce P0 metrics)
  select --run DIR      -- per-epoch dev metrics of a P2-1 run; frozen rule: lowest genuine-CS MER (beam-5), then overall
                           MER, then Mandarin CER, then earliest epoch
  evaluate --runs DIRS  -- selected-epoch comparison against the frozen baseline (beam-5 primary, greedy secondary)

Dev references (exposed D-dev-select panel, already opened in P0) are read only here, never by GPU decode jobs.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import art, fillers, read_json, read_jsonl, write_json  # noqa: E402
from reft.provenance import git_state, now, sha  # noqa: E402
from reft.text import compare, en_error_kind, loops, one, ratios, utterance_stratum  # noqa: E402

P21 = ROOT / "configs/reft/p2_1_frozen.yaml"
STRATA = ("CS", "EN_ONLY", "ZH_ONLY", "OTHER")


def p21() -> dict:
    return yaml.safe_load(P21.read_text())


def dev_rows():
    return read_json(art("p2_0/dev_manifest.json"))["rows"]


def dev_refs(ids):
    from reft.baseline_auto import dev_refs as _d
    return _d(ids)


# ====================================================================== scoring

def score_utterance(ref: str, hyp: str, FILL: set, loop_units: int) -> dict:
    o = one(ref, hyp)
    sub = [k for k in range(len(o["ref"])) if o["tags"][k] == "EN" and o["ref"][k] not in FILL]
    zh = [k for k in range(len(o["ref"])) if o["tags"][k] == "ZH" and o["ref"][k] not in FILL]
    return {"o": o, "counts": o["counts"], "sub_en_ok": [o["by"][k][0] == "match" for k in sub],
            "zh_ok": [o["by"][k][0] == "match" for k in zh],
            "en_kinds": [en_error_kind(o, k) for k in sub if o["by"][k][0] != "match"],
            "loop": loops(o["hyp"]) >= loop_units}


def corpus(rows: list[dict]) -> dict:
    c = np.array([r["counts"] for r in rows]).sum(0) if rows else np.zeros(11, int)
    return {**ratios(c), "mixed_errors": int(c[2]), "ref_units": int(c[3]), "n_utterances": len(rows),
            "substantive_en_errors": sum(len(r["sub_en_ok"]) - sum(r["sub_en_ok"]) for r in rows),
            "zh_non_filler_errors": sum(len(r["zh_ok"]) - sum(r["zh_ok"]) for r in rows),
            "en_error_kinds": dict(collections.Counter(k for r in rows for k in r["en_kinds"])),
            "loop_utterances": sum(r["loop"] for r in rows), "not_terminated": sum(not r["terminated"] for r in rows)}


def score_system(meta: list[dict], refs: dict, recs: dict, FILL: set, loop_units: int) -> list[dict]:
    out = []
    for m in meta:
        i = m["utterance_id"]
        s = score_utterance(refs[i], recs[i]["text"], FILL, loop_units)
        out.append({"id": i, "dialogue": m["dialogue_id"], **utterance_stratum(refs[i], FILL), **s,
                    "terminated": bool(recs[i]["terminated"])})
    return out


def metrics(scored: list[dict]) -> dict:
    return {"overall": corpus(scored), **{s: corpus([r for r in scored if r["stratum"] == s]) for s in STRATA}}


def paired(base: list[dict], sysr: list[dict], reps: int, seed: int) -> dict:
    """Changes of a system against the frozen baseline (same utterance order)."""
    corr = corrupt = new_zh = zh_rep = zh_ok = out_dmg = out_den = 0
    d_corr, d_corrupt = set(), set()
    net = {}
    for b, s in zip(base, sysr):
        assert b["id"] == s["id"]
        c = sum(1 for x, y in zip(b["sub_en_ok"], s["sub_en_ok"]) if not x and y)
        k = sum(1 for x, y in zip(b["sub_en_ok"], s["sub_en_ok"]) if x and not y)
        corr, corrupt = corr + c, corrupt + k
        if c:
            d_corr.add(b["dialogue"])
        if k:
            d_corrupt.add(b["dialogue"])
        net[b["id"]] = c - k
        new_zh += sum(1 for x, y in zip(b["zh_ok"], s["zh_ok"]) if x and not y)
        zh_rep += sum(1 for x, y in zip(b["zh_ok"], s["zh_ok"]) if not x and y)
        zh_ok += sum(b["zh_ok"])
        cmp_ = compare(b["o"], s["o"], b["id"], b["dialogue"], False)
        out_dmg, out_den = out_dmg + cmp_["outside_damage"], out_den + cmp_["outside_den"]
    rng = np.random.default_rng(seed)
    dl = sorted({r["dialogue"] for r in base})
    per = {d: {k: (np.array([r["counts"] for r in rr if r["dialogue"] == d]).sum(0)) for k, rr in (("b", base), ("s", sysr))} for d in dl}
    perc = {d: {k: (np.array([r["counts"] for r in rr if r["dialogue"] == d and r["stratum"] == "CS"] or [np.zeros(11, int)]).sum(0))
                for k, rr in (("b", base), ("s", sysr))} for d in dl}
    dm, dmc = [], []
    for _ in range(reps):
        pick = rng.choice(dl, len(dl))
        tb, ts = sum(per[d]["b"] for d in pick), sum(per[d]["s"] for d in pick)
        dm.append(ts[2] / ts[3] - tb[2] / tb[3])
        cb, cs = sum(perc[d]["b"] for d in pick), sum(perc[d]["s"] for d in pick)
        if cb[3] and cs[3]:
            dmc.append(cs[2] / cs[3] - cb[2] / cb[3])
    mb, ms = metrics(base), metrics(sysr)
    tot = corr - corrupt
    return {"substantive_en_corrections": corr, "substantive_en_corruptions": corrupt, "net": tot,
            "dialogues_with_corrections": len(d_corr), "dialogues_with_corruptions": len(d_corrupt),
            "new_zh_errors": new_zh, "zh_repairs": zh_rep, "mandarin_retention": 1 - new_zh / max(1, zh_ok),
            "outside_target_damage": out_dmg, "outside_target_denominator": out_den,
            "improved_utterances": sum(s["counts"][2] < b["counts"][2] for b, s in zip(base, sysr)),
            "worsened_utterances": sum(s["counts"][2] > b["counts"][2] for b, s in zip(base, sysr)),
            "max_utterance_share_of_net": (max(net.values()) / tot) if tot > 0 else None,
            "delta_mer": {"point": ms["overall"]["mer"] - mb["overall"]["mer"], "ci95": [float(np.quantile(dm, q)) for q in (0.025, 0.975)]},
            "delta_cs_mer": {"point": (ms["CS"]["mer"] or 0) - (mb["CS"]["mer"] or 0),
                             "ci95": [float(np.quantile(dmc, q)) for q in (0.025, 0.975)] if dmc else None}}


def select_epoch(records: list[dict]) -> dict:
    """records: [{epoch, cs_mer, mer, zh_cer}] (beam-5). Lowest CS MER, then MER, then Mandarin CER, then earliest."""
    best = min(records, key=lambda r: (r["cs_mer"], r["mer"], r["zh_cer"], r["epoch"]))
    return {"epoch": best["epoch"]}


# ====================================================================== commands

def _strip(scored):
    return [{k: v for k, v in r.items() if k != "o"} for r in scored]


def run_score_baseline():
    C = p21()
    meta = dev_rows()
    refs = dev_refs([m["utterance_id"] for m in meta])
    FILL = fillers()
    out = {}
    for regime in ("beam5", "greedy"):
        p = art(f"p2_0/frozen_baseline_dev150_{regime}.jsonl")
        if not p.exists():
            continue
        recs = {r["id"]: r for r in read_jsonl(p)}
        if len(recs) != len(meta):
            out[regime] = {"n": len(recs), "note": "subset only; not scored as a corpus"}
            continue
        sc = score_system(meta, refs, recs, FILL, C["evaluation"]["loop_units"])
        out[regime] = {"metrics": metrics(sc), "sha256": sha(p)}
    p0 = read_json(art("p0/evaluation.json"))["metrics"]
    rep = {"p0_AUTO_overall_mer": p0["overall"]["AUTO"]["mer"], "p0_AUTO_cs_mer": p0["CS"]["AUTO"]["mer"]}
    if "metrics" in out.get("beam5", {}):
        m = out["beam5"]["metrics"]
        rep.update(p2_overall_mer=m["overall"]["mer"], p2_cs_mer=m["CS"]["mer"],
                   equal_to_p0=abs(m["overall"]["mer"] - rep["p0_AUTO_overall_mer"]) < 1e-12 and abs(m["CS"]["mer"] - rep["p0_AUTO_cs_mer"]) < 1e-12)
    write_json(art("p2_0/frozen_baseline_metrics.json"), {"schema": "reft_p2_0_baseline_metrics_v1", "utc": now(), "git": git_state(),
                                                          "regimes": out, "p0_reproduction": rep}, overwrite=True)
    print(json.dumps(rep, indent=1))


def run_select(run_dir: str):
    C = p21()
    R = Path(run_dir)
    meta = dev_rows()
    refs = dev_refs([m["utterance_id"] for m in meta])
    FILL = fillers()
    recs, table = [], []
    for ep in sorted(int(p.name.split("epoch")[1]) for p in R.glob("epoch*") if p.name.split("epoch")[1].isdigit()):
        if ep == 0:
            continue
        row = {"epoch": ep}
        for regime in ("beam5", "greedy"):
            d = {r["id"]: r for r in read_jsonl(R / f"epoch{ep}" / f"dev_{regime}.jsonl")}
            m = metrics(score_system(meta, refs, d, FILL, C["evaluation"]["loop_units"]))
            row[regime] = {"mer": m["overall"]["mer"], "cs_mer": m["CS"]["mer"], "zh_cer": m["overall"]["zh_cer"],
                           "en_wer": m["overall"]["en_wer"], "pier": m["overall"]["pier"]}
        table.append(row)
        recs.append({"epoch": ep, **{k: row["beam5"][k] for k in ("cs_mer", "mer", "zh_cer")}})
    sel = select_epoch(recs)
    write_json(R / "selection.json", {"schema": "reft_p2_selection_v1", "utc": now(), "rule": C["selection"]["rule"],
                                      "epochs": table, "selected": sel})
    print(json.dumps({"epochs": table, "selected": sel}, indent=1))


def run_evaluate(run_dirs: list[str]):
    C = p21()
    meta = dev_rows()
    refs = dev_refs([m["utterance_id"] for m in meta])
    FILL = fillers()
    BS = C["evaluation"]["bootstrap"]
    res = {}
    for regime in ("beam5", "greedy"):
        assert sha(art(f"p2_0/frozen_baseline_dev150_{regime}.jsonl")) == C["frozen_baseline"][f"{regime}_sha256"], "pinned baseline changed"
        base = score_system(meta, refs, {r["id"]: r for r in read_jsonl(art(f"p2_0/frozen_baseline_dev150_{regime}.jsonl"))},
                            FILL, C["evaluation"]["loop_units"])
        res[regime] = {"baseline": metrics(base)}
        for rd in run_dirs:
            R = Path(rd)
            ep = read_json(R / "selection.json")["selected"]["epoch"]
            sc = score_system(meta, refs, {r["id"]: r for r in read_jsonl(R / f"epoch{ep}" / f"dev_{regime}.jsonl")}, FILL,
                              C["evaluation"]["loop_units"])
            man = read_json(R / "manifest.json")
            res[regime][man["method"]] = {"epoch": ep, "metrics": metrics(sc), "vs_baseline": paired(base, sc, BS["reps"], BS["seed"]),
                                          "efficiency": man.get("efficiency")}
    out = art("p2_1/evaluation.json")
    write_json(out, {"schema": "reft_p2_1_evaluation_v1", "utc": now(), "git": git_state(), "results": res,
                     "note": "exploratory dev150 (exposed); selection on the same panel; no untouched-test claim"})
    print("written", out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score-baseline", "select", "evaluate"])
    ap.add_argument("--run")
    ap.add_argument("--runs", nargs="*")
    a = ap.parse_args(argv)
    if a.cmd == "score-baseline":
        run_score_baseline()
    elif a.cmd == "select":
        run_select(a.run)
    else:
        run_evaluate(a.runs)


if __name__ == "__main__":
    main()
