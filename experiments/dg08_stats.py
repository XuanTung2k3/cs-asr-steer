#!/usr/bin/env python
"""DG-08 offline statistics and paper-ready tables (CPU only).

Consumes the locked D-test per-system transcripts, computes point tables
(greedy Table A, beam-5 Table B) with per-seed mean +/- std, an efficiency
table (Table C), and dialogue-block bootstrap 95% CIs for M* vs SALSA / LoRA /
Frozen Whisper on PIER, MER, and canonical net-correction utility (Table D).

Corpus MER/PIER/WER/CER/retention are micro-averaged additive sums, so the
bootstrap aggregates per-dialogue sufficient statistics and resamples the 15
D-test dialogues with replacement -- exact and fast over 2000 replicates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from csasr.evaluation import canonical, retention as ret
from csasr.utils.config import load_config

SEEDS = [13, 42, 73]
LEARNED = ["F2_SALSA", "F3_LORA", "F4_MSTAR"]
DTEST = REPO / "results/dg08/dtest"


def _comp(ref: str, base: str, hyp: str) -> dict[str, float]:
    cm = canonical.corpus_metrics([ref], [hyp])
    tr = canonical.correction_corruption([ref], [base], [hyp])
    rr = ret.retention_report([ref], [base], [hyp])
    en_n = int(cm["num_en_ref"]); zh_n = int(cm["num_zh_ref"])
    return {
        "mer_num": int(cm["substitutions"] + cm["deletions"] + cm["insertions"]),
        "mer_den": int(cm["num_ref_units"]),
        "en_err": int(round(cm["en_wer"] * en_n)) if en_n else 0, "en_n": en_n,
        "zh_err": int(round(cm["zh_cer"] * zh_n)) if zh_n else 0, "zh_n": zh_n,
        "poi_err": int(cm["num_poi_errors"]), "poi_n": int(cm["num_poi"]),
        "corrections": int(tr["corrections"]), "corruptions": int(tr["corruptions"]),
        "mret_num": int(rr["matrix_zh"]["numerator"]), "mret_den": int(rr["matrix_zh"]["denominator"]),
        "eret_num": int(rr["embedded_en"]["numerator"]), "eret_den": int(rr["embedded_en"]["denominator"]),
    }


def _corpus_from_components(agg: Mapping[str, float]) -> dict[str, float | None]:
    def rate(num, den):
        return (agg[num] / agg[den]) if agg[den] else None
    return {
        "mer": rate("mer_num", "mer_den"), "pier": rate("poi_err", "poi_n"),
        "en_wer": rate("en_err", "en_n"), "zh_cer": rate("zh_err", "zh_n"),
        "corrections": agg["corrections"], "corruptions": agg["corruptions"],
        "utility": agg["corrections"] - agg["corruptions"],
        "matrix_retention": rate("mret_num", "mret_den"),
        "embedded_retention": rate("eret_num", "eret_den"),
    }


def _load_texts(regime: str, tag: str) -> dict[str, str] | None:
    path = DTEST / regime / f"{tag}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["texts"]


def _arm_components(ids, refs, base, hyp) -> tuple[dict, dict]:
    """Per-utterance and per-utterance-keyed components for one arm."""
    per_utt = {}
    total = {k: 0 for k in ("mer_num", "mer_den", "en_err", "en_n", "zh_err", "zh_n",
                            "poi_err", "poi_n", "corrections", "corruptions",
                            "mret_num", "mret_den", "eret_num", "eret_den")}
    for uid in ids:
        c = _comp(refs[uid], base[uid], hyp[uid])
        per_utt[uid] = c
        for k in total:
            total[k] += c[k]
    return per_utt, total


def _mean_std(values: Sequence[float | None]) -> dict[str, float | None]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "seeds": list(values)}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0)),
            "seeds": [None if v is None else float(v) for v in values]}


def _dialogue_aggregate(per_utt: Mapping[str, dict], dialogues: Mapping[str, str]):
    keys = ("mer_num", "mer_den", "en_err", "en_n", "zh_err", "zh_n", "poi_err", "poi_n",
            "corrections", "corruptions", "mret_num", "mret_den", "eret_num", "eret_den")
    dmap: dict[str, dict[str, float]] = {}
    for uid, comp in per_utt.items():
        d = dialogues[uid]
        acc = dmap.setdefault(d, {k: 0 for k in keys})
        for k in keys:
            acc[k] += comp[k]
    return dmap


def _metric_from_dialogues(dstats: Mapping[str, dict], dlist: Sequence[str], metric: str):
    keys = ("mer_num", "mer_den", "en_err", "en_n", "zh_err", "zh_n", "poi_err", "poi_n",
            "corrections", "corruptions", "mret_num", "mret_den", "eret_num", "eret_den")
    agg = {k: 0 for k in keys}
    for d in dlist:
        for k in keys:
            agg[k] += dstats[d][k]
    corpus = _corpus_from_components(agg)
    return corpus[metric]


def bootstrap_pairs(regime: str, refs, dialogues, ids, reps: int, seed: int) -> dict[str, Any]:
    """Dialogue-block bootstrap CIs for M* vs SALSA / LoRA / F0 on PIER/MER/utility."""
    base = _load_texts(regime, "F0_FROZEN_WHISPER")
    if base is None:
        return {}
    # per-arm per-dialogue sufficient statistics
    dstats: dict[str, dict] = {}
    have: dict[str, bool] = {}
    for system in LEARNED:
        for s in SEEDS:
            hyp = _load_texts(regime, f"{system}_seed{s}")
            have[f"{system}:{s}"] = hyp is not None
            if hyp is None:
                continue
            per_utt, _ = _arm_components(ids, refs, base, hyp)
            dstats[f"{system}:{s}"] = _dialogue_aggregate(per_utt, dialogues)
    dialogue_list = sorted(set(dialogues[u] for u in ids))
    rng = np.random.default_rng(seed)
    n = len(dialogue_list)
    resamples = [list(rng.choice(dialogue_list, size=n, replace=True)) for _ in range(reps)]

    def paired_ci(a_key_fn, b_key_fn, metric: str) -> dict[str, Any] | None:
        diffs = []
        for sample in resamples:
            seed_diffs = []
            for s in SEEDS:
                ak, bk = a_key_fn(s), b_key_fn(s)
                if ak not in dstats or (bk is not None and bk not in dstats):
                    continue
                a = _metric_from_dialogues(dstats[ak], sample, metric)
                if bk is None:  # deterministic F0 baseline (same for every seed)
                    b = _f0_metric[metric](sample)
                else:
                    b = _metric_from_dialogues(dstats[bk], sample, metric)
                if a is None or b is None:
                    continue
                seed_diffs.append(a - b)
            if seed_diffs:
                diffs.append(float(np.mean(seed_diffs)))
        if not diffs:
            return None
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return {"mean_diff": float(np.mean(diffs)), "ci95": [float(lo), float(hi)],
                "excludes_zero": bool(lo > 0 or hi < 0), "n_replicates": len(diffs)}

    # F0 per-dialogue stats (baseline vs itself -> metric of F0 transcripts)
    f0_per_utt, _ = _arm_components(ids, refs, base, base)
    f0_dstats = _dialogue_aggregate(f0_per_utt, dialogues)
    global _f0_metric
    _f0_metric = {m: (lambda sample, m=m: _metric_from_dialogues(f0_dstats, sample, m))
                  for m in ("pier", "mer", "utility")}

    out: dict[str, Any] = {"regime": regime, "reps": reps, "seed": seed,
                           "n_dialogues": n, "arms_present": have}
    comparisons = {
        "MSTAR_vs_SALSA": (lambda s: f"F4_MSTAR:{s}", lambda s: f"F2_SALSA:{s}"),
        "MSTAR_vs_LORA": (lambda s: f"F4_MSTAR:{s}", lambda s: f"F3_LORA:{s}"),
        "MSTAR_vs_FROZEN": (lambda s: f"F4_MSTAR:{s}", lambda s: None),
    }
    for cname, (a_fn, b_fn) in comparisons.items():
        out[cname] = {m: paired_ci(a_fn, b_fn, m) for m in ("pier", "mer", "utility")}
    return out


def point_table(regime: str, refs, ids) -> dict[str, Any]:
    base = _load_texts(regime, "F0_FROZEN_WHISPER")
    if base is None:
        return {}
    table: dict[str, Any] = {}
    # F0
    _, tot = _arm_components(ids, refs, base, base)
    table["F0_FROZEN_WHISPER"] = {"deterministic": True, **_corpus_from_components(tot)}
    # F1 (greedy only, deterministic)
    f1 = _load_texts(regime, "F1_DG04_FROZEN_STEERING")
    if f1 is not None:
        _, tot = _arm_components(ids, refs, base, f1)
        table["F1_DG04_FROZEN_STEERING"] = {"deterministic": True, **_corpus_from_components(tot)}
    # learned finalists: per-seed + mean/std
    for system in LEARNED:
        per_seed = {}
        for s in SEEDS:
            hyp = _load_texts(regime, f"{system}_seed{s}")
            if hyp is None:
                continue
            _, tot = _arm_components(ids, refs, base, hyp)
            per_seed[s] = _corpus_from_components(tot)
        if not per_seed:
            continue
        metrics = ["mer", "pier", "en_wer", "zh_cer", "corrections", "corruptions",
                   "utility", "matrix_retention", "embedded_retention"]
        agg = {"per_seed": per_seed}
        for m in metrics:
            agg[m] = _mean_std([per_seed[s][m] if s in per_seed else None for s in SEEDS])
        table[system] = agg
    return table


def efficiency_table() -> dict[str, Any]:
    def train_summary(rel: str) -> dict[str, Any] | None:
        p = REPO / rel
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    sources = {
        "F2_SALSA": {42: "results/dg07/LB1_SALSA_EXACT_GLOBAL/summary.json",
                     13: "results/dg08/train/SALSA/seed13/summary.json",
                     73: "results/dg08/train/SALSA/seed73/summary.json"},
        "F3_LORA": {42: "results/dg07/LB2_LORA_MATCHED_BUDGET/summary.json",
                    13: "results/dg08/train/LORA/seed13/summary.json",
                    73: "results/dg08/train/LORA/seed73/summary.json"},
        "F4_MSTAR": {42: "results/dg06/d1/summary.json",
                     13: "results/dg08/train/MSTAR/seed13/summary.json",
                     73: "results/dg08/train/MSTAR/seed73/summary.json"},
    }
    backbone = 1543490560
    out: dict[str, Any] = {}
    for system, seedmap in sources.items():
        params = None; times = []; mems = []; ckpt_size = None; updates = None
        for s, rel in seedmap.items():
            summ = train_summary(rel)
            if not summ:
                continue
            params = params or summ.get("trainable_parameters") or summ.get("trainable_parameter_count")
            rt = summ.get("runtime_sec")
            if rt:
                times.append(float(rt))
            pk = summ.get("peak_gpu_memory_bytes")
            if pk:
                mems.append(int(pk))
            ckpt_size = ckpt_size or summ.get("checkpoint_size_bytes")
            updates = updates or summ.get("optimizer_updates") or 2019
        out[system] = {
            "trainable_parameters": params,
            "percent_of_backbone": (100.0 * params / backbone) if params else None,
            "checkpoint_size_bytes": ckpt_size,
            "optimizer_updates": updates,
            "train_wall_sec_mean": float(np.mean(times)) if times else None,
            "train_wall_sec_std": float(np.std(times)) if times else None,
            "peak_gpu_memory_bytes_mean": float(np.mean(mems)) if mems else None,
            "n_seeds_with_timing": len(times),
        }
    timing_path = REPO / "results/dg08/timing/timing.json"
    if timing_path.exists():
        out["inference_timing"] = json.loads(timing_path.read_text(encoding="utf-8"))["records"]
    return out


def _fmt(v, p=4):
    if v is None:
        return "n/a"
    if isinstance(v, dict):
        mean, std = v.get("mean"), v.get("std")
        return "n/a" if mean is None else f"{mean:.{p}f}±{std:.{p}f}"
    if isinstance(v, float):
        return f"{v:.{p}f}"
    return str(v)


def _markdown(table: dict[str, Any], regime: str) -> str:
    cols = ["mer", "pier", "en_wer", "zh_cer", "corrections", "corruptions", "utility",
            "matrix_retention", "embedded_retention"]
    lines = [f"### DG-08 {regime} D-test ({'greedy' if regime=='greedy' else 'beam-5'})",
             "", "| System | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    order = ["F0_FROZEN_WHISPER", "F1_DG04_FROZEN_STEERING", "F2_SALSA", "F3_LORA", "F4_MSTAR"]
    for name in order:
        if name not in table:
            continue
        row = table[name]
        cells = [name]
        for c in cols:
            cells.append(_fmt(row.get(c), 4 if c in ("mer", "pier", "en_wer", "zh_cer",
                                                     "matrix_retention", "embedded_retention") else 1))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    cfg = load_config("configs/dg08_locked_eval.yaml")
    manifest = json.loads((REPO / "results/dg08/dtest/dtest_manifest.json").read_text(encoding="utf-8"))
    refs = manifest["references"]; dialogues = manifest["dialogues"]; ids = manifest["ids"]
    reps = int(cfg["bootstrap"]["repetitions"]); bseed = int(cfg["bootstrap"]["seed"])
    out_tables = REPO / "results/dg08/tables"; out_tables.mkdir(parents=True, exist_ok=True)
    out_stats = REPO / "results/dg08/stats"; out_stats.mkdir(parents=True, exist_ok=True)

    all_tables: dict[str, Any] = {}
    for regime, label in (("greedy", "A"), ("beam5", "B")):
        if not (DTEST / regime / "F0_FROZEN_WHISPER.json").exists():
            continue
        table = point_table(regime, refs, ids)
        all_tables[regime] = table
        (out_tables / f"table_{label}_{regime}.json").write_text(
            json.dumps(table, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        (out_tables / f"table_{label}_{regime}.md").write_text(_markdown(table, regime), encoding="utf-8")
        boot = bootstrap_pairs(regime, refs, dialogues, ids, reps, bseed)
        (out_stats / f"bootstrap_{regime}.json").write_text(
            json.dumps(boot, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    eff = efficiency_table()
    (out_tables / "table_C_efficiency.json").write_text(
        json.dumps(eff, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print("DG-08 statistics written to results/dg08/tables and results/dg08/stats")
    for regime in all_tables:
        print(f"\n{_markdown(all_tables[regime], regime)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
