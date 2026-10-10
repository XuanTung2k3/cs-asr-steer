#!/usr/bin/env python
"""P2-R diagnostic population (CPU, evaluator/diagnostic namespace only).

Builds the frozen D1/D3 position panel and the D2 target/control ledger from the matched
baseline token sequences, the frozen R2 evaluator alignment and the role references. Nothing
here is importable by, or passed to, the deployable decode path (experiments/inference_cf_cached.py,
experiments/inference_cf_p2.py); the P2-R GPU runner receives the written plan file explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from csasr.data.normalize import normalize_text, segment_units
from csasr.evaluation.mer import MATCH, align_tokens
from csasr.inference_cf.core import atomic_json, digest

CONFIG = ROOT / "configs/inference_cf/p2_r_mechanism_diagnosis.json"
MODEL = Path("/mnt/data/tungnx/whisper-large-v3")
STRATA = ("EN-confusion", "EN-correct", "ZH-correct")
RAW_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9'’\-]*")


def order_key(tag: str, uid: str, t: int, stratum: str) -> str:
    return hashlib.sha256(f"{tag}|{uid}|{t}|{stratum}".encode()).hexdigest()


def raw_latin_surfaces(raw: str, ref_units) -> dict[int, str]:
    """Map normalized Latin unit index -> raw-case surface when the raw word order is unambiguous."""
    latin = [i for i, u in enumerate(ref_units) if u.kind == "latin"]
    raw_words = RAW_LATIN.findall(raw)
    if len(raw_words) != len(latin):
        return {}
    out = {}
    for i, w in zip(latin, raw_words):
        if normalize_text(w) != ref_units[i].surface:
            return {}
        out[i] = w
    return out


def target_set(tokenizer, prefix_ids: list[int], word: str, raw: str | None) -> tuple[list[int], str | None]:
    """Reference-consistent first-token set for an English word after the exact generated prefix.

    Surface variants: {raw (if recoverable), lower, Capitalized, UPPER} x {no space, one space}.
    A variant is valid only if encoding prefix+variant keeps the generated prefix IDs unchanged.
    A target token must be a distinguishing beginning of the word: its stripped, lower-cased text
    is a prefix of the word with length >= min(2, len(word)).
    """
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    forms = {word, word.capitalize(), word.upper()} | ({raw} if raw else set())
    out, prefix_changed = set(), 0
    for f in sorted(forms):
        for sep in ("", " "):
            ids = tokenizer.encode(prefix_text + sep + f, add_special_tokens=False)
            if ids[:len(prefix_ids)] != list(prefix_ids) or len(ids) <= len(prefix_ids):
                prefix_changed += 1
                continue
            tid = ids[len(prefix_ids)]
            s = tokenizer.decode([tid], clean_up_tokenization_spaces=False).strip().lower()
            if s and word.lower().startswith(s) and len(s) >= min(2, len(word)):
                out.add(int(tid))
    if not out:
        return [], ("prefix_changed" if prefix_changed == 2 * len(forms) else "no_distinguishing_target")
    return sorted(out), None


def build(config: dict) -> dict:
    from transformers import WhisperTokenizer
    from experiments.inference_cf_p0_r2_evaluate import unit_to_token_positions
    from experiments.inference_cf_p0_r2_prepare import build_panels
    from csasr.inference_cf.core_r2 import prefix_utf8_complete
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode

    tok = WhisperTokenizer.from_pretrained(MODEL, local_files_only=True)
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    _, ev = build_panels()
    r2m = json.loads((ROOT / "results/inference_cf/p0_r2/manifest.json").read_text())
    if digest(ev) != r2m["evaluation_panel_hash"]:
        raise ValueError("evaluation panel differs from frozen R2")
    refs = {r["utterance_id"]: r for r in ev["rows"]}
    units = json.loads((ROOT / "results/inference_cf/p0_r2/evaluation_units.json").read_text())
    p2dir = ROOT / config["baseline_run"]
    panel = json.loads((p2dir / "panel.json").read_text())["rows"]
    base = {}
    for i, item in enumerate(panel):
        row = json.loads((p2dir / "rows" / f"{i:03d}.json").read_text())
        base[item["utterance_id"]] = row["systems"][config["baseline_system"]]
    by_u = defaultdict(list)
    for u in units:
        by_u[u["utterance_id"]].append(u)
    ledger = Counter()
    candidates = {s: [] for s in STRATA}
    confusion_all = []
    for uid in sorted(by_u):
        us = by_u[uid]
        ids, text = base[uid]["tokens"], base[uid]["text"]
        ref_units = segment_units(normalize_text(refs[uid]["reference"]))
        hyp_units = segment_units(normalize_text(text))
        ops = align_tokens([x.surface for x in ref_units], [x.surface for x in hyp_units])
        op_of = {op.ref_idx: i for i, op in enumerate(ops) if op.ref_idx is not None}
        tpos = unit_to_token_positions(tok, ids, text)
        raw = raw_latin_surfaces(refs[uid]["reference"], ref_units)
        per_pos = Counter(u["position"] for u in us if u["reason"] is None and u["position"] is not None)
        for u in us:
            s = u["stratum"]
            if s not in STRATA:
                continue
            ledger[(s, "candidate_units")] += 1
            t = u["position"]

            def drop(reason):
                ledger[(s, reason)] += 1
            if u["reason"] is not None or t is None:
                drop("r2_unalignable"); continue
            if t < config["site"]["min_step"]:
                drop("forced_prefix_first_content_token"); continue
            if not prefix_utf8_complete(tok, ids[:t], byte_decoder):
                drop("mid_character_prefix"); continue
            op = ops[op_of[u["reference_unit_index"]]]
            hu = op.hyp_idx
            if hu is None or tpos[hu] != t or (hu > 0 and tpos[hu - 1] == t):
                drop("not_token_start"); continue
            if s != "ZH-correct" and per_pos[t] > 1:
                drop("multiple_reference_units_at_position"); continue
            rec = {"utterance_id": uid, "dialogue_id": u["dialogue_id"], "t": t, "stratum": s,
                   "reference_unit_index": u["reference_unit_index"], "surface": u["surface"],
                   "baseline_token": int(ids[t]), "ctc_midpoint_sec": u["ctc_midpoint_sec"]}
            if s == "ZH-correct":
                rec["target_ids"] = [int(ids[t])]
            else:
                tset, why = target_set(tok, ids[:t], u["surface"], raw.get(u["reference_unit_index"]))
                if not tset:
                    drop(why); continue
                if s == "EN-correct" and int(ids[t]) not in tset:
                    drop("baseline_token_not_in_reference_set"); continue
                if s == "EN-confusion" and int(ids[t]) in tset:
                    drop("baseline_token_in_reference_set"); continue
                rec["target_ids"] = tset
            if s == "EN-confusion":
                # span-initial: previous reference unit is correctly matched, or none precedes it
                i = op_of[u["reference_unit_index"]] - 1
                while i >= 0 and ops[i].ref_idx is None:
                    i -= 1
                rec["span_initial"] = bool(i < 0 or ops[i].op == MATCH)
            rec["order"] = order_key(config["population"]["hash_tag"], uid, t, s)
            ledger[(s, "valid")] += 1
            candidates[s].append(rec)
            if s == "EN-confusion":
                confusion_all.append(rec)
    return select(config, candidates, ledger)


def round_robin(cands: list[dict], quota: int, per_utt: int, chosen_utts: set[str], utt_cap: int,
                restrict: set[str] | None, used: Counter | None = None) -> list[dict]:
    by_d = defaultdict(list)
    for c in cands:
        if restrict is None or c["utterance_id"] in restrict:
            by_d[c["dialogue_id"]].append(c)
    for d in by_d:
        by_d[d].sort(key=lambda c: c["order"])
    picked = []
    used = Counter() if used is None else used
    cursor = {d: 0 for d in by_d}
    progress = True
    while len(picked) < quota and progress:
        progress = False
        for d in sorted(by_d):
            if len(picked) >= quota:
                break
            lst = by_d[d]
            while cursor[d] < len(lst):
                c = lst[cursor[d]]
                cursor[d] += 1
                u = c["utterance_id"]
                if used[u] >= per_utt:
                    continue
                if u not in chosen_utts and len(chosen_utts) >= utt_cap:
                    continue
                picked.append(c); used[u] += 1; chosen_utts.add(u); progress = True
                break
    return picked


def select(config: dict, candidates: dict, ledger: Counter) -> dict:
    pop = config["population"]
    quota, per_utt, cap = pop["positions_per_stratum"], pop["max_positions_per_utterance_per_stratum"], pop["max_utterances"]
    chosen: set[str] = set()
    d1 = {"EN-confusion": round_robin(candidates["EN-confusion"], quota, per_utt, chosen, cap, None)}
    for s in ("EN-correct", "ZH-correct"):
        used = Counter()          # the per-utterance cap spans both passes
        first = round_robin(candidates[s], quota, per_utt, chosen, cap, set(chosen), used)
        taken = {(c["utterance_id"], c["t"]) for c in first}
        rest = [c for c in candidates[s] if (c["utterance_id"], c["t"]) not in taken]
        more = round_robin(rest, quota - len(first), per_utt, chosen, cap, None, used) if len(first) < quota else []
        d1[s] = first + more
    for s, lst in d1.items():
        for k, c in enumerate(lst):
            c["selection_rank"] = k
            c["companion"] = k < pop["companion_positions_per_stratum"]
    utts = sorted(chosen)
    d2_targets = sorted([c for c in candidates["EN-confusion"] if c["utterance_id"] in chosen],
                        key=lambda c: (c["utterance_id"], c["order"]))
    d2_controls = [c for s in ("EN-correct", "ZH-correct") for c in candidates[s] if c["utterance_id"] in chosen]
    coverage = {s: {"positions": len(v), "utterances": len({c["utterance_id"] for c in v}),
                    "dialogues": len({c["dialogue_id"] for c in v}),
                    "span_initial": sum(bool(c.get("span_initial")) for c in v) if s == "EN-confusion" else None}
                for s, v in d1.items()}
    minimum = pop["minimum_interpretable"]
    estimable = {s: coverage[s]["positions"] >= minimum["positions"] and coverage[s]["dialogues"] >= minimum["dialogues"]
                 for s in STRATA}
    return {"schema": "p2r_population_v1", "config_hash": digest(config), "utterances": utts,
            "d1": d1, "d2_targets": d2_targets, "d2_controls": d2_controls,
            "coverage": coverage, "estimable": estimable,
            "d2_coverage": {"targets": len(d2_targets), "target_utterances": len({c["utterance_id"] for c in d2_targets}),
                            "controls": len(d2_controls)},
            "ledger": {f"{s}|{k}": v for (s, k), v in sorted(ledger.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    config = json.loads(CONFIG.read_text())
    pop = build(config)
    pop["population_hash"] = digest({k: v for k, v in pop.items()})
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError("P2-R population exists; never overwrite a frozen population")
    atomic_json(out, pop)
    print(json.dumps({"coverage": pop["coverage"], "estimable": pop["estimable"], "d2": pop["d2_coverage"],
                      "utterances": len(pop["utterances"]), "hash": pop["population_hash"]}, indent=2))


if __name__ == "__main__":
    main()
