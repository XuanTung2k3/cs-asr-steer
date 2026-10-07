#!/usr/bin/env python
"""P2-TTA-A3 runner (soft AUTO-KL conditioning-gap debug). Contract: docs/inference_cf/P2_TTA_A3_SPEC.md,
P2_TTA_A3_CLAUDE_DESIGN.md, configs/inference_cf/p2_tta_a3.json, docs/inference_cf/P2_TTA_A3_PANEL.json.

panel     (CPU) reference-free 24-panel from sealed theta0 FORCED/AUTO decoded text of the fixed-100 panel.
prepare   (CPU) teacher/reuse plan: y_B (P2-SEQ S0), y_A (historical AUTO), A2 outputs (TTA1/TTA0-R rows), masks.
manifest  (CPU) resolved manifest after committed PASS_TO_P2_TTA_A3.
run       (GPU) per utterance: theta0 encoder once; theta0 forced decode = S0; AUTO prompt recovered with the historical
          `detect_language(input_features=...)` call; theta0 AUTO replay (must reproduce historical AUTO text);
          A3 episode (fresh fp32 LN masters + fresh AdamW, 2 steps, forward KL(q_AUTO || p_FORCED) on the common y_A
          path); final forced-ZH decode; exact reset. First row: independent live loss/gradient check.
seal      (CPU) output seal before any reference access.
No reference, evaluator, steering, D2 or LID input anywhere in this module.
"""
from __future__ import annotations

import argparse
import hashlib
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

from csasr.inference_cf.core import atomic_json, digest, file_hash

SCHEMA = "p2_tta_a3_v1"
CONFIG = "configs/inference_cf/p2_tta_a3.json"
PANEL = "docs/inference_cf/P2_TTA_A3_PANEL.json"
PARENT = "docs/inference_cf/P2_SEL_MINI_PANEL.json"
FUNNEL = "configs/inference_cf/p2_tta_funnel.json"
BASE = "results/inference_cf/p2tta_a3"
PLAN = f"{BASE}/plan_sealed.json"
CB = [50258, 50260, 50360, 50364]
TRANSCRIBE, NOTIMESTAMPS, SOT = 50360, 50364, 50258
MAX_NEW = 200
REL_GRAD_TOL = 0.02
TARGET, MAX_D, MAX_PER_DIALOGUE = 24, 16, 2


def git(*a):
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def committed(rel: str) -> bool:
    blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout
    return git("ls-files", rel) == rel and hashlib.sha256(blob).hexdigest() == sha_file(ROOT / rel)


# ---- reference-free panel ---------------------------------------------------------------------------------------

def build_panel(parent_rows: list[dict], forced_text: dict, auto_text: dict) -> list[dict]:
    """D = exact stored theta0 AUTO text != FORCED text; A = equal. All D if |D| <= 16, else dialogue round-robin
    (dialogue first-appearance order, parent order within dialogue). Controls fill to 24 from A: first one per
    dialogue NOT represented by D (dialogue first-appearance order, parent order within), then round-robin over
    D-represented dialogues with < 2 selected, then any dialogue with < 2, then (only if still short) any dialogue.
    No reference/quality input."""
    order = []
    for r in parent_rows:
        if r["dialogue_id"] not in order:
            order.append(r["dialogue_id"])
    grp = {r["utterance_id"]: ("D" if auto_text[r["utterance_id"]] != forced_text[r["utterance_id"]] else "A") for r in parent_rows}
    idx = {r["utterance_id"]: i for i, r in enumerate(parent_rows)}
    by_d = {d: [r for r in parent_rows if r["dialogue_id"] == d] for d in order}
    D = [r for r in parent_rows if grp[r["utterance_id"]] == "D"]
    if len(D) > MAX_D:
        queues = {d: [r for r in by_d[d] if grp[r["utterance_id"]] == "D"] for d in order}
        D = []
        while len(D) < MAX_D:
            for d in order:
                if queues[d] and len(D) < MAX_D:
                    D.append(queues[d].pop(0))
    sel = list(D)
    count = {d: sum(r["dialogue_id"] == d for r in sel) for d in order}
    avail = {d: [r for r in by_d[d] if grp[r["utterance_id"]] == "A"] for d in order}

    def take(d):
        r = avail[d].pop(0)
        sel.append(r)
        count[d] += 1

    for d in order:                                  # 1) unrepresented dialogues, one each
        if len(sel) < TARGET and count[d] == 0 and avail[d]:
            take(d)
    represented = [d for d in order if any(r["dialogue_id"] == d for r in D)]
    for pool, cap in ((represented, MAX_PER_DIALOGUE), (order, MAX_PER_DIALOGUE), (order, None)):
        progress = True                              # 2) round-robin over D-represented dialogues (<2 selected);
        while len(sel) < TARGET and progress:        # 3) any dialogue <2; 4) only if still required, any dialogue
            progress = False
            for d in pool:
                if len(sel) < TARGET and avail[d] and (cap is None or count[d] < cap):
                    take(d)
                    progress = True
    if len(sel) != TARGET or len({r["utterance_id"] for r in sel}) != TARGET:
        raise ValueError("cannot build 24-panel")
    return [{"utterance_id": r["utterance_id"], "dialogue_id": r["dialogue_id"], "group": grp[r["utterance_id"]],
             "parent_index": idx[r["utterance_id"]]} for r in sel]


def cmd_panel(args) -> None:
    f = json.loads((ROOT / FUNNEL).read_text())
    parent = json.loads((ROOT / PARENT).read_text())
    if sha_file(ROOT / PARENT) != f["TTA1"]["panel_sha256"]:
        raise ValueError("parent panel bytes")
    base = {b["utterance_id"]: b for b in f["baseline100"]}
    ft, at, src = {}, {}, {}
    for r in parent["rows"]:
        b = base[r["utterance_id"]]
        if sha_file(ROOT / b["forced_path"]) != b["forced_sha256"] or sha_file(ROOT / b["auto_path"]) != b["auto_sha256"]:
            raise ValueError("baseline bytes")
        ft[r["utterance_id"]] = json.loads((ROOT / b["forced_path"]).read_text())["systems"]["S0"]["text"]
        at[r["utterance_id"]] = json.loads((ROOT / b["auto_path"]).read_text())["systems"]["B0_AUTO"]["text"]
        src[r["utterance_id"]] = {"forced_path": b["forced_path"], "forced_sha256": b["forced_sha256"],
                                  "auto_path": b["auto_path"], "auto_sha256": b["auto_sha256"]}
    rows = build_panel(parent["rows"], ft, at)
    nD = sum(1 for u in ft if ft[u] != at[u])
    doc = {"schema": "p2_tta_a3_panel_v1", "role": "already-exposed D-dev-select (fixed-100 subset, reference-free)",
           "parent": PARENT, "parent_byte_sha256": sha_file(ROOT / PARENT),
           "selection": "D = exact stored theta0 AUTO decoded text != FORCED decoded text (no normalization); all D if |D|<=16; "
                        "controls from A: one per D-unrepresented dialogue (dialogue first appearance, parent order), then "
                        "round-robin over D-represented dialogues with <2 selected, then any dialogue <2, then any only if required; no references/"
                        "errors/durations/A2 outcomes/quality",
           "parent_D": nD, "parent_A": len(ft) - nD, "rows": [{**r, **src[r["utterance_id"]]} for r in rows]}
    out = ROOT / PANEL
    if out.exists():
        raise FileExistsError("panel exists; never overwrite")
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"sha256": sha_file(out), "D": sum(r["group"] == "D" for r in rows), "A": sum(r["group"] == "A" for r in rows),
                      "dialogues": len({r["dialogue_id"] for r in rows}),
                      "max_per_dialogue": max(sum(x["dialogue_id"] == r["dialogue_id"] for x in rows) for r in rows)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("panel")
    args = ap.parse_args()
    {"panel": cmd_panel}[args.cmd](args)


if __name__ == "__main__":
    main()
