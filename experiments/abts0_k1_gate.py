#!/usr/bin/env python
"""ABTS-0 K1 amendment seal and pre-score gate G1 (CPU; reference-free). Approved in ABTS0_K1_REPORT.md section 10.

seal-amendments -- before capture: verify the sealed plan (hash, config, all source pins, design pin) and pin every
                   evaluator-side file (v1 + v2 evaluator, this gate, sbatch, groups/registry, evaluation libraries)
                   -> amendment_seal.json with status READY_FOR_K1_CAPTURE (or BLOCKED with reasons).
check           -- after capture + select + the pushed candidate seal, before PHASE=score:
                   integrity (any failure -> K1_BLOCKED) and MM capacity (failure -> K1_INSUFFICIENT/STRUCTURAL, no
                   scoring) -> gate_prescore.json. Exit 0 only on PASS. Opens no reference or label: MM membership
                   comes from groups_prerun.json, frozen before capture.
verify          -- used by slurm PHASE=score and by the v2 evaluator: re-checks the gate and amendment seal.
verify-amendments -- used by slurm PHASE=capture: refuse to capture unless the amendments are sealed and unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from csasr.inference_cf.core import atomic_json, digest, file_hash
import experiments.abts0_k1 as R

BASE = R.BASE
AMEND_SEAL = f"{BASE}/amendment_seal.json"
GATE = f"{BASE}/gate_prescore.json"
GROUPS = f"{BASE}/groups_prerun.json"
AMEND_FILES = ("experiments/abts0_k1_evaluate.py", "experiments/abts0_k1_evaluate_v2.py", "experiments/abts0_k1_gate.py",
               "slurm/abts0_k1.sbatch", GROUPS, f"{BASE}/registry_prerun.json", R.PLAN, R.CONFIG,
               "docs/research_review/ABTS0_DESIGN.md", "src/csasr/evaluation/mer.py", "src/csasr/evaluation/pier.py",
               "src/csasr/evaluation/normalization.py", "tests/test_abts0_k1.py", "tests/test_abts0_k1_v2.py")
N_ROWS = 300


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, text=True, capture_output=True).stdout.strip()


def committed_and_pushed(path: str) -> bool:
    """File tracked, identical to HEAD, and its last commit is on a remote-tracking branch."""
    if subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=ROOT, capture_output=True).returncode:
        return False
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", path], cwd=ROOT).returncode:
        return False
    last = _git("log", "-1", "--format=%H", "--", path)
    return bool(last) and bool(_git("branch", "-r", "--contains", last))


# ---- amendment seal -----------------------------------------------------------------------------------------------

def cmd_seal_amendments(args) -> None:
    problems = []
    plan = R.load_plan()                                    # plan self-hash + config hash
    try:
        R.check_sources(plan)
    except ValueError as e:
        problems.append(str(e))
    c = R.cfg()
    if file_hash(ROOT / c["design"]["path"]) != c["design"]["sha256"]:
        problems.append("design changed")
    if set(plan["sources"]) != set(R.SOURCES):
        problems.append("plan source list differs from runner SOURCES")
    groups = json.loads((ROOT / GROUPS).read_text())
    if groups["plan_hash"] != plan["plan_hash"] or groups["k1_outputs_read"] or len(groups["rows"]) != N_ROWS \
            or digest({k: v for k, v in groups.items() if k != "groups_hash"}) != groups["groups_hash"]:
        problems.append("groups_prerun invalid")
    if (ROOT / R.RUN / "capture").exists() and any((ROOT / R.RUN / "capture").iterdir()):
        problems.append("capture outputs already exist: amendments would not be pre-registered")
    seal = {"schema": "abts0_k1_amendment_seal_v1", "plan_hash": plan["plan_hash"],
            "plan_sources": plan["sources"], "config_sha256": plan["config_sha256"], "design": c["design"]["sha256"],
            "files": {p: file_hash(ROOT / p) for p in AMEND_FILES}, "git_head": R.git_head(),
            "amendments": ["A1 timing-free lexical label", "A2 onset recall /reachable and /genuine heard",
                           "A3 BLOCKED>INSUFFICIENT(STRUCTURAL|REACHABLE|COVERAGE|CLASSES)>FAIL>PASS", "G1 pre-score gate"],
            "runtime_unchanged": "association 200 ms, acoustic thresholds, selection rules, population, GO numbers",
            "problems": problems, "status": "BLOCKED" if problems else "READY_FOR_K1_CAPTURE"}
    seal["seal_hash"] = digest(seal)
    atomic_json(ROOT / AMEND_SEAL, seal)
    print(seal["status"], seal["seal_hash"], problems)


def verify_amendment_seal() -> dict:
    seal = json.loads((ROOT / AMEND_SEAL).read_text())
    if digest({k: v for k, v in seal.items() if k != "seal_hash"}) != seal["seal_hash"]:
        raise ValueError("amendment seal hash")
    if seal["status"] != "READY_FOR_K1_CAPTURE":
        raise ValueError(f"amendment seal status {seal['status']}")
    bad = [p for p, h in seal["files"].items() if file_hash(ROOT / p) != h]
    bad += [p for p, h in seal["plan_sources"].items() if file_hash(ROOT / p) != h]
    if bad:
        raise ValueError(f"files changed after the amendment seal: {bad}")
    return seal


# ---- G1 -----------------------------------------------------------------------------------------------------------

def capacity(rows: list[dict], groups: dict, go: dict) -> dict:
    """Reference-free necessary capacity of the frozen MM GO gates from the sealed selection only."""
    per_d = Counter()
    for row in rows:
        ch = row["chosen"]
        if groups[row["utterance_id"]]["group"] == "MM" and ch is not None and ch["encoding_valid"]:
            per_d[groups[row["utterance_id"]]["dialogue_id"]] += 1
    n, nd, n2 = sum(per_d.values()), len(per_d), sum(1 for v in per_d.values() if v >= 2)
    checks = {"mm_encoding_valid_chosen": n >= go["accepted_min"], "mm_dialogues": nd >= go["accepted_dialogues_min"],
              "mm_dialogues_with_2": n2 >= go["auroc_dialogues_min"]}
    return {"mm_encoding_valid_chosen": n, "mm_dialogues": nd, "mm_dialogues_with_2": n2, "by_dialogue": dict(per_d),
            "checks": checks, "ok": all(checks.values())}


def integrity(plan: dict) -> list[str]:
    bad = []
    caps = []
    for r in plan["rows"]:
        p = ROOT / R.RUN / "capture" / f"{r['index']:03d}.json"
        if not p.exists():
            bad.append(f"missing capture {r['index']:03d}")
            continue
        d = json.loads(p.read_text())
        if d["plan_hash"] != plan["plan_hash"] or d["utterance_id"] != r["utterance_id"] or d["index"] != r["index"]:
            bad.append(f"capture identity {r['index']:03d}")
        if not d["B0"]["equals_B0M"]:
            bad.append(f"B0 != B0M {r['utterance_id']}")
        caps.append(d)
    if len(caps) != N_ROWS:
        bad.append(f"capture rows {len(caps)}/{N_ROWS}")
    objs = {d[b]["heads"]["path"]: d[b]["heads"]["sha256"] for d in caps for b in ("B0", "EN")}
    bad += [f"archive {p}" for p, h in objs.items() if not Path(p).exists() or file_hash(p) != h]
    rt_path = ROOT / R.RUN / "runtime_capture.json"
    if not rt_path.exists():
        bad.append("runtime_capture.json missing")
    else:
        rt = json.loads(rt_path.read_text())
        if not (rt.get("weights_unchanged_probe") and rt.get("model_grads_none")):
            bad.append("capture runtime: weights/grads")
        if any(any(f in p for f in R.FORBIDDEN) for p in rt.get("opened_paths", [])):
            bad.append("capture runtime: forbidden path opened")
    seal = json.loads((ROOT / R.CAND_SEAL).read_text())
    if seal["selection_sha256"] != file_hash(ROOT / R.SELECTION) or seal["plan_hash"] != plan["plan_hash"]:
        bad.append("selection seal mismatch")
    if any(file_hash(ROOT / p) != h for p, h in seal["capture_files"].items()):
        bad.append("capture changed after selection seal")
    for p in (R.SELECTION, R.CAND_SEAL):
        if not committed_and_pushed(p):
            bad.append(f"not committed+pushed: {p}")
    return bad


def cmd_check(args) -> int:
    sys.addaudithook(R._audit_open)
    amend = verify_amendment_seal()
    plan = R.load_plan()
    R.check_sources(plan)
    groups = json.loads((ROOT / GROUPS).read_text())["rows"]
    sel = json.loads((ROOT / R.SELECTION).read_text())
    bad = integrity(plan)
    cap = capacity(sel["rows"], groups, R.cfg()["evaluator"]["go_MM"])
    R.firewall_check()
    status = "K1_BLOCKED" if bad else ("PASS" if cap["ok"] else "K1_INSUFFICIENT")
    out = {"schema": "abts0_k1_gate_v1", "status": status, "reason": None if status == "PASS" else
           ("INTEGRITY" if bad else "STRUCTURAL"), "integrity_failures": bad, "capacity": cap,
           "selection_sha256": file_hash(ROOT / R.SELECTION), "plan_hash": plan["plan_hash"],
           "amendment_seal_hash": amend["seal_hash"], "references_opened": False, "git_head": R.git_head()}
    out["gate_hash"] = digest(out)
    atomic_json(ROOT / GATE, out)
    print(status, out["reason"], {k: cap[k] for k in ("mm_encoding_valid_chosen", "mm_dialogues", "mm_dialogues_with_2")})
    return 0 if status == "PASS" else 1


def verify_gate() -> dict:
    g = json.loads((ROOT / GATE).read_text())
    if digest({k: v for k, v in g.items() if k != "gate_hash"}) != g["gate_hash"]:
        raise ValueError("gate hash")
    if g["status"] != "PASS":
        raise ValueError(f"gate status {g['status']}: no scoring / evaluation")
    if g["selection_sha256"] != file_hash(ROOT / R.SELECTION):
        raise ValueError("selection changed after gate")
    if g["amendment_seal_hash"] != verify_amendment_seal()["seal_hash"]:
        raise ValueError("amendment seal changed after gate")
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("seal-amendments", "check", "verify", "verify-amendments"))
    args = ap.parse_args()
    if args.stage == "seal-amendments":
        cmd_seal_amendments(args)
    elif args.stage == "check":
        sys.exit(cmd_check(args))
    elif args.stage == "verify-amendments":
        print("amendment seal verified", verify_amendment_seal()["seal_hash"])
    else:
        verify_gate()
        print("gate PASS verified")


if __name__ == "__main__":
    main()
