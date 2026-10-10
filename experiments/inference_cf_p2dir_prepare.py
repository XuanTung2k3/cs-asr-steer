#!/usr/bin/env python
"""CPU-only P2-DIR preparation (frozen spec sections 4, 10, 11).

``construction``  seal the construction-only projection of the frozen 180-position population
                  (utterance_id, dialogue_id, t, stratum ONLY) after verifying the P2-RJ positions
                  and parent P2-R population hashes against the frozen config; D-dev-select firewall.
``manifest``      immutable per-stage manifest (extract | exp1): clean committed Git HEAD/tree,
                  config/spec/source hashes, environment (python/torch/transformers/numpy/LAPACK),
                  local model file hashes, partition and suppression hashes; exp1 additionally pins
                  the extraction run and the sealed D1 folds.
``folds``         D1 NEW_P2_DIR_CROSSFIT_V1 leave-one-dialogue-out construction from the extraction
                  states (CPU float64); writes every fold's provenance, arrays and sealed vector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from csasr.inference_cf.core import atomic_json, digest, file_hash
from csasr.inference_cf import unique

SCHEMA = "p2dir_direction_identification_v1"
CONFIG = "configs/inference_cf/p2_dir_direction_identification.json"
SPEC = "docs/inference_cf/P2_DIR_DIRECTION_IDENTIFICATION_SPEC.md"
DESIGN = "docs/inference_cf/P2_DIR_CODEX_DESIGN.md"
POSITIONS = "results/inference_cf/p2rj/positions.json"
P2R_POPULATION = "results/inference_cf/p2r/population.json"
CONSTRUCTION = "results/inference_cf/p2dir/construction_population.json"
INFERENCE_PANEL = "results/inference_cf/p0_r2/inference_panel.json"
BASELINE_PANEL = "results/inference_cf/p2_A_r1_L16/panel.json"
MODEL = Path("/mnt/data/tungnx/whisper-large-v3")
MODEL_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
               "merges.txt", "added_tokens.json", "special_tokens_map.json", "normalizer.json",
               "preprocessor_config.json", "model.safetensors")
SOURCES = (SPEC, CONFIG, DESIGN, "docs/inference_cf/P2_DIR_PRE_RUN_AUDIT.md", POSITIONS, P2R_POPULATION, CONSTRUCTION,
           "src/csasr/inference_cf/directions.py", "src/csasr/inference_cf/unique.py",
           "src/csasr/inference_cf/readout.py", "src/csasr/inference_cf/core_p1.py",
           "src/csasr/inference_cf/core_r2.py", "src/csasr/lss/sites.py", "src/csasr/models/hooks.py",
           "experiments/inference_cf_p2dir.py", "experiments/inference_cf_p2dir_prepare.py",
           "experiments/inference_cf_p2dir_analyze.py", "experiments/inference_cf_p2dir_audit.py",
           "experiments/inference_cf_p2r.py", "experiments/inference_cf_p2rj.py",
           "experiments/inference_cf_cached.py", "slurm/inference_cf_p2dir.sbatch")
FORBIDDEN_ROLES = ("router-calib", "D-dev-confirm", "D-test")


def git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def environment() -> dict:
    import torch
    import transformers
    env = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
           "numpy": np.__version__, "cuda": torch.version.cuda, "platform": platform.platform()}
    try:
        env["numpy_config"] = np.show_config(mode="dicts")
    except TypeError:
        env["numpy_config"] = None
    try:
        from threadpoolctl import threadpool_info
        env["threadpools"] = threadpool_info()
    except Exception as exc:   # informational only
        env["threadpools"] = repr(exc)
    return env


def model_hashes() -> dict:
    return {f: file_hash(MODEL / f) for f in MODEL_FILES if (MODEL / f).exists()}


def firewall(utterances: list[str]) -> dict:
    inf = json.loads((ROOT / INFERENCE_PANEL).read_text())
    base = json.loads((ROOT / BASELINE_PANEL).read_text())
    if inf["role"] != "D-dev-select" or base["role"] != "D-dev-select":
        raise ValueError("panel role is not D-dev-select")
    allowed = {r["utterance_id"] for r in inf["rows"]} & {r["utterance_id"] for r in base["rows"]}
    bad = [u for u in utterances if u not in allowed]
    if bad:
        raise ValueError(f"utterances outside the exposed D-dev-select panel: {bad[:5]}")
    return {"role": "D-dev-select", "panel_hash": file_hash(ROOT / INFERENCE_PANEL), "checked": len(utterances)}


# ---- construction projection ------------------------------------------------------------------

def cmd_construction(args) -> None:
    cfg = json.loads((ROOT / CONFIG).read_text())
    for rel, h in cfg["source_sha256"].items():
        if rel.startswith("results/inference_cf/p2r") or rel.startswith("results/inference_cf/p2rj"):
            if file_hash(ROOT / rel) != "sha256:" + h:
                raise ValueError(f"frozen source changed: {rel}")
    pos = json.loads((ROOT / POSITIONS).read_text())
    pop = json.loads((ROOT / P2R_POPULATION).read_text())
    if pos["positions_hash"] != cfg["positions_hash"] or pop["population_hash"] != cfg["p2r_parent_population_hash"]:
        raise ValueError("frozen population hash mismatch")
    if digest({k: v for k, v in pos.items() if k != "positions_hash"}) != pos["positions_hash"]:
        raise ValueError("positions self-hash mismatch")
    counts = {}
    for p in pos["positions"]:
        counts[p["stratum"]] = counts.get(p["stratum"], 0) + 1
    if counts != cfg["strata_counts"]:
        raise ValueError(f"strata counts {counts}")
    rows = [{k: p[k] for k in unique.ALLOWED_FIELDS} for p in pos["positions"]]
    unique.check_construction_fields(rows)
    if len({(r["utterance_id"], r["t"]) for r in rows}) != len(rows) or any(int(r["t"]) < 1 for r in rows):
        raise ValueError("duplicate or ineligible position")
    doc = {"schema": SCHEMA + "_construction", "positions_hash": pos["positions_hash"],
           "p2r_population_hash": pop["population_hash"], "field_allowlist": list(unique.ALLOWED_FIELDS),
           "utterances": pos["utterances"], "positions": rows, "firewall": firewall(pos["utterances"]),
           "folds": {k: unique.fold_members(rows, k) for k in unique.fold_dialogues(rows)}}
    doc["construction_hash"] = digest(doc)
    path = ROOT / CONSTRUCTION
    if path.exists():
        old = json.loads(path.read_text())
        if old["construction_hash"] != doc["construction_hash"]:
            raise FileExistsError("frozen construction projection differs; never overwrite")
        print("construction unchanged", doc["construction_hash"])
        return
    atomic_json(path, doc)
    print(json.dumps({"positions": len(rows), "folds": len(doc["folds"]), "construction_hash": doc["construction_hash"]}))


# ---- folds --------------------------------------------------------------------------------------

def cmd_folds(args) -> None:
    ext = ROOT / args.extract
    em = json.loads((ext / "manifest.json").read_text())
    if em["schema"] != SCHEMA or em["stage"] != "extract" or \
            digest({k: v for k, v in em.items() if k != "manifest_hash"}) != em["manifest_hash"]:
        raise ValueError("invalid extraction manifest")
    rt = json.loads((ext / "runtime.json").read_text())
    if rt.get("status") != "completed":
        raise ValueError("extraction not completed")
    con = json.loads((ROOT / em["construction"]).read_text())
    if con["construction_hash"] != em["construction_hash"]:
        raise ValueError("construction hash")
    unique.check_construction_fields(con["positions"])
    with np.load(ext / "states.npz") as z:
        HB = z["H_B"].astype(np.float64)
    index = {(p["utterance_id"], int(p["t"])): i for i, p in enumerate(con["positions"])}
    out = ROOT / args.out
    if (out / "folds.json").exists():
        raise FileExistsError("sealed folds exist; never overwrite")
    (out / "vectors").mkdir(parents=True, exist_ok=True)
    (out / "arrays").mkdir(parents=True, exist_ok=True)
    folds = {}
    for k in unique.fold_dialogues(con["positions"]):
        mem = unique.fold_members(con["positions"], k)
        if mem != con["folds"][k]:
            raise ValueError(f"fold membership differs from the sealed plan: {k}")
        A = HB[[index[(u, t)] for u, t in mem["A"]]]
        Bm = HB[[index[(u, t)] for u, t in mem["B"]]]
        rec = unique.fit_fold(A, Bm)
        s = unique.summary(rec)
        s.update(dialogue=k, n_excluded=len(mem["excluded"]), membership_hash=digest(mem),
                 state_rows_hash={"A": unique.array_hash(A), "B": unique.array_hash(Bm)})
        np.savez(out / "arrays" / f"{k}.npz", **unique.arrays(rec))
        s["arrays_path"] = str((out / "arrays" / f"{k}.npz").relative_to(ROOT))
        s["arrays_sha256"] = file_hash(out / "arrays" / f"{k}.npz")
        if rec["status"] == "ok":
            vp = out / "vectors" / f"{k}.npy"
            np.save(vp, rec["vector"], allow_pickle=False)
            s["vector_path"] = str(vp.relative_to(ROOT))
            s["vector_file_sha256"] = file_hash(vp)
        folds[k] = s
    doc = {"schema": SCHEMA + "_folds", "definition": unique.VERSION, "extraction_manifest_hash": em["manifest_hash"],
           "extraction_states_hash": file_hash(ext / "states.npz"), "construction_hash": con["construction_hash"],
           "git_commit": git("rev-parse", "HEAD"), "environment": environment(), "folds": folds,
           "valid_folds": sum(f["status"] == "ok" for f in folds.values()), "created_unix": time.time()}
    doc["folds_hash"] = digest(doc)
    atomic_json(out / "folds.json", doc)
    print(json.dumps({"folds": len(folds), "valid": doc["valid_folds"],
                      "invalid": {k: f["reason"] for k, f in folds.items() if f["status"] != "ok"},
                      "folds_hash": doc["folds_hash"]}))


# ---- manifest -----------------------------------------------------------------------------------

def cmd_manifest(args) -> None:
    from transformers import GenerationConfig, WhisperProcessor
    from csasr.inference_cf.core_r2 import tokenizer_partition

    cfg = json.loads((ROOT / CONFIG).read_text())
    con = json.loads((ROOT / CONSTRUCTION).read_text())
    pos = json.loads((ROOT / POSITIONS).read_text())
    if con["construction_hash"] != digest({k: v for k, v in con.items() if k != "construction_hash"}):
        raise ValueError("construction self-hash")
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *SOURCES)
    if dirty:
        raise ValueError("commit P2-DIR sources before preparing a manifest:\n" + dirty)
    partition = tokenizer_partition(WhisperProcessor.from_pretrained(MODEL, local_files_only=True).tokenizer)
    gen = GenerationConfig.from_pretrained(MODEL, local_files_only=True)
    man = {"schema": SCHEMA, "stage": args.stage, "git_commit": git("rev-parse", "HEAD"),
           "git_tree": git("rev-parse", "HEAD^{tree}"), "config": CONFIG, "config_hash": digest(cfg),
           "spec": SPEC, "spec_sha256": file_hash(ROOT / SPEC), "construction": CONSTRUCTION,
           "construction_hash": con["construction_hash"], "positions": POSITIONS, "positions_hash": pos["positions_hash"],
           "p2r_population_hash": cfg["p2r_parent_population_hash"], "partition_hash": partition["hash"],
           "partition_version": partition["version"],
           "suppression_hash": digest({"suppress": list(gen.suppress_tokens or []),
                                       "begin": list(gen.begin_suppress_tokens or [])}),
           "role": "D-dev-select", "firewall": firewall(con["utterances"]), "environment": environment(),
           "model": {"dir": str(MODEL), "files": model_hashes()},
           "precision": {"forward": "bfloat16", "attention": "eager", "geometry": "float64 CPU"},
           "sources": {p: file_hash(ROOT / p) for p in SOURCES}, "created_unix": time.time()}
    if args.stage == "exp1":
        ext = ROOT / args.extraction
        em = json.loads((ext / "manifest.json").read_text())
        folds = json.loads((ROOT / args.folds / "folds.json").read_text())
        if digest({k: v for k, v in folds.items() if k != "folds_hash"}) != folds["folds_hash"]:
            raise ValueError("folds self-hash")
        if folds["extraction_manifest_hash"] != em["manifest_hash"] or folds["extraction_states_hash"] != file_hash(ext / "states.npz"):
            raise ValueError("folds were not built from this extraction")
        for k, f in folds["folds"].items():
            if f["status"] == "ok" and file_hash(ROOT / f["vector_path"]) != f["vector_file_sha256"]:
                raise ValueError(f"fold vector changed: {k}")
        man.update(extraction_run=args.extraction, extraction_manifest_hash=em["manifest_hash"],
                   extraction_states_hash=file_hash(ext / "states.npz"),
                   folds=str(Path(args.folds) / "folds.json"), folds_hash=folds["folds_hash"])
    man["manifest_hash"] = digest(man)
    if args.dry_run:
        print(json.dumps({"manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}, indent=2))
        return
    out = ROOT / args.out
    if (out / "manifest.json").exists():
        raise FileExistsError("P2-DIR manifest exists; never overwrite a frozen run")
    atomic_json(out / "manifest.json", man)
    print(json.dumps({"stage": args.stage, "manifest_hash": man["manifest_hash"], "git_commit": man["git_commit"]}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("construction")
    f = sub.add_parser("folds")
    f.add_argument("--extract", required=True)
    f.add_argument("--out", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--stage", required=True, choices=("extract", "exp1"))
    m.add_argument("--out", required=True)
    m.add_argument("--extraction")
    m.add_argument("--folds")
    m.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    {"construction": cmd_construction, "folds": cmd_folds, "manifest": cmd_manifest}[args.cmd](args)


if __name__ == "__main__":
    main()
