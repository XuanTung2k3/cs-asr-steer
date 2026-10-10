#!/usr/bin/env python
"""CPU-only reconstruction of canonical candidate outside harm for BASIS-A3.

This annotates already-decoded CS-Dialogue result blocks.  It does not decode,
alter panels, or participate in R2 selection; POI transitions remain the
headline correction/corruption population and candidate utility is separate.
SEAME is intentionally not handled because it has no accepted CS candidate
population in this study.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(REPO), str(REPO / "src")]

from csasr.evaluation.dg03_outside_harm import corpus_outside_harm  # noqa: E402
from csasr.utils.config import load_config  # noqa: E402
from experiments.dg04_frozen_baselines import _build_outside_sets  # noqa: E402


ROOT = REPO / "results/basis_a3_raw_cond_scope_depth"


def _atomic(path: Path, payload: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def repair(root: Path = ROOT) -> dict:
    dcfg = load_config("lss/l1b_candidates_dialogue_v2r3.yaml")
    panel = json.loads((root / "panels/cs_dialogue_300.json").read_text())
    ids = [str(x["utterance_id"]) for x in panel["rows"]]
    # Reuse the frozen D-dev-select references and accepted existing_ctc
    # candidate partition used by DG-03/DG-04; never use result quality here.
    import pyarrow.parquet as pq
    data_root = Path(dcfg["v2_namespace"]["role_root"]).parent.parent
    role = pq.read_table(data_root / "manifests/roles/role_D-dev-select.parquet",
                         columns=["utterance_id", "transcript_raw"],
                         filters=[("role", "==", "D-dev-select")]).to_pandas()
    refs = {str(u): str(t) for u, t in zip(role["utterance_id"], role["transcript_raw"])}
    refs = {u: refs[u] for u in ids}
    sets, langs, diag = _build_outside_sets(dcfg, refs, ids)
    paths = sorted((root / "raw_r1/cs_dialogue").glob("L*/*.json"))
    paths += sorted((root / "raw_r2/cs_dialogue").glob("L*/*.json"))
    paths += sorted((root / "conditioning/cs_dialogue").glob("L*/*.json"))
    updated = 0
    for path in paths:
        result = json.loads(path.read_text())
        base = [result["baseline_texts"][u] for u in ids]
        method = [result["texts"][u] for u in ids]
        summary = corpus_outside_harm([refs[u] for u in ids], base, method, sets, langs)
        m = result.setdefault("metrics", {})
        m["outside_harm"] = int(summary["outside_harm"])
        m["outside_harm_accounting"] = summary
        m["candidate_level_utility"] = float(summary["utility"])
        result.setdefault("provenance", {})["outside_harm_accounting"] = {
            "method": "frozen DG-03 existing_ctc candidate partition",
            "role": "D-dev-select", "population": diag,
            "gpu_rerun_required": False,
        }
        _atomic(path, result)
        updated += 1
    return {"updated": updated, "population": diag, "dataset": "cs_dialogue"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    print(json.dumps(repair(Path(args.root)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
