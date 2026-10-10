#!/usr/bin/env python
"""P2-PATH5 pre-outcome panel freeze tool (reference-free). Builds docs/inference_cf/P2_PATH5_PANEL.json.

FULL300 = the canonical P0-R2 / P2-A r1 D-dev-select panel (results/inference_cf/p2_A_r1_L16/panel.json, 20 dialogues x 15),
which is the byte-pinned `parent_panel` of the historical fixed100 (P2_SEL_MINI_PANEL.json). NEW200 = FULL300 - FIXED100.
Reads only IDs/dialogues/roles (role manifest ID/dialogue/role columns), audio bytes, sealed token outputs/hashes and the
OPP0 derived G1A outputs. No references, error counts, POI/language labels or durations are read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

FULL = "results/inference_cf/p2_A_r1_L16/panel.json"
FIXED = "docs/inference_cf/P2_SEL_MINI_PANEL.json"
P4PANEL = "docs/inference_cf/P2_PATH4_PANEL.json"
DERIVED = "results/inference_cf/p2opp0/derived_g1a.json"
ROLE = "/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/manifests/roles/role_D-dev-select.parquet"
OUT = "docs/inference_cf/P2_PATH5_PANEL.json"
CB = [50258, 50260, 50360, 50364]
EOS = 50257


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fp(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def membership(name, ids, dlg):
    return fp({"partition": name, "rows": [{"utterance_id": u, "dialogue_id": dlg[u]} for u in ids]})


def build() -> dict:
    import pyarrow.parquet as pq
    from transformers import WhisperProcessor
    import experiments.inference_cf_p2tta0 as t0run
    from csasr.inference_cf.episodic_tta import clean_teacher, special_ids, valid_mask
    from transformers import GenerationConfig
    fx = json.loads((ROOT / FIXED).read_text())
    full_p = json.loads((ROOT / FULL).read_text())
    assert fx["parent_panel"] == FULL and fx["parent_panel_sha256"] == sha(ROOT / FULL)
    full = [r["utterance_id"] for r in full_p["rows"]]
    fixed = [r["utterance_id"] for r in fx["rows"]]
    role = pq.read_table(ROLE, columns=["utterance_id", "dialogue_id", "role"]).to_pylist()
    rm = {}
    for x in role:
        if x["utterance_id"] in set(full):
            rm.setdefault(x["utterance_id"], []).append((x["dialogue_id"], x["role"]))
    assert set(rm) == set(full) and all(len(v) == 1 and v[0][1] == "D-dev-select" for v in rm.values())
    dlg = {u: rm[u][0][0] for u in full}
    assert all(dlg[r["utterance_id"]] == r["dialogue_id"] for r in fx["rows"])
    assert len(set(full)) == 300 and len(set(fixed)) == 100 and set(fixed) < set(full) and len(set(dlg.values())) == 20
    new = [u for u in full if u not in set(fixed)]
    parts = {"FULL300": full, "FIXED100": fixed, "NEW200": new}
    tok = WhisperProcessor.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True).tokenizer
    gen = GenerationConfig.from_pretrained("/mnt/data/tungnx/whisper-large-v3", local_files_only=True)
    sp, sup, beg = special_ids(tok), list(gen.suppress_tokens or []), list(gen.begin_suppress_tokens or [])
    p4 = {r["utterance_id"]: r for r in json.loads((ROOT / P4PANEL).read_text())["rows"]}
    der = json.loads((ROOT / DERIVED).read_text())
    dr = {r["utterance_id"]: r for r in der["rows"]}
    rows = []
    for i, (pr, u) in enumerate(zip(full_p["rows"], full)):
        arel = f"results/inference_cf/p2_A_r1_L16/rows/{i:03d}.json"
        ar = json.loads((ROOT / arel).read_text())
        a = ar["systems"]["B0_AUTO"]
        assert ar["identity"] == u and ar["status"] == "ok"
        y_A = clean_teacher(a["tokens"], CB, EOS, sp)
        assert y_A == a["tokens"]
        fp64 = t0run.audio_fingerprint_64k(pr["audio_path"])
        assert fp64 == pr["audio_sha256"]
        row = {"full_index": i, "utterance_id": u, "dialogue_id": dlg[u], "partition": "FIXED100" if u in set(fixed) else "NEW200",
               "audio": {"path": pr["audio_path"], "fingerprint_64k_sizeprefixed": fp64, "full_sha256": t0run.audio_full_sha256(pr["audio_path"])},
               "B0_AUTO": {"path": arel, "byte_sha256": sha(ROOT / arel), "selector": "systems.B0_AUTO", "tokens_sha256": fp(a["tokens"]),
                           "terminated": a["terminated"], "length": len(a["tokens"])},
               "teacher": {"y_A_token_sha256": fp(y_A), "valid_mask_sha256": fp(valid_mask(y_A, sup, beg, sp, EOS)), "source": "clean_teacher(B0_AUTO)"}}
        if u in p4:
            q, d = p4[u], dr[u]
            row["fixed100"] = {"fixed_index": fixed.index(u), "B0_FORCED": q["B0_FORCED"], "A2": q["A2"], "A2_checkpoint": q["A2_checkpoint"],
                               "B0_sealed": q["B0_sealed"], "A2_sealed": q["A2_sealed"], "teacher": q["teacher"],
                               "PATH4_row": {"path": f"results/inference_cf/p2path4/run1/rows/{fixed.index(u):03d}.json",
                                             "byte_sha256": sha(ROOT / f"results/inference_cf/p2path4/run1/rows/{fixed.index(u):03d}.json")},
                               "audit_only_PATH4_mode": d["path4_mode"],
                               "G1A_derived_target": {"tokens_sha256": fp(d["G1A"]["tokens"]), "terminated": d["G1A"]["terminated"],
                                                      "length": len(d["G1A"]["tokens"]), "source": d["source"]}}
            assert q["teacher"]["y_A_token_sha256"] == row["teacher"]["y_A_token_sha256"]
        rows.append(row)
    order = fixed + new
    doc = {"schema": "p2_path5_panel_v1", "status": "PRE_OUTCOME_FREEZE", "role": "already-exposed D-dev-select development (NOT fresh validation)",
           "parent_full300": {"path": FULL, "byte_sha256": sha(ROOT / FULL), "schema": full_p["schema"], "ordering": full_p["ordering"]},
           "fixed100": {"path": FIXED, "byte_sha256": sha(ROOT / FIXED), "parent_panel_field": fx["parent_panel"],
                        "parent_panel_sha256_field": fx["parent_panel_sha256"]},
           "derived_G1A": {"path": DERIVED, "byte_sha256": sha(ROOT / DERIVED), "derived_hash": der["derived_hash"]},
           "dialogue_mapping_verification": {"path": ROLE, "columns_read": ["utterance_id", "dialogue_id", "role"], "filter": "FULL300 IDs only",
                                             "checks": "300 IDs, one row each, role D-dev-select, 20 dialogues x 15; fixed100 dialogues agree"},
           "partitions": {k: {"count": len(v), "dialogues": len({dlg[u] for u in v}), "ids": v, "membership_order_sha256": membership(k, v, dlg)}
                          for k, v in parts.items()},
           "partition_hash_convention": "sha256 sorted-key compact UTF8 JSON {partition, rows:[{utterance_id,dialogue_id}]}; FULL300/NEW200 FULL300 order; FIXED100 parent fixed100 order",
           "execution_order": order, "execution_order_sha256": fp(order),
           "rows": rows, "references_or_error_counts_used": False, "PATH5_outcomes_computed": False}
    return doc


def main() -> None:
    out = ROOT / OUT
    if out.exists():
        raise FileExistsError("panel exists; never overwrite")
    doc = build()
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "ids"} for k, v in doc["partitions"].items()}, indent=1))
    print("panel sha256", sha(out), "execution_order_sha256", doc["execution_order_sha256"])


if __name__ == "__main__":
    main()
