#!/usr/bin/env python
"""DIR-SPRINT0 identity-only population freeze (CPU; never imports a model, evaluator or reference column).

* Prospective outcome population: 12 D-dev-select utterances per sorted dialogue (20 dialogues -> 240), ranked by
  SHA256("DIR-SPRINT0-population-v1|240924|" + UID) then UID, after excluding every ID returned by the hash-checked
  exposure registry ``csasr.inference_cf.exposure_registry.exposed_ids()`` (documented FULL300 registry + SRD2-G0 400
  addendum) and any utterance without readable mono 16 kHz audio.
* D5 calibration bank: ALL FULL300 utterances (the documented-exposure registry), unlabeled, no filtering. It is
  disjoint from the prospective 240 by construction. It is used only for reference-free concept-prototype
  construction, never for an outcome.

Role parquet reads use the six-column metadata allowlist only (no transcript, label or outcome column).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

ROLE = Path("/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/manifests/roles/role_D-dev-select.parquet")
OUT = ROOT / "docs/inference_cf/DIR_SPRINT0_POPULATION.json"
SEED = 240924
TAG = "DIR-SPRINT0-population-v1"
PER_DIALOGUE = 12
COLUMNS = ("utterance_id", "dialogue_id", "role", "audio_path", "audio_sha256", "duration_sec")
FULL300_HASH = "sha256:22a160bdab929e54dd2bbe41a51bd21fd7dde04e63850d000cf77bb2c245b2fc"   # SRD2_G0_POPULATION FULL300_ids_hash


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def digest(obj) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                                                 allow_nan=False).encode()).hexdigest()


def order_key(uid: str) -> str:
    return hashlib.sha256(f"{TAG}|{SEED}|{uid}".encode()).hexdigest()


def audio_record(r: dict) -> dict:
    import soundfile as sf
    info = sf.info(r["audio_path"])
    return {"utterance_id": r["utterance_id"], "dialogue_id": r["dialogue_id"], "audio_path": r["audio_path"],
            "audio_full_sha256": sha(r["audio_path"]), "role_audio_sha256": r["audio_sha256"],
            "source_duration_sec": float(r["duration_sec"]), "sample_rate": int(info.samplerate),
            "channels": int(info.channels), "frames": int(info.frames)}


def build() -> dict:
    import pyarrow.parquet as pq
    from csasr.inference_cf.exposure_registry import load_addenda, exposed_ids
    rows = pq.read_table(ROLE, columns=list(COLUMNS)).to_pylist()
    if len({r["utterance_id"] for r in rows}) != len(rows) or any(r["role"] != "D-dev-select" for r in rows):
        raise ValueError("role identity failure")
    add = load_addenda(ROOT)
    exp = exposed_ids(ROOT)
    reg = json.loads((ROOT / add["base_machine_registry"]["path"]).read_text())
    documented = sorted(r["utterance_id"] for r in reg["roster"] if r["exclusion_reasons"])
    if digest(documented) != FULL300_HASH or len(documented) != 300:
        raise ValueError("documented registry is not the FULL300 identity set")
    addendum_of = {}
    for e in add["entries"]:
        for u in e["utterance_ids"]:
            addendum_of.setdefault(u, []).append(f"exposure_addendum:{e['addendum_id']}")
    roster, groups = [], {}
    for r in sorted(rows, key=lambda x: x["utterance_id"]):
        uid = r["utterance_id"]
        reasons = []
        if uid in set(documented):
            reasons.append("exposure_registry:documented_FULL300")
        reasons += addendum_of.get(uid, [])
        if uid in exp["ids"] and not reasons:
            raise ValueError(f"exposed id {uid} without a recorded source")
        if not Path(r["audio_path"]).is_file():
            reasons.append("missing_audio")
        roster.append({"utterance_id": uid, "dialogue_id": r["dialogue_id"], "exclusion_reasons": reasons,
                       "order_key": order_key(uid)})
        if not reasons:
            groups.setdefault(r["dialogue_id"], []).append(r)
    if set(exp["ids"]) - {x["utterance_id"] for x in roster if x["exclusion_reasons"]}:
        raise ValueError("an exposed id is not excluded")
    if len(groups) != 20 or any(len(v) < PER_DIALOGUE for v in groups.values()):
        raise ValueError("cannot select 12 per dialogue")
    selected = []
    for dialogue in sorted(groups):
        ranked = sorted(groups[dialogue], key=lambda x: (order_key(x["utterance_id"]), x["utterance_id"]))
        picked = []
        for r in ranked:
            rec = audio_record(r)
            if rec["sample_rate"] != 16000 or rec["channels"] != 1 or rec["frames"] <= 0:
                raise ValueError(f"selected audio is not mono 16 kHz: {r['utterance_id']}")
            picked.append(rec)
            if len(picked) == PER_DIALOGUE:
                break
        for rec in picked:
            selected.append({"canonical_index": len(selected), **rec, "order_key": order_key(rec["utterance_id"])})
    by_id = {r["utterance_id"]: r for r in rows}
    bank = []
    for uid in sorted(documented, key=lambda u: (by_id[u]["dialogue_id"], u)):
        rec = audio_record(by_id[uid])
        if rec["sample_rate"] != 16000 or rec["channels"] != 1 or rec["frames"] <= 0:
            raise ValueError(f"bank audio is not mono 16 kHz: {uid}")
        bank.append({"bank_index": len(bank), **rec})
    sel_ids = {r["utterance_id"] for r in selected}
    if sel_ids & set(exp["ids"]) or sel_ids & set(documented) or len(selected) != 240:
        raise ValueError("selection overlaps the exposure registry")
    manifest = {
        "schema": "dir_sprint0_population_v1", "role": "D-dev-select", "seed": SEED, "selection_tag": TAG,
        "source": str(ROLE), "source_sha256": sha(ROLE), "source_columns_read": list(COLUMNS),
        "selection_rule": "12 per sorted dialogue; smallest SHA256(tag|seed|UID), UID tie-break; identity only",
        "exclusion_registry": {"reader": "csasr.inference_cf.exposure_registry.exposed_ids",
                               "addenda_sha256": exp["addenda_sha256"], "base_ledger_sha256": exp["base_ledger_sha256"],
                               "base_machine_registry": add["base_machine_registry"], "by_source": exp["by_source"],
                               "exposed_ids_count": len(exp["ids"]), "exposed_ids_hash": digest(sorted(exp["ids"]))},
        "roster_count": len(roster), "eligible_count": sum(not r["exclusion_reasons"] for r in roster),
        "excluded_count": sum(bool(r["exclusion_reasons"]) for r in roster),
        "eligible_per_dialogue": {d: len(v) for d, v in sorted(groups.items())},
        "roster": roster, "roster_hash": digest(roster),
        "selected": selected, "selected_hash": digest(selected),
        "selected_ids_hash": digest([r["utterance_id"] for r in selected]),
        "selected_total_duration_sec": sum(r["source_duration_sec"] for r in selected),
        "selected_longer_than_30s": sum(r["source_duration_sec"] > 30 for r in selected),
        "calibration_bank": {"definition": "ALL FULL300 documented-exposure utterances (unlabeled; D5 concept prototypes only)",
                             "ids_hash_sorted": digest(documented), "count": len(bank), "rows": bank,
                             "rows_hash": digest(bank), "total_duration_sec": sum(r["source_duration_sec"] for r in bank),
                             "overlap_with_selected": 0, "correctness_filtering": False,
                             "use": "reference-free L16 DG-02 state capture + phone evidence; never an outcome"},
        "selection_uses_reference_or_outcome": False,
        "exposure_limit": "known exposures (documented registry + append-only addenda); new utterances from the same 20 "
                          "previously used dialogues; not independent confirmation"}
    manifest["manifest_hash"] = digest(manifest)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    path = Path(a.out)
    if path.exists():
        raise FileExistsError("never overwrite a frozen population")
    res = build()
    path.write_text(json.dumps(res, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({k: res[k] for k in ("roster_count", "eligible_count", "excluded_count", "selected_hash",
                                          "selected_ids_hash", "manifest_hash", "selected_total_duration_sec",
                                          "selected_longer_than_30s")}))
    print(json.dumps({k: res["calibration_bank"][k] for k in ("count", "ids_hash_sorted", "rows_hash", "total_duration_sec")}))


if __name__ == "__main__":
    main()
