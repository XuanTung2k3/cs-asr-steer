"""Exposure registry reader: original documented exposure plus the append-only addenda.

``docs/current/DATA_EXPOSURE.md`` (the prose ledger) is byte-pinned by the frozen SRD2-G0 contract, so exposures
after that freeze are registered in ``docs/current/DATA_EXPOSURE_ADDENDA.json`` instead. Any new D-dev-select
sample selector (DIR-SPRINT0 first) must exclude ``exposed_ids()``. That set is:

* the machine-readable documented-exposure registry inventoried at the SRD2-G0 freeze (the 300 roster IDs with
  non-empty exclusion reasons in ``docs/inference_cf/SRD2_G0_POPULATION.json``), and
* every utterance ID listed in an addendum entry (A1 = the 400 SRD2-G0 utterances).

Every input is hash-checked against the addenda file. A mismatch raises, so a selector can never silently run on a
stale or edited ledger. Identity metadata only: no audio, reference or outcome is read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ADDENDA = "docs/current/DATA_EXPOSURE_ADDENDA.json"
SCHEMA = "data_exposure_addenda_v1"


class ExposureRegistryError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ids_digest(ids: list[str]) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def load_addenda(root: Path) -> dict:
    root = Path(root)
    add = json.loads((root / ADDENDA).read_text(encoding="utf-8"))
    if add.get("schema") != SCHEMA:
        raise ExposureRegistryError("unknown addenda schema")
    for key in ("base_ledger", "base_machine_registry"):
        ref = add[key]
        if _sha(root / ref["path"]) != ref["sha256"]:
            raise ExposureRegistryError(f"{key} changed since the addenda were written: {ref['path']}")
    seen = set()
    for e in add["entries"]:
        if e["addendum_id"] in seen:
            raise ExposureRegistryError(f"duplicate addendum {e['addendum_id']}")
        seen.add(e["addendum_id"])
        ids = list(e["utterance_ids"])
        if len(ids) != e["count"] or len(set(ids)) != len(ids) or _ids_digest(ids) != e["utterance_ids_hash"]:
            raise ExposureRegistryError(f"addendum {e['addendum_id']} ID list does not match its hash/count")
        pop = e.get("population_file")
        if pop and _sha(root / pop["path"]) != pop["sha256"]:
            raise ExposureRegistryError(f"addendum {e['addendum_id']} population file changed")
    return add


def exposed_ids(root: Path) -> dict:
    """Return {"ids": frozenset, "by_source": {name: count}, "addenda_sha256": ..., "base_ledger_sha256": ...}."""
    root = Path(root)
    add = load_addenda(root)
    reg = json.loads((root / add["base_machine_registry"]["path"]).read_text(encoding="utf-8"))
    documented = {r["utterance_id"] for r in reg["roster"] if r["exclusion_reasons"]}
    by_source, ids = {"documented_registry_at_srd2_freeze": len(documented)}, set(documented)
    for e in add["entries"]:
        by_source[e["addendum_id"]] = len(e["utterance_ids"])
        ids |= set(e["utterance_ids"])
    return {"ids": frozenset(ids), "by_source": by_source, "addenda_sha256": _sha(root / ADDENDA),
            "base_ledger_sha256": add["base_ledger"]["sha256"], "role": reg["role"]}
