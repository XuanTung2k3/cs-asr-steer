"""Which synthetic source audio has already been seen, and by whom.

A confirmatory gate is only confirmatory once. Job 38573's synthetic gate results
were read during review -- the per-family medians, the p90s, the bias -- and the
selection rule was then changed in response to them. Those 100 items can still
be scored, but they can no longer answer "does the configuration we just chose
hold on data we had not looked at". Reusing them would be the oldest way to make
a gate pass: tune, look, tune again, report the last number.

So exposure is recorded rather than remembered:

* every rendered set writes its source utterance ids and a fingerprint into
  `synthetic/exposure_ledger.json`;
* evaluating Gate A **exposes** the gate generation it read, so the next
  evaluation builds a fresh one from sources no earlier generation used;
* sets rendered before this ledger existed are bootstrapped from the item tables
  they left on disk and recorded as exposed, because they were.

The development set is deliberately *not* filtered by exposure. Development
evidence is meant to be reused -- it is what selects the configuration, and a dev
set that changed every run would make the selection unreproducible. Only the gate
pool excludes exposed sources, and `partition_sources` already keeps the two
pools disjoint, so no dev item can reach a gate generation anyway.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from ...utils.hashing import sha256_strings
from .synthetic import source_ids

LEDGER_FILE = "synthetic/exposure_ledger.json"
LEDGER_SCHEMA = "lss_synthetic_exposure_v1"

#: item tables earlier runs left behind, and the purpose each one records.
#: Scanned once so a set rendered before the ledger existed is still accounted
#: for instead of quietly reappearing as a "fresh" gate.
LEGACY_ITEM_TABLES = {
    "metrics/l1b_synthetic_dev_items.parquet": "dev",
    "metrics/l1b_synthetic_gate_items.parquet": "gate",
}

BOOTSTRAP_REASON = (
    "rendered before the exposure ledger existed; its scores were read during "
    "the job 38573 review and informed the selection rule, so it is exposed"
)


class GenerationConflictError(RuntimeError):
    """A generation number was reused for a different immutable item set."""


class ExposureLedgerError(RuntimeError):
    """The durable exposure history is unreadable or internally inconsistent."""


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def ledger_path(artifacts_root: str | Path) -> Path:
    return Path(artifacts_root) / LEDGER_FILE


def empty_ledger() -> dict[str, Any]:
    return {"schema_version": LEDGER_SCHEMA, "generations": [],
            "updated_at": _now()}


def load_ledger(artifacts_root: str | Path) -> dict[str, Any]:
    """The ledger on disk, or an empty one when it has never existed.

    A corrupt existing ledger is not equivalent to no history: treating it as
    empty would recycle source recordings and generation numbers whose earlier
    results may already have been inspected. Fail closed and leave the bytes in
    place for recovery.
    """
    path = ledger_path(artifacts_root)
    if not path.is_file():
        return empty_ledger()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExposureLedgerError(f"unreadable exposure ledger {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != LEDGER_SCHEMA:
        raise ExposureLedgerError(
            f"unsupported or malformed exposure ledger {path}; expected "
            f"schema {LEDGER_SCHEMA!r}")
    generations = payload.get("generations")
    if not isinstance(generations, list):
        raise ExposureLedgerError(f"exposure ledger {path} has no generation list")
    seen: set[tuple[str, int]] = set()
    for offset, entry in enumerate(generations):
        if not isinstance(entry, dict):
            raise ExposureLedgerError(
                f"exposure ledger {path} generation {offset} is not an object")
        try:
            key = (str(entry["purpose"]), int(entry["generation"]))
            source_values = entry["source_utterance_ids"]
            pair_values = entry.get("pair_ids", [])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExposureLedgerError(
                f"exposure ledger {path} generation {offset} is malformed: {exc}") \
                from exc
        if not isinstance(source_values, list) or not isinstance(pair_values, list):
            raise ExposureLedgerError(
                f"exposure ledger {path} generation {key} has malformed ID lists")
        sources = [str(value) for value in source_values]
        if key in seen:
            raise ExposureLedgerError(
                f"exposure ledger {path} repeats immutable generation {key}")
        seen.add(key)
        expected = sha256_strings(sorted(sources))
        if entry.get("source_fingerprint") != expected:
            raise ExposureLedgerError(
                f"exposure ledger {path} generation {key} has a source "
                "fingerprint mismatch")
    return payload


def save_ledger(artifacts_root: str | Path, ledger: Mapping[str, Any]) -> Path:
    from ...nat5h.statusing import atomic_write_json

    path = ledger_path(artifacts_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**dict(ledger), "updated_at": _now()}
    atomic_write_json(path, payload)
    return path


def exposed_source_ids(ledger: Mapping[str, Any]) -> set[str]:
    """Every source utterance any recorded generation reserved.

    This deliberately includes active, abandoned, and exposed generations.
    Exposure is per *source recording*, not per rendered pair: splicing the same
    Mandarin clip against a different English one does not make it unseen, and an
    interrupted attempt does not return its sources to the confirmatory pool.
    """
    out: set[str] = set()
    for generation in ledger.get("generations") or []:
        out.update(str(x) for x in (generation.get("source_utterance_ids") or []))
    return out


def generations_for(ledger: Mapping[str, Any], purpose: str) -> list[dict[str, Any]]:
    return [g for g in (ledger.get("generations") or [])
            if str(g.get("purpose")) == str(purpose)]


def current_generation(ledger: Mapping[str, Any], purpose: str = "gate") -> int:
    """The generation number a run should use for ``purpose``.

    An unexposed generation is reused -- rendering it twice from the same sources
    would be waste, not freshness. Once it has been read by a gate evaluation the
    next number is returned, so the following evaluation cannot see it again.
    """
    existing = generations_for(ledger, purpose)
    if not existing:
        return 1
    unexposed = [int(g["generation"]) for g in existing
                 if not g.get("exposed") and not g.get("abandoned")]
    if unexposed:
        return max(unexposed)
    return max(int(g["generation"]) for g in existing) + 1


def next_generation(ledger: Mapping[str, Any], purpose: str = "gate") -> int:
    """A never-before-used generation number, irrespective of lifecycle state."""
    existing = generations_for(ledger, purpose)
    return 1 if not existing else max(int(g["generation"]) for g in existing) + 1


def find_generation(ledger: Mapping[str, Any], purpose: str,
                    generation: int) -> dict[str, Any] | None:
    for entry in generations_for(ledger, purpose):
        if int(entry.get("generation", -1)) == int(generation):
            return entry
    return None


def bootstrap(artifacts_root: str | Path, *, ledger: Mapping[str, Any] | None = None
              ) -> dict[str, Any]:
    """Record sets that were rendered before the ledger existed.

    Reads only the item tables those runs left on disk, so a root with no history
    bootstraps to an empty ledger and a root that already has entries is left
    alone.
    """
    root = Path(artifacts_root)
    book = dict(ledger if ledger is not None else load_ledger(root))
    book.setdefault("generations", [])
    if any(int(g.get("generation", -1)) == 0 for g in book["generations"]):
        return book
    added: list[dict[str, Any]] = []
    for relative, purpose in sorted(LEGACY_ITEM_TABLES.items()):
        path = root / relative
        if not path.is_file():
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception:                                   # unreadable leftover
            continue
        sources = sorted(source_ids(frame))
        if not sources:
            continue
        added.append(_entry(purpose=purpose, generation=0, sources=sources,
                            pair_ids=sorted(frame["pair_id"].astype(str))
                            if "pair_id" in frame else [],
                            reason=BOOTSTRAP_REASON, exposed=True,
                            recorded_from=str(path)))
    book["generations"] = list(book["generations"]) + added
    return book


def _entry(*, purpose: str, generation: int, sources: Sequence[str],
           pair_ids: Sequence[str], reason: str, exposed: bool,
           recorded_from: str = "",
           alignment_request_sha256: str | None = None,
           item_set_sha256: str | None = None) -> dict[str, Any]:
    return {
        "purpose": str(purpose),
        "generation": int(generation),
        "recorded_at": _now(),
        "reason": str(reason),
        "exposed": bool(exposed),
        "source_utterance_ids": [str(s) for s in sources],
        "source_fingerprint": sha256_strings(sorted(str(s) for s in sources)),
        "pair_ids": [str(p) for p in pair_ids],
        "recorded_from": recorded_from,
        "alignment_request_sha256": alignment_request_sha256,
        "item_set_sha256": item_set_sha256,
    }


def record_generation(artifacts_root: str | Path, frame: pd.DataFrame, *,
                      purpose: str, generation: int, reason: str,
                      exposed: bool = False,
                      alignment_request_sha256: str | None = None,
                      item_set_sha256: str | None = None,
                      ledger: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Add one immutable generation record, idempotently for identical bytes.

    Lifecycle fields (``exposed``/``abandoned`` and their timestamps) may change,
    but the source IDs, pair IDs, source fingerprint, and alignment-request
    fingerprint attached to a generation number may never be replaced. A retry
    needing different items must allocate ``next_generation`` instead.
    """
    root = Path(artifacts_root)
    book = dict(ledger if ledger is not None else load_ledger(root))
    book.setdefault("generations", [])
    entry = _entry(purpose=purpose, generation=generation,
                   sources=sorted(source_ids(frame)),
                   pair_ids=sorted(frame["pair_id"].astype(str))
                   if frame is not None and "pair_id" in getattr(frame, "columns", [])
                   else [],
                   reason=reason, exposed=exposed,
                   alignment_request_sha256=alignment_request_sha256,
                   item_set_sha256=item_set_sha256)
    previous = find_generation(book, purpose, generation)
    if previous is not None:
        immutable = ("source_fingerprint", "source_utterance_ids", "pair_ids")
        differing = [key for key in immutable if previous.get(key) != entry.get(key)]
        for fingerprint in ("alignment_request_sha256", "item_set_sha256"):
            old_value = previous.get(fingerprint)
            new_value = entry.get(fingerprint)
            if old_value is not None and old_value != new_value:
                differing.append(fingerprint)
        if differing:
            raise GenerationConflictError(
                f"{purpose} generation {generation} is immutable and the new "
                f"item set differs in {sorted(set(differing))}; allocate "
                "next_generation() instead")
        # Idempotent registration may fill a fingerprint absent from a legacy
        # entry, but it cannot replace one. Preserve all lifecycle history.
        for fingerprint in ("alignment_request_sha256", "item_set_sha256"):
            if previous.get(fingerprint) is None and entry.get(fingerprint) is not None:
                previous[fingerprint] = entry[fingerprint]
        if exposed and not previous.get("exposed"):
            previous["exposed"] = True
            previous["exposed_at"] = _now()
            previous["exposed_reason"] = str(reason)
        save_ledger(root, book)
        return book
    book["generations"] = list(book["generations"]) + [entry]
    save_ledger(root, book)
    return book


def abandon_unexposed(artifacts_root: str | Path, *, purpose: str,
                      reason: str) -> dict[str, Any]:
    """Retire interrupted attempts without changing their immutable identity."""
    root = Path(artifacts_root)
    book = load_ledger(root)
    changed = False
    for entry in generations_for(book, purpose):
        if entry.get("exposed") or entry.get("abandoned"):
            continue
        entry["abandoned"] = True
        entry["abandoned_at"] = _now()
        entry["abandoned_reason"] = str(reason)
        changed = True
    if changed:
        save_ledger(root, book)
    return book


def mark_exposed(artifacts_root: str | Path, *, purpose: str, generation: int,
                 reason: str) -> dict[str, Any]:
    """Burn a generation: it has been read and can no longer confirm anything."""
    root = Path(artifacts_root)
    book = load_ledger(root)
    entry = find_generation(book, purpose, generation)
    if entry is None:
        return book
    entry["exposed"] = True
    entry["exposed_at"] = _now()
    entry["exposed_reason"] = str(reason)
    save_ledger(root, book)
    return book


def eligible_sources(pool: pd.DataFrame, ledger: Mapping[str, Any], *,
                     key: str = "utterance_id") -> tuple[pd.DataFrame, dict[str, Any]]:
    """The part of a source pool no recorded generation has used.

    Returns the filtered pool and a report naming how many were excluded, so a
    shrinking pool is visible rather than inferred from a pair count.
    """
    exposed = exposed_source_ids(ledger)
    if pool is None or not len(pool) or key not in getattr(pool, "columns", []):
        return pool, {"pool": 0 if pool is None else int(len(pool)),
                      "excluded": 0, "eligible": 0 if pool is None else int(len(pool)),
                      "exposed_sources_known": len(exposed)}
    mask = ~pool[key].astype(str).isin(exposed)
    filtered = pool[mask].reset_index(drop=True)
    return filtered, {
        "pool": int(len(pool)),
        "excluded": int((~mask).sum()),
        "eligible": int(len(filtered)),
        "exposed_sources_known": len(exposed),
        "exposed_fingerprint": sha256_strings(sorted(exposed)) if exposed else "",
    }


def assert_unexposed(frame: pd.DataFrame, ledger: Mapping[str, Any], *,
                     allow: Iterable[str] = ()) -> dict[str, Any]:
    """Fail loudly if a rendered set reuses source audio that was already read.

    ``allow`` names generations' sources that are legitimately reused -- the
    development set's own previous rendering, which is meant to be stable.
    """
    used = source_ids(frame)
    forbidden = (exposed_source_ids(ledger) - {str(a) for a in allow}) & used
    report = {"sources": len(used), "reused_exposed": sorted(forbidden)[:10],
              "fresh": not forbidden}
    if forbidden:
        raise AssertionError(
            f"{len(forbidden)} source recording(s) in this gate set were already "
            "read by an earlier generation, so it cannot serve as a fresh "
            f"confirmatory gate (e.g. {sorted(forbidden)[:5]})")
    return report
