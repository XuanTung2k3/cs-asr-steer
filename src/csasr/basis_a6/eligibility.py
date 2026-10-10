"""A6-TT eligibility tables and common eligible manifests."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


def _quantile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values); pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi: return float(values[lo])
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def eligibility_table(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in records:
        groups[(row["model"], row["dataset"], row["side"], int(row["layer"]))].append(row)
    out = []
    for key, rows in sorted(groups.items()):
        raw = [r for r in rows if r.get("raw_eligible", r.get("eligible", False))]
        us = [r for r in rows if r.get("unique_shared_eligible", r.get("eligible", False))]
        cond = [r for r in rows if r.get("conditioning_eligible", False)]
        n_a = [int(r["n_A"]) for r in rows if r.get("n_A") is not None]
        n_b = [int(r["n_B"]) for r in rows if r.get("n_B") is not None]
        ranks = [int(r["rank"]) for r in rows if r.get("rank") is not None]
        out.append({"model": key[0], "dataset": key[1], "side": key[2], "layer": key[3],
                    "N_total": len(rows), "N_raw_eligible": len(raw),
                    "N_unique_shared_eligible": len(us), "N_conditioning_eligible": len(cond),
                    "raw_coverage": len(raw) / len(rows) if rows else 0.0,
                    "unique_shared_coverage": len(us) / len(rows) if rows else 0.0,
                    "conditioning_coverage": len(cond) / len(rows) if rows else 0.0,
                    "n_A_p10": _quantile(n_a, .1), "n_A_median": _quantile(n_a, .5), "n_A_p90": _quantile(n_a, .9),
                    "n_B_p10": _quantile(n_b, .1), "n_B_median": _quantile(n_b, .5), "n_B_p90": _quantile(n_b, .9),
                    "rank_p10": _quantile(ranks, .1), "rank_median": _quantile(ranks, .5), "rank_p90": _quantile(ranks, .9)})
    return out


def common_eligible_manifest(records: Iterable[dict[str, Any]], *, methods=("add_unique", "minus_shared", "unique_minus_shared")) -> dict:
    rows = list(records)
    by_group = defaultdict(lambda: defaultdict(set))
    for row in rows:
        group = (row["model"], row["dataset"], row["side"], int(row["layer"]))
        by_group[group][row["method"]].add(row["utterance_id"])
    return {"schema_version": "basis_a6_common_eligible_v1", "groups": {
        "|".join(map(str, key)): sorted(set.intersection(*(by_group[key].get(m, set()) for m in methods)))
        for key in sorted(by_group)}}
