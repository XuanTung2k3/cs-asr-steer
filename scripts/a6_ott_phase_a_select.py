#!/usr/bin/env python3
"""Aggregate and select A6-OTT Phase-A rows.

This module intentionally has a narrow input surface: it reads only the
accepted per-utterance rows below ``phase_a/rows``.  In particular, it never
walks the result namespace looking for compatible JSON, so Phase-B and
SEAME outcomes cannot accidentally enter Phase-A selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from csasr.evaluation import canonical, retention


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "a6_ott_upper_bound"
ROW_ROOT = RESULTS / "phase_a" / "rows"
OUT_ROOT = RESULTS / "search"
FAMILIES = (("add_unique", "encoder"), ("add_unique", "decoder"),
            ("conditioning_cs", "decoder"))
FLOOR = 0.5


def _sha(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def _metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute aggregate outcomes from the canonical transcript surface."""
    eligible = [r for r in rows if bool(r.get("eligible", True))]
    refs = [str(r["reference"]) for r in eligible]
    base = [str(r["baseline_hypothesis"]) for r in eligible]
    steer = [str(r["steered_hypothesis"]) for r in eligible]
    if not refs:
        return {"eligible_count": 0, "eligibility_rate": 0.0,
                "poi_net_utility": None, "pier_gain": None,
                "matrix_retention": None, "intervention_energy": None}
    bm = canonical.corpus_metrics(refs, base)
    sm = canonical.corpus_metrics(refs, steer)
    trans = canonical.correction_corruption(refs, base, steer)
    ret = retention.retention_report(refs, base, steer)
    energy = []
    for row in eligible:
        diag = row.get("perturbation_diagnostics") or {}
        if diag.get("edit_norm_sum") is not None:
            energy.append(float(diag["edit_norm_sum"]))
    return {
        "eligible_count": len(eligible),
        "total_count": len(rows),
        "eligibility_rate": len(eligible) / len(rows),
        "metrics": sm,
        "baseline_metrics": bm,
        "mer_gain": canonical.gain(bm.get("mer"), sm.get("mer")),
        "pier_gain": canonical.gain(bm.get("pier"), sm.get("pier")),
        "en_wer_gain": canonical.gain(bm.get("en_wer"), sm.get("en_wer")),
        "zh_cer_gain": canonical.gain(bm.get("zh_cer"), sm.get("zh_cer")),
        "poi_corrections": int(trans["corrections"]),
        "poi_corruptions": int(trans["corruptions"]),
        "poi_net_utility": int(trans["net_corrections"]),
        "matrix_retention": ret["matrix_zh"]["rate"],
        "embedded_retention": ret["embedded_en"]["rate"],
        "monolingual_retention": ret["monolingual"]["rate"],
        "transcript_change_rate": sum(a != b for a, b in zip(base, steer)) / len(steer),
        "intervention_energy": (sum(energy) / len(energy)) if energy else None,
        "target_logit_movement": None,
        "target_logit_status": "UNAVAILABLE_IN_PHASE_A_ROW_SCHEMA",
    }


def _load_rows(model: str) -> list[dict[str, Any]]:
    model_root = (ROW_ROOT / model).resolve()
    allowed_root = ROW_ROOT.resolve()
    if model_root.parent != allowed_root:
        raise RuntimeError("model row path escaped Phase-A allowlist")
    freeze = json.loads((RESULTS / "manifests" / "v2" / "SEARCH_FREEZE.json").read_text())
    allowed_ids = {uid for values in freeze["ids"].values() for uid in values}
    rows: list[dict[str, Any]] = []
    for path in sorted(model_root.glob("**/*.json")):
        resolved = path.resolve()
        if not resolved.is_relative_to(allowed_root):
            raise RuntimeError(f"row escaped Phase-A allowlist: {path}")
        row = json.loads(path.read_text())
        if row.get("phase") != "A" or row.get("regime") != "oracle_tt":
            raise RuntimeError(f"non-Phase-A row in selection directory: {path}")
        # Preserved acceptance artifacts may share this physical namespace;
        # they are outside the immutable v2 Search panel and are excluded by
        # canonical ID before any scientific aggregation.
        if str(row.get("utterance_id")) not in allowed_ids:
            continue
        if row.get("status") != "PASS":
            raise RuntimeError(f"non-PASS row in selection directory: {path}")
        if row.get("provenance", {}).get("fixed_direction_artifact") is not None:
            raise RuntimeError(f"fixed direction entered Phase-A selection: {path}")
        rows.append(row)
    return rows


def _rank_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    def val(name: str, default: float) -> float:
        v = item.get(name)
        return default if v is None else float(v)
    # Higher utility/gain/retention, then lower energy.
    return (val("poi_net_utility", -math.inf), val("pier_gain", -math.inf),
            val("matrix_retention", -math.inf), -val("intervention_energy", math.inf))


def aggregate(model: str) -> dict[str, Any]:
    rows = _load_rows(model)
    grouped: dict[tuple[str, str, int, float], list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        key = str(row["canonical_key"])
        if key in seen:
            raise RuntimeError(f"duplicate canonical key: {key}")
        seen.add(key)
        if row.get("model") != model or row.get("dataset") not in {"cs_dialogue", "ascend"}:
            raise RuntimeError("Phase-A row has wrong model or dataset")
        grouped[(str(row["method_family"]), str(row["side"]),
                 int(row["layer"]), float(row["rho"]))].append(row)

    table: list[dict[str, Any]] = []
    for (family, side, layer, rho), cell in sorted(grouped.items()):
        # Every candidate setting must cover the full 40-utterance Search panel.
        ids = {str(r["utterance_id"]) for r in cell}
        if len(ids) != 40:
            raise RuntimeError(f"incomplete Phase-A setting {family}/{side}/L{layer}/rho{rho}: {len(ids)}")
        agg = _metric_rows(cell)
        table.append({"family": family, "method_family": family, "side": side,
                      "layer": layer, "rho": rho, "decode_mode": "greedy",
                      **agg, "source_row_count": len(cell),
                      "source_row_keys_hash": _sha(sorted(ids))})

    selected: dict[str, list[dict[str, Any]]] = {}
    for family, side in FAMILIES:
        candidates = [x for x in table if x["family"] == family and x["side"] == side]
        rankable = [x for x in candidates if x["eligibility_rate"] >= FLOOR]
        rankable.sort(key=_rank_key, reverse=True)
        for i, x in enumerate(rankable):
            x["rank_within_family"] = i + 1
            x["rankable"] = True
        for x in candidates:
            x.setdefault("rankable", False)
        selected[f"{family}/{side}"] = rankable[:2]

    rule = {
        "name": "a6_ott_phase_a_selection_v1",
        "eligibility_floor": FLOOR,
        "ranking": ["poi_net_utility", "pier_gain", "matrix_retention",
                     "lower_intervention_energy"],
        "source": "docs/current/A6_OTT_SELECTION_PROTOCOL.md §3",
        "input_allowlist": "results/a6_ott_upper_bound/phase_a/rows/<model>/**",
        "forbidden_inputs": ["confirm", "phase_b", "seame", "transfer"],
    }
    payload = {
        "schema_version": "a6_ott_phase_a_selection_v1",
        "status": "PASS",
        "phase": "A",
        "model": model,
        "data_role": "search",
        "selection_only_phase_a": True,
        "git_head": _git_head(),
        "selection_rule": rule,
        "selection_rule_hash": _sha(rule),
        "logical_rows": len(rows),
        "settings": table,
        "top2_by_family": selected,
        "source_allowlist_asserted": True,
        "confirmation_and_seame_outcomes_loaded": False,
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("whisper", "qwen3_asr_1p7b"), action="append")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for model in args.model or ["whisper", "qwen3_asr_1p7b"]:
        payload = aggregate(model)
        name = "WHISPER_SELECTION.json" if model == "whisper" else "QWEN_SELECTION.json"
        (OUT_ROOT / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(name, "logical_rows=", payload["logical_rows"])
        for family, rows in payload["top2_by_family"].items():
            print(family, [(r["layer"], r["rho"], r["poi_net_utility"]) for r in rows])
        rescue = None
        if model == "qwen3_asr_1p7b":
            diagnosis = {
                "schema_version": "a6_ott_qwen_phase_a_diagnosis_v1",
                "phase": "A",
                "model": model,
                "data_role": "search",
                "selection_only_phase_a": True,
                "target_logit_note": "Target-logit diagnostics are unavailable in the physical row schema; no target-logit outcome was inferred.",
                "by_family_layer_rho": {
                    family: [
                        {
                            "layer": item["layer"], "rho": item["rho"],
                            "hidden_perturbation": item["intervention_energy"],
                            "target_logit_movement": item["target_logit_movement"],
                            "target_logit_status": item["target_logit_status"],
                            "transcript_change_rate": item["transcript_change_rate"],
                            "corrections": item["poi_corrections"],
                            "corruptions": item["poi_corruptions"],
                            "poi_net_utility": item["poi_net_utility"],
                            "pier": (item.get("metrics") or {}).get("pier"),
                            "pier_gain": item["pier_gain"],
                            "matrix_retention": item["matrix_retention"],
                            "eligibility_rate": item["eligibility_rate"],
                        }
                        for item in payload["settings"]
                        if item["family"] == family
                    ]
                    for family, _side in FAMILIES
                },
                "confirmation_and_seame_outcomes_loaded": False,
            }
            (OUT_ROOT / "QWEN_PHASE_A_DIAGNOSIS.json").write_text(
                json.dumps(diagnosis, indent=2) + "\n")
            rescue = {
                "schema_version": "a6_ott_qwen_rescue_decision_v1",
                "status": "NOT_TRIGGERED" if any(
                    any((r.get("poi_net_utility") or 0) > 0 for r in rows)
                    for rows in payload["top2_by_family"].values()) else "TRIGGERED",
                "trigger_rule": "no Qwen family has selected setting with positive Search poi_net_utility at eligibility >= 0.5",
                "selection_source": "QWEN_SELECTION.json",
                "confirmation_and_seame_outcomes_loaded": False,
            }
            (OUT_ROOT / "QWEN_RESCUE_DECISION.json").write_text(
                json.dumps(rescue, indent=2) + "\n")
            print("qwen_rescue", rescue["status"])


if __name__ == "__main__":
    main()
