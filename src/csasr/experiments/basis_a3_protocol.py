"""Frozen BASIS-A3 protocol, panel and geometry utilities.

This module is deliberately CPU-safe.  It freezes decisions and consumes
already-frozen decoder direction artifacts; no model loading or decoding is
performed here.
"""
from __future__ import annotations

import csv
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "results/basis_a3_raw_cond_scope_depth"
DATA_ROOT = Path("/mnt/data/tungnx/seame_accent_stress_pilot")
A2 = REPO / "results/basis_frozen_layer_atlas"
ENCODER_LAYERS = tuple(range(32))
DECODER_LAYERS = tuple(range(32))
COND_LAYERS = (24, 26, 27, 31)
R1_RHO = (0.5,)
R2_RHO = (0.25, 1.0)
ALL_RHO = (0.25, 0.5, 1.0)
SCOPES = ("global", "oracle_local")
DATASETS = ("cs_dialogue", "seame_dev_man", "seame_dev_sge")
FORBIDDEN_SPLITS = ("D-dev-confirm", "D-test")
SEAME_PANEL_FREEZE = DATA_ROOT / "manifests/eval_100.freeze.json"
SEAME_PANEL_CSV = DATA_ROOT / "manifests/eval_100.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def canonical_sha(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False, default=str) + "\n", encoding="utf-8")


def _cs_panel() -> dict:
    """Reconstitute exactly the prior DG-03–DG-07 300-utterance key set."""
    prior = json.loads((REPO / "results/dg04/results/B0.json").read_text())
    ids = list(prior["texts"])
    role = Path("/mnt/data/tungnx/cs-asr-steer/artifacts_dialogue_v2r3/manifests/roles/role_D-dev-select.parquet")
    import pandas as pd
    frame = pd.read_parquet(role)
    frame["utterance_id"] = frame["utterance_id"].astype(str)
    frame = frame.set_index("utterance_id").loc[ids].reset_index()
    rows = []
    for row in frame.to_dict(orient="records"):
        uid = str(row["utterance_id"])
        rows.append({"utterance_id": uid, "dialogue_id": str(row.get("dialogue_id", "")),
                     "audio_path": str(row["audio_path"]),
                     "duration_sec": float(row["duration_sec"]),
                     "reference": str(row["transcript_raw"]),
                     "data_role": "D-dev-select"})
    return {"schema_version": "basis_a3_panel_v1", "dataset": "cs_dialogue",
            "data_role": "D-dev-select", "selection": "exact prior DG-03–DG-07 300-ID key set",
            "rows": rows, "count": len(rows), "fingerprint": canonical_sha(rows),
            "source_baseline": "results/dg04/results/B0.json:texts"}


def _seame_panel(split: str) -> dict:
    if not SEAME_PANEL_CSV.is_file() or not SEAME_PANEL_FREEZE.is_file():
        raise FileNotFoundError("frozen SEAME eval_100 panel is unavailable")
    freeze = json.loads(SEAME_PANEL_FREEZE.read_text())
    rows = []
    with SEAME_PANEL_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            uid = row["utt_id"]
            rows.append({
                "utterance_id": uid,
                "dialogue_id": f"SEAME:{split}:{row['speaker_id']}",
                "audio_path": str(DATA_ROOT / "audio/orig" / f"{uid}.wav"),
                "duration_sec": float(row["duration"]),
                "reference": row["normalized_transcript"],
                "data_role": split,
                "speaker_id": row["speaker_id"],
                "language_runs": ast.literal_eval(row["language_runs"]),
                "target_english_text": row["target_english_text"],
                "target_span_indices": json.loads(row["target_span_indices"]),
                "selection_reason": row["selection_reason"],
                "panel_source_row": int(row["source_row"]),
            })
    return {"schema_version": "basis_a3_panel_v1", "dataset": split,
            "data_role": f"SEAME-{split}",
            "selection": "reused frozen canonical SEAME eval_100 panel; no outcome selection",
            "target_count": 300, "shortfall_reason": "repository's frozen panel is the only accepted SEAME artifact",
            "source_freeze": str(SEAME_PANEL_FREEZE), "source_freeze_sha256": sha256_file(SEAME_PANEL_FREEZE),
            "source_freeze_n": int(freeze["n"]), "rows": rows, "count": len(rows),
            "fingerprint": canonical_sha(rows)}


def freeze_panels(out_dir: Path = RESULTS / "panels") -> dict:
    panels = {"cs_dialogue": _cs_panel(), "seame_dev_man": _seame_panel("dev_man"),
              "seame_dev_sge": _seame_panel("dev_sge")}
    for name, panel in panels.items():
        write_json(out_dir / f"{name}_300.json", panel)
    return panels


def geometry_rows(direction_dir: Path = A2 / "directions") -> list[dict[str, Any]]:
    rows = []
    for layer in DECODER_LAYERS:
        raw = np.asarray(np.load(direction_dir / f"raw_L{layer}.npy"), dtype=np.float64)
        cond = np.asarray(np.load(direction_dir / f"conditioning_L{layer}.npy"), dtype=np.float64)
        if raw.ndim != 1 or cond.ndim != 1 or raw.shape != cond.shape:
            raise ValueError(f"layer {layer}: Raw/Conditioning shape mismatch")
        if not np.isfinite(raw).all() or not np.isfinite(cond).all():
            raise ValueError(f"layer {layer}: non-finite direction")
        nr, nc = float(np.linalg.norm(raw)), float(np.linalg.norm(cond))
        if nr == 0.0 or nc == 0.0:
            raise ValueError(f"layer {layer}: degenerate direction")
        cos = float(np.dot(raw, cond) / (nr * nc))
        cos = float(np.clip(cos, -1.0, 1.0))
        raw_hat, cond_hat = raw / nr, cond / nc
        unit_l2 = float(np.linalg.norm(raw_hat - cond_hat))
        identity = math.sqrt(max(0.0, 2.0 - 2.0 * cos))
        if not math.isclose(unit_l2, identity, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"layer {layer}: normalized L2 identity failed")
        rows.append({"layer": layer, "cos_raw_cond": cos,
                     "angle_deg": float(np.degrees(np.arccos(cos))),
                     "raw_l2": float(np.linalg.norm(raw - cond)),
                     "unit_l2": unit_l2, "raw_norm": nr, "cond_norm": nc,
                     "raw_artifact": str((direction_dir / f"raw_L{layer}.npy").relative_to(REPO)),
                     "conditioning_artifact": str((direction_dir / f"conditioning_L{layer}.npy").relative_to(REPO))})
    return rows


def write_geometry(out_dir: Path = RESULTS / "geometry") -> dict:
    rows = geometry_rows()
    payload = {"schema_version": "basis_a3_raw_cond_decoder_geometry_v1",
               "direction_source": str(A2 / "directions.json"), "layers": rows,
               "validation": {"finite": True, "same_dimension": True,
                              "unit_l2_identity": True}}
    write_json(out_dir / "raw_cond_decoder_geometry.json", payload)
    import csv as _csv
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "raw_cond_decoder_geometry.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["layer", "cos_raw_cond", "angle_deg", "raw_l2", "unit_l2", "raw_norm", "cond_norm"]
        w = _csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in rows)
    return payload


def select_r2_layers(result_root: Path = RESULTS) -> dict[str, Any]:
    """Mechanical R2 selection from CS-Dialogue R1 only.

    The score is rank-based to avoid arbitrary unit conversion between error
    counts and retention rates.  Higher is better for POI net correction and
    matrix retention; lower is better for POI corruption.  The middle layer is
    the layer whose net-correction rank is closest to the median.
    """
    import csv as _csv
    path = result_root / "tables/raw_full_depth.csv"
    if not path.is_file():
        raise FileNotFoundError("R1 table is required before mechanical R2 selection")
    rows = [r for r in _csv.DictReader(path.open()) if r["Dataset"] == "cs_dialogue"]
    grouped = {}
    for side in ("encoder", "decoder"):
        for layer in range(32):
            z = [r for r in rows if r["Side"] == side and int(r["Layer"]) == layer]
            if len(z) != 2: continue
            corr = sum(float(r["Corr"]) for r in z) / 2.0
            corrupt = sum(float(r["Corrupt"]) for r in z) / 2.0
            ret = np.mean([float(r["Matrix-Ret"]) for r in z if r["Matrix-Ret"] not in ("", "None")])
            grouped[(side, layer)] = {"corr": corr, "corrupt": corrupt, "ret": float(ret), "net": corr - corrupt}
    def rank(values, reverse=False):
        order = sorted(values, key=values.get, reverse=reverse)
        return {k: i for i, k in enumerate(order)}
    selected = {}
    for side, max_n in (("encoder", 3), ("decoder", 4)):
        vals = {l: grouped[(side,l)] for l in range(32) if (side,l) in grouped}
        if not vals: continue
        rc, rr, rk = rank({l:v["net"] for l,v in vals.items()}, True), rank({l:v["ret"] for l,v in vals.items()}, True), rank({l:v["corrupt"] for l,v in vals.items()}, False)
        score = {l: rc[l] + rr[l] + rk[l] for l in vals}
        best = min(score, key=score.get)
        order = sorted(vals, key=lambda l: vals[l]["net"])
        negative = order[0]
        middle = min(vals, key=lambda l: abs(vals[l]["net"] - np.median([v["net"] for v in vals.values()])))
        if side == "decoder":
            chosen = [0, 24, 27]
            if negative not in chosen: chosen.append(negative)
            chosen = chosen[:max_n]
        else:
            chosen = []
            for x in (best, middle, negative):
                if x not in chosen: chosen.append(x)
            chosen = chosen[:max_n]
        labels = ({str(best): "E_best", str(middle): "E_mid", str(negative): "E_negative"}
                  if side == "encoder" else
                  {"0": "D0", "24": "D24", "27": "D27", str(negative): "D_negative"})
        selected[side] = {"layers": chosen, "labels": labels,
                          "score": {str(l): int(score[l]) for l in score}, "metrics": {str(l): vals[l] for l in vals}}
    payload = {"schema_version": "basis_a3_r2_selection_v1", "source": "CS-Dialogue R1 only",
               "mechanical": True, "selected": selected}
    write_json(result_root / "selection/r2_layers.json", payload)
    return payload
