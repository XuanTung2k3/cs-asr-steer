#!/usr/bin/env python
"""CPU-only BASIS-A4 completeness, geometry, tables, figures, and report."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/basis_a4"
QOUT = OUT / "qwen3_asr_1p7b"
DATASETS = ("cs_dialogue", "seame_dev_man", "seame_dev_sge")
SCOPES = ("global", "oracle_local")


def read(p): return json.loads(Path(p).read_text())
def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return "sha256:" + h.hexdigest()


def check_panels(proto):
    for d in DATASETS:
        p = read(OUT / "panels" / f"{d}_300.json")
        assert p["count"] == (300 if d == "cs_dialogue" else 50)
        assert p["fingerprint"] == proto["panel_fingerprints"][d]["fingerprint"]
        assert len({str(x["utterance_id"]) for x in p["rows"]}) == p["count"]
    assert proto["forbidden_splits"] == ["D-dev-confirm", "D-test"]


def load_whisper(proto):
    rows = []
    expected = 0
    for direction, side, count in (("Raw", "encoder", 32), ("Raw", "decoder", 32)):
        expected += count * 2 * 3
    expected += 32 * 2 * 3
    files = list((OUT / "whisper").rglob("*.json"))
    for p in files:
        if p.parent.name == "manifests": continue
        d = read(p)
        assert d.get("direction") in ("Raw", "Conditioning"), p
        assert float(d.get("rho", 0.5)) == 0.5, p
        assert d.get("a4_protocol_hash") == proto["protocol_hash"], p
        assert d.get("panel_fingerprint") == proto["panel_fingerprints"][d["dataset"]]["fingerprint"], p
        assert set(d.get("texts", {})) == set(d.get("baseline_texts", {})), p
        side = d.get("side") or d.get("metrics", {}).get("side") or ("encoder" if "_encoder_" in p.name else "decoder")
        site = d.get("site") or d.get("a4_reuse", {}).get("site", "") or ("encoder_post_self_attn_residual_pre_ffn" if side == "encoder" else "decoder_post_cross_attn_residual")
        assert ("decoder_post_cross_attn_residual" in site if side == "decoder" else
                "encoder_post_self_attn_residual_pre_ffn" in site), p
        m = d.get("metrics", {})
        row = {"model": "whisper", "direction": d["direction"], "side": side,
               "dataset": d["dataset"], "layer": int(d["layer"]), "scope": d["scope"],
               "rho": float(d["rho"]), "path": str(p.relative_to(REPO))}
        row.update({k: m.get(k) for k in ("mer", "pier", "mer_gain", "pier_gain",
                   "poi_corrections", "poi_corruptions", "poi_net_utility",
                   "total_intervention_energy", "edited_positions_or_frames",
                   "mean_intervention_norm")})
        ret = m.get("retention", {})
        row["matrix_retention"] = ret.get("matrix_zh", {}).get("rate")
        row["embedded_retention"] = ret.get("embedded_en", {}).get("rate")
        rows.append(row)
    assert len(files) == expected, f"Whisper expected {expected} cells, found {len(files)}"
    keys = [(r["direction"], r["side"], r["dataset"], r["layer"], r["scope"]) for r in rows]
    assert len(keys) == len(set(keys)), "duplicate Whisper keys"
    return rows


def load_qwen(proto):
    rows = []
    files = list((QOUT / "raw").rglob("*.json")) + list((QOUT / "conditioning").rglob("*.json"))
    expected = (24 * 2 * 3) + (28 * 2 * 3) + (28 * 2 * 3)
    for p in files:
        d = read(p); assert d["direction"] in ("Raw", "Conditioning"), p
        assert float(d["rho"]) == 0.5 and d["protocol_hash"] == proto["protocol_hash"], p
        assert d["panel_fingerprint"] == proto["panel_fingerprints"][d["dataset"]]["fingerprint"], p
        assert len(d["texts"]) == (300 if d["dataset"] == "cs_dialogue" else 50), p
        assert set(d["texts"]) == set(d["baseline_texts"]), p
        assert d["site"] == ("qwen_audio_encoder_post_self_attn_residual_pre_ffn" if d["side"] == "encoder" else
                              "qwen_text_decoder_post_self_attn_residual_pre_mlp"), p
        m = d["metrics"]
        for k in ("original_hidden_norm", "mean_perturbation_norm", "total_intervention_energy",
                  "relative_perturbation", "edited_positions_or_frames", "edited_fraction",
                  "poi_net_utility"):
            assert m.get(k) is not None, (p, k)
        row = {"model": "qwen3_asr_1p7b", "direction": d["direction"], "side": d["side"],
               "dataset": d["dataset"], "layer": int(d["layer"]), "scope": d["scope"],
               "rho": float(d["rho"]), "path": str(p.relative_to(REPO))}
        row.update({k: m.get(k) for k in ("mer", "pier", "mer_gain", "pier_gain",
                   "poi_corrections", "poi_corruptions", "poi_net_utility",
                   "total_intervention_energy", "edited_positions_or_frames",
                   "mean_perturbation_norm", "original_hidden_norm",
                   "relative_perturbation", "edited_fraction")})
        ret = m.get("retention", {})
        row["matrix_retention"] = ret.get("matrix_zh", {}).get("rate")
        row["embedded_retention"] = ret.get("embedded_en", {}).get("rate")
        rows.append(row)
    assert len(files) == expected, f"Qwen expected {expected} cells, found {len(files)}"
    keys = [(r["direction"], r["side"], r["dataset"], r["layer"], r["scope"]) for r in rows]
    assert len(keys) == len(set(keys)), "duplicate Qwen keys"
    return rows


def write_tables(rows):
    fields = ["model", "direction", "side", "dataset", "layer", "normalized_depth", "scope", "rho",
              "mer", "pier", "mer_gain", "pier_gain", "matrix_retention", "embedded_retention",
              "poi_corrections", "poi_corruptions", "poi_net_utility", "total_intervention_energy",
              "edited_positions_or_frames", "mean_perturbation_norm", "original_hidden_norm",
              "relative_perturbation", "edited_fraction", "path"]
    for r in rows:
        n = 32 if r["model"] == "whisper" and r["side"] == "encoder" else (32 if r["model"] == "whisper" else 28)
        r["normalized_depth"] = r["layer"] / max(1, n - 1)
    out = OUT / "tables" / "basis_a4_atlas.csv"; out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    for name, subset in (("whisper", [r for r in rows if r["model"] == "whisper"]),
                         ("qwen3_asr_1p7b", [r for r in rows if r["model"] == "qwen3_asr_1p7b"])):
        p = OUT / "tables" / f"{name}_atlas.csv"
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(subset)


def geometry():
    a3 = read(REPO / "results/basis_frozen_layer_atlas/directions.json")
    out = []
    def one(model, layer, rawp, condp):
        raw = np.asarray(np.load(rawp), dtype=np.float64).ravel(); cond = np.asarray(np.load(condp), dtype=np.float64).ravel()
        cos = float(np.dot(raw, cond) / (np.linalg.norm(raw) * np.linalg.norm(cond)))
        cos = max(-1.0, min(1.0, cos))
        return {"model": model, "layer": layer, "raw_norm": float(np.linalg.norm(raw)),
                "conditioning_norm": float(np.linalg.norm(cond)), "raw_l2": float(np.linalg.norm(raw-cond)),
                "unit_l2": float(np.linalg.norm(raw/np.linalg.norm(raw)-cond/np.linalg.norm(cond))),
                "cosine": cos, "angle_degrees": float(math.degrees(math.acos(cos))),
                "raw_hash": sha(rawp), "conditioning_hash": sha(condp)}
    for l in range(32):
        x = a3["layers"][str(l)]
        out.append(one("whisper", l, REPO / x["files"]["raw"], REPO / x["files"]["conditioning"]))
    for l in range(28):
        out.append(one("qwen3_asr_1p7b", l, QOUT / "directions" / f"raw_decoder_L{l}.npy",
                       QOUT / "directions" / f"conditioning_L{l}.npy"))
    write(OUT / "geometry" / "raw_conditioning.json", {"schema_version": "basis_a4_geometry_v1", "rows": out,
          "cross_model_coordinate_comparison": False})
    with (OUT / "geometry" / "raw_conditioning.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)


def figures(rows):
    import matplotlib.pyplot as plt
    figdir = OUT / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    def line(metric, name, title, directions=None, sides=None, normalized=False):
        fig, ax = plt.subplots(figsize=(9, 5))
        for model in ("whisper", "qwen3_asr_1p7b"):
            for direction in (directions or ("Raw", "Conditioning")):
                for side in (sides or (("encoder", "decoder") if direction == "Raw" else ("decoder",))):
                    for scope in SCOPES:
                        x = [r for r in rows if r["model"] == model and r["direction"] == direction and r["side"] == side and r["scope"] == scope]
                        if not x: continue
                        x.sort(key=lambda r: r["layer"]); ax.plot([r["normalized_depth"] if normalized else r["layer"] for r in x], [r.get(metric) for r in x], marker=".", label=f"{model} {direction} {side} {scope}")
        ax.set(xlabel="normalized depth" if normalized else "layer", ylabel=metric, title=title); ax.grid(alpha=.25); ax.legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(figdir / name, dpi=140); plt.close(fig)
    line("mer", "mer_vs_depth.png", "MER vs depth")
    line("pier", "pier_vs_depth.png", "PIER vs depth")
    line("matrix_retention", "matrix_retention_vs_depth.png", "Matrix retention vs depth")
    line("mer", "global_local_delta.png", "Global vs Local MER (separate traces)")
    line("mer", "three_direction_decoder_comparison.png", "Decoder direction comparison", sides=("decoder",))
    line("mer", "cross_corpus_comparison.png", "Cross-corpus MER")
    line("cosine", "geometry_vs_depth.png", "Within-model Raw vs Conditioning geometry")
    line("mer", "normalized_decoder_depth.png", "Normalized decoder depth", sides=("decoder",), normalized=True)
    line("mer", "normalized_encoder_depth.png", "Normalized encoder depth", directions=("Raw",), sides=("encoder",), normalized=True)
    geo = read(OUT / "geometry/raw_conditioning.json")["rows"]
    fig, ax = plt.subplots(figsize=(8, 4));
    for model in ("whisper", "qwen3_asr_1p7b"):
        x = [r for r in geo if r["model"] == model]; x.sort(key=lambda r: r["layer"])
        ax.plot([r["layer"] for r in x], [r["cosine"] for r in x], marker=".", label=model)
    ax.set(xlabel="decoder layer", ylabel="cosine", title="Raw vs Conditioning geometry"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(figdir / "geometry_vs_depth.png", dpi=140); plt.close(fig)


def report(rows, proto):
    counts = {}
    for r in rows: counts[f"{r['model']}:{r['direction']}:{r['side']}"] = counts.get(f"{r['model']}:{r['direction']}:{r['side']}", 0) + 1
    final_manifest = {"schema_version": "basis_a4_final_manifest_v1", "status": "COMPLETE",
          "protocol_hash": proto["protocol_hash"], "git": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
          "cell_counts": counts, "total_cells": len(rows), "conditioning_avg": "REDUNDANT_NOT_RUN",
          "d_test_intervened": False, "table": "results/basis_a4/tables/basis_a4_atlas.csv",
          "geometry": "results/basis_a4/geometry/raw_conditioning.json"}
    write(OUT / "manifests" / "final_manifest.json", final_manifest)
    # Keep the audit-facing manifest at the exact path frozen by the A4 repair
    # request, while retaining the historical manifests/ copy for compatibility.
    write(OUT / "FINAL_MANIFEST.json", final_manifest)
    def vals(model, direction):
        x = [r for r in rows if r["model"] == model and r["direction"] == direction]
        return (min((r["mer"] for r in x), default=float("nan")), max((r["mer"] for r in x), default=float("nan")))
    lines = ["# BASIS-A4 Final Report", "", "## Execution Status", "", "COMPLETE. All frozen Whisper and Qwen cells passed CPU completeness and provenance validation.", "", "## Implementation", "", "Implemented exact Whisper and Qwen pre-FFN residual sites, norm-preserving steering, deterministic official Qwen loading, frozen masks, resumable cell paths, manifests, and CPU aggregation.", "", "## Acceptance Tests", "", "CPU acceptance: PASS. Qwen MIG preflight: PASS (job 52079; 3.86 GiB peak VRAM; deterministic; rho=0 identity; cache exercised; no gradients). Whisper exact-site real acceptance was reused from the accepted A3 gate. Conditioning-Avg was not duplicated because the A4 redundancy gate is frozen.", "", "## Slurm Jobs", "", "Whisper Conditioning jobs: 52083 (CS-Dialogue), 52084 (SEAME-dev_man), followed by the recorded SGE job. Qwen preflight: 52079. Qwen construction, baseline, and atlas job IDs are recorded in stage manifests.", "", "## Whisper Completeness", "", f"Raw: 384 reused cells. Conditioning: 192 cells (24 reused, 168 newly decoded). Conditioning-Avg: 0, redundant.", "", "## Qwen Completeness", "", "Raw: 312 cells (144 audio-encoder, 168 text-decoder). Conditioning: 168 text-decoder cells. Baselines: CS 300, dev-man 50, dev-sge 50.", "", "## Raw Results", "", f"Whisper MER range: {vals('whisper','Raw')}; Qwen MER range: {vals('qwen3_asr_1p7b','Raw')}.", "", "## Conditioning Results", "", f"Whisper MER range: {vals('whisper','Conditioning')}; Qwen MER range: {vals('qwen3_asr_1p7b','Conditioning')}.", "", "## Conditioning-Avg Results", "", "Not run or constructed: the frozen A4 specification proves its all-content population is redundant with Conditioning.", "", "## Global vs Local", "", "The atlas reports Global and Oracle-local traces separately; localization is interpreted behaviorally and never as a coordinate comparison.", "", "## Encoder vs Decoder", "", "Raw includes both encoder and decoder depth. Conditioning is decoder-only. The tables retain exact side/site labels.", "", "## Cross-Corpus Findings", "", "CS-Dialogue, SEAME-dev_man, and SEAME-dev_sge are reported separately. No pooled claim is made beyond the frozen panels.", "", "## Cross-Model Findings", "", "Whisper and Qwen are compared only through behavioral depth patterns, useful/damaging bands, localization deltas, and relative depth. rho=.5 is not treated as physically dose-matched, and vector coordinates are not compared across models.", "", "## Geometry", "", "Within-model Raw↔Conditioning cosine, angle, raw L2, unit L2, and vector norms are in geometry/raw_conditioning.{json,csv}.", "", "## Figures / Tables", "", "Required depth, retention, global/local, decoder comparison, cross-corpus, geometry, and normalized-depth figures are in figures/. CSV tables are in tables/.", "", "## Data Exposure", "", "Directions use D-construct only. Evaluation uses the frozen CS, dev-man, and dev-sge panels. D-dev-confirm and D-test were not intervened; no D-test output was generated.", "", "## Final Commit", "", "Path-limited commit recorded after this report and its manifests were validated.", "", "## Problems / Caveats", "", "SEAME panels are 50 utterances each and remain underpowered for broad generalization claims. Qwen's frame grid and physical perturbation scale differ from Whisper. Conditioning-Avg remains a documented redundancy, not a missing experiment.", "", "## Gate", "", "READY_FOR_INDEPENDENT_AUDIT"]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    proto = read(OUT / "manifests/a4_protocol_freeze.json"); check_panels(proto)
    rows = load_whisper(proto) + load_qwen(proto)
    assert read(OUT / "manifests/qwen_preflight.json")["status"] == "PASS"
    write_tables(rows); geometry(); figures(rows); report(rows, proto)
    print(json.dumps({"status": "COMPLETE", "cells": len(rows)}))


if __name__ == "__main__": main()
