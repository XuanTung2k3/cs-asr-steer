#!/usr/bin/env python
"""DIR-SPRINT0 D5 design inputs (CPU; provider environment; no CS-Dialogue audio, transcript, label or outcome).

Three frozen design artifacts, written once into ``results/inference_cf/dir_sprint0/design/``:

* ``concept`` -> ``d5_concept_table.json``. For every symbol of the frozen provider feature table, the fraction of its
  panphon segments in each of the six primary concept classes and the two descriptive (language-confounded) classes.
  Segments are the table's own frozen ``segments`` (already parsed by panphon 0.22.2 at table build). Per-segment
  panphon numeric values (+1/0/-1) decide class membership:

    NAS  nas=+1                         STOP son=-1 & cont=-1         FRIC son=-1 & cont=+1
    LAB  cons=+1 & lab=+1               COR  cons=+1 & cor=+1         DOR  cons=+1 & cor=-1 & lab=-1 & hi=+1
    descriptive only: VOI_OBS son=-1 & voi=+1 (voiced obstruent);  ASP sg=+1 (spread glottis / aspiration)

  panphon codes glottals (h, ɦ) as [+son, +cons, -hi, +back]; they therefore fall in no primary class (not FRIC,
  because +son; not DOR, because -hi). Uvulars (-hi) are likewise outside DOR; neither EN nor ZH espeak uses them.

  Non-segmental rows (blank, special, tone-only, excluded) carry no concept value. Tone never enters.
* ``engineering`` -> ``d5_threshold_engineering.json``. Unlabeled engineering evidence for the frozen D5 evidence
  thresholds: the provider is run once on each permitted public engineering clip (16 kHz ones) and on synthetic
  silence / low-level noise. Per frame j it records the segmental mass s_j = sum over segmental symbols of p_jv, and
  per 1-second window (stride 0.25 s, frozen frame-overlap weights w_j) the emitting weight E = sum_j w_j [s_j >= 0.5]
  together with the raw segmental mass. Nothing here is a CS-Dialogue utterance and nothing is scored against a
  transcript.
* ``reference`` -> ``d5_provider_reference.{json,npz}``. The audited CPU posteriors of the two Job A provider-apparatus
  clips (hash-equal to the sealed engineering check), for the in-job cross-host tolerance check.

Run with ``/mnt/data/tungnx/cs-asr-steer/envs/dir_sprint0_d5/bin/python -I``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

import numpy as np

TABLE = ROOT / "results/inference_cf/dir_sprint0/d5_provider/feature_table.json"
MANIFEST = ROOT / "results/inference_cf/dir_sprint0/d5_provider/provider_manifest.json"
OUT_DIR = ROOT / "results/inference_cf/dir_sprint0/design"
VERSION = "dir_sprint0_d5_concepts_v1"
PRIMARY = ("NAS", "STOP", "FRIC", "LAB", "COR", "DOR")
DESCRIPTIVE = ("VOI_OBS", "ASP")
RULES = {
    "NAS": "nas=+1",
    "STOP": "son=-1 & cont=-1",
    "FRIC": "son=-1 & cont=+1",
    "LAB": "cons=+1 & lab=+1",
    "COR": "cons=+1 & cor=+1",
    "DOR": "cons=+1 & cor=-1 & lab=-1 & hi=+1",
    "VOI_OBS": "son=-1 & voi=+1",
    "ASP": "sg=+1",
}


def member(f: dict) -> dict:
    """Class membership of ONE panphon segment from its numeric features (+1/0/-1)."""
    return {"NAS": f["nas"] == 1,
            "STOP": f["son"] == -1 and f["cont"] == -1,
            "FRIC": f["son"] == -1 and f["cont"] == 1,
            "LAB": f["cons"] == 1 and f["lab"] == 1,
            "COR": f["cons"] == 1 and f["cor"] == 1,
            "DOR": f["cons"] == 1 and f["cor"] == -1 and f["lab"] == -1 and f["hi"] == 1,
            "VOI_OBS": f["son"] == -1 and f["voi"] == 1,
            "ASP": f["sg"] == 1}


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def concept_digest(t: dict) -> str:
    return "sha256:" + hashlib.sha256(json.dumps({k: t[k] for k in ("classes", "rules", "rows", "feature_table_digest")},
                                                 sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_concepts() -> dict:
    import panphon
    ft = panphon.FeatureTable()
    tab = json.loads(TABLE.read_text(encoding="utf-8"))
    classes = list(PRIMARY) + list(DESCRIPTIVE)
    rows = []
    for r in tab["rows"]:
        rec = {"id": r["id"], "symbol": r["symbol"], "status": r["status"], "segments": r["segments"], "values": None,
               "segment_members": None}
        if r["status"] == "segmental":
            mem = []
            for s in r["segments"]:
                if not ft.seg_known(s):
                    raise ValueError(f"panphon no longer knows segment {s!r} of symbol {r['symbol']!r}")
                f = dict(zip(ft.names, ft.fts(s).numeric()))
                mem.append({k: bool(v) for k, v in member(f).items()})
            rec["segment_members"] = [[k for k in classes if m[k]] for m in mem]
            rec["values"] = [sum(m[k] for m in mem) / len(mem) for k in classes]
        rows.append(rec)
    import importlib.metadata as md
    out = {"schema": VERSION, "classes": classes, "primary": list(PRIMARY), "descriptive_only": list(DESCRIPTIVE),
           "rules": RULES, "aggregation": "fraction of the symbol's frozen panphon segments in the class",
           "feature_table": str(TABLE.relative_to(ROOT)), "feature_table_sha256": sha(TABLE),
           "feature_table_digest": tab["table_digest"], "panphon_version": md.version("panphon"),
           "excluded_from_primary": {"VOI_OBS": "voicing: English b/d/g vs Mandarin unaspirated p/t/k written with the "
                                                "same espeak letters (language-convention confound)",
                                     "ASP": "aspiration: marked for Mandarin, never for English in espeak (confound)"},
           "tone": "recorded separately by the provider table; never converted into a concept value",
           "rows": rows}
    out["counts"] = {k: sum(1 for r in rows if r["values"] is not None and r["values"][i] > 0) for i, k in enumerate(classes)}
    out["concept_table_digest"] = concept_digest(out)
    return out


def windows(n_samples: int, win: int = 16000, stride: int = 4000) -> list[tuple[int, int]]:
    if n_samples <= win:
        return [(0, n_samples)]
    return [(a, a + win) for a in range(0, n_samples - win + 1, stride)]


def build_engineering() -> dict:
    import soundfile as sf
    from csasr.inference_cf.phone_provider import PhoneProvider, feature_evidence, interval_weights
    import torch
    torch.set_num_threads(4)
    m = json.loads(MANIFEST.read_text())
    prov = PhoneProvider(MANIFEST, TABLE, device="cpu")
    adir = Path(m["engineering_audio"]["dir"])
    clips = {}
    for name, meta in sorted(m["engineering_audio"]["files"].items()):
        if meta["sample_rate"] != 16000:
            continue
        p = adir / name
        if sha(p) != meta["sha256"]:
            raise ValueError(f"engineering clip changed: {name}")
        x, sr = sf.read(str(p), dtype="float32")
        clips[name] = (x, meta["language"])
    rng = np.random.default_rng(240924)
    clips["synthetic_silence_3s"] = (np.zeros(48000, dtype=np.float32), "none")
    clips["synthetic_noise_3s_std1e-3"] = ((rng.standard_normal(48000) * 1e-3).astype(np.float32), "none")
    clips["synthetic_noise_3s_std1e-2"] = ((rng.standard_normal(48000) * 1e-2).astype(np.float32), "none")
    out = {}
    for name, (x, lang) in clips.items():
        r = prov.posteriors(x, 16000)
        if r["status"] != "ok":
            raise RuntimeError(f"{name}: {r['status']}")
        rows = []
        for a, b in windows(len(x)):
            w = interval_weights(a, b, r["n_frames"])
            ev = feature_evidence(r["log_probs"], w, prov.table)
            mass = ev["mass"]
            nonblank = mass["segmental"] + mass["excluded"] + mass["special"] + mass["tone_only"]
            p = np.exp(r["log_probs"].astype(np.float64))
            per_frame_seg = (w * (p[:, prov.table.valid].sum(axis=1)))
            pf = per_frame_seg[per_frame_seg > 0]
            neff = float(pf.sum() ** 2 / (pf * pf).sum()) if pf.size else 0.0
            xx = x[a:b].astype(np.float64)
            s_j = p[:, prov.table.valid].sum(axis=1)
            rows.append({"start": a, "end": b, "frames": int(np.count_nonzero(w)), "weight_total": float(w.sum()),
                         "emitting_weight": float((w * (s_j >= 0.5)).sum()),
                         "M_seg": mass["segmental"], "nonblank": nonblank,
                         "seg_share_of_nonblank": (mass["segmental"] / nonblank) if nonblank > 0 else None,
                         "seg_share_of_all": mass["segmental"] / float(w.sum()), "n_eff_segmental_frames": neff,
                         "rms": float(np.sqrt(np.mean(xx * xx))) if xx.size else 0.0})
        def qs(key):
            v = np.asarray([z[key] for z in rows if z[key] is not None], dtype=np.float64)
            return {"min": float(v.min()), "p05": float(np.quantile(v, .05)), "p25": float(np.quantile(v, .25)),
                    "median": float(np.median(v)), "max": float(v.max())} if v.size else None
        pall = np.exp(r["log_probs"].astype(np.float64))
        sj = pall[:, prov.table.valid].sum(axis=1)
        out[name] = {"language": lang, "samples": int(len(x)), "n_windows": len(rows), "log_probs_sha256": r["log_probs_sha256"],
                     "summary": {k: qs(k) for k in ("frames", "emitting_weight", "M_seg", "seg_share_of_nonblank",
                                                     "seg_share_of_all", "n_eff_segmental_frames", "rms")},
                     "frame_segmental_mass": {"n_frames": int(sj.size),
                                              "quantiles_0_50_75_90_95_99_100": [float(x) for x in np.quantile(sj, [0, .5, .75, .9, .95, .99, 1])],
                                              "fraction_ge_0.5": float((sj >= .5).mean()),
                                              "fraction_in_0.2_0.8": float(((sj > .2) & (sj < .8)).mean()),
                                              "emitting_frames_per_second": float((sj >= .5).sum() / (len(x) / 16000))},
                     "windows": rows}
    return {"schema": VERSION + "_engineering", "provider_manifest_digest": m["manifest_digest"],
            "window": "1 s (16000 samples), stride 0.25 s, frozen interval_weights (fractional 400-sample overlap)",
            "threads": torch.get_num_threads(), "clips": out,
            "scope": "public engineering clips (16 kHz) and synthetic signals only; no CS-Dialogue data, no transcript"}


def build_reference() -> tuple[dict, dict]:
    """Provider apparatus reference: CPU float32 4-thread posteriors of the two apparatus clips; their hashes must equal
    the sealed provider engineering check C hashes (so the arrays are the audited ones)."""
    import soundfile as sf
    from csasr.inference_cf.phone_provider import PhoneProvider
    import torch
    torch.set_num_threads(4)
    m = json.loads(MANIFEST.read_text())
    eng = json.loads((ROOT / "results/inference_cf/dir_sprint0/d5_provider/engineering_checks.json").read_text())
    prov = PhoneProvider(MANIFEST, TABLE, device="cpu")
    adir = Path(m["engineering_audio"]["dir"])
    arrays, rec = {}, {}
    for name in ("sample1.flac", "aishell_example_mandarin.wav"):
        p = adir / name
        if sha(p) != m["engineering_audio"]["files"][name]["sha256"]:
            raise ValueError(f"engineering clip changed: {name}")
        x, sr = sf.read(str(p), dtype="float32")
        r = prov.posteriors(x, sr)
        if r["log_probs_sha256"] != eng["C"]["files"][name]["log_probs_sha256"]:
            raise ValueError(f"{name}: posteriors differ from the sealed engineering hash")
        arrays[name] = r["log_probs"]
        rec[name] = {"audio_sha256": m["engineering_audio"]["files"][name]["sha256"], "log_probs_sha256": r["log_probs_sha256"],
                     "shape": list(r["log_probs"].shape), "sealed_engineering_hash_equal": True}
    return {"schema": VERSION + "_provider_reference", "threads": torch.get_num_threads(), "device": "cpu",
            "dtype": "float32", "clips": rec, "provider_manifest_digest": m["manifest_digest"]}, arrays


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("concept", "engineering", "reference"))
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.what == "reference":
        jp, npzp = OUT_DIR / "d5_provider_reference.json", OUT_DIR / "d5_provider_reference.npz"
        if jp.exists() or npzp.exists():
            raise FileExistsError("provider reference exists; never overwritten")
        rec, arrays = build_reference()
        np.savez(npzp, **{k.replace(".", "_"): v for k, v in arrays.items()})
        rec["npz_sha256"] = sha(npzp)
        rec["npz_keys"] = {k: k.replace(".", "_") for k in arrays}
        jp.write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps(rec))
        return
    path = OUT_DIR / ("d5_concept_table.json" if a.what == "concept" else "d5_threshold_engineering.json")
    if path.exists():
        raise FileExistsError(f"{path} exists; frozen design artifacts are never overwritten")
    res = build_concepts() if a.what == "concept" else build_engineering()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    if a.what == "concept":
        print(json.dumps({"digest": res["concept_table_digest"], "counts": res["counts"]}))
    else:
        print(json.dumps({k: v["summary"] for k, v in res["clips"].items()}, indent=1))


if __name__ == "__main__":
    main()
