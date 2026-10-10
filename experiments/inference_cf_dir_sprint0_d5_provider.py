#!/usr/bin/env python
"""DIR-SPRINT0 D5 provider setup: frozen provider manifest + engineering verification (checks A-J).

    manifest     hash every provider file, the isolated environment, the feature table and the adapter
    engineering  run checks A-J on synthetic signals and permitted public engineering audio only

Run with the isolated provider environment python (``/mnt/data/tungnx/cs-asr-steer/envs/dir_sprint0_d5``). No
CS-Dialogue audio, role parquet, reference, label, alignment or DIR-SPRINT0 outcome is read; every opened path is
logged and screened. The public samples check only that the tool behaves plausibly. They establish no code-switching
accuracy and select nothing.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

import numpy as np

from csasr.inference_cf import phone_provider as pp

OUT_DIR = ROOT / "results/inference_cf/dir_sprint0/d5_provider"
MANIFEST = OUT_DIR / "provider_manifest.json"
TABLE = OUT_DIR / "feature_table.json"
CHECKS = OUT_DIR / "engineering_checks.json"
REVISION = "2c733782da5604684829819a5eb744c193fe9398"
MODEL_DIR = Path(f"/mnt/data/tungnx/cs-asr-steer/providers/wav2vec2-xlsr-53-espeak-cv-ft/{REVISION}")
MODEL_FILES = (".gitattributes", "README.md", "config.json", "preprocessor_config.json", "special_tokens_map.json",
               "tokenizer_config.json", "vocab.json", "pytorch_model.bin")
PUBLISHED_WEIGHTS_LFS_SHA256 = "sha256:04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551"
ENV_DIR = Path("/mnt/data/tungnx/cs-asr-steer/envs/dir_sprint0_d5")
WHEELHOUSE = Path("/mnt/data/tungnx/cs-asr-steer/envs/dir_sprint0_d5_wheelhouse")
ACL1_PY = Path("/home/tungnx/miniconda3/envs/acl1/bin/python")
AUDIO_DIR = Path("/mnt/data/tungnx/cs-asr-steer/providers/engineering_audio")
AUDIO = {
    "sample1.flac": {"url": "https://cdn-media.huggingface.co/speech_samples/sample1.flac", "language": "en",
                     "origin": "LibriSpeech sample used as the provider model-card widget", "license": "CC-BY-4.0 (LibriSpeech)"},
    "sample2.flac": {"url": "https://cdn-media.huggingface.co/speech_samples/sample2.flac", "language": "en",
                     "origin": "LibriSpeech sample used as the provider model-card widget", "license": "CC-BY-4.0 (LibriSpeech)"},
    "aishell_example_mandarin.wav": {
        "url": "https://huggingface.co/speechbrain/asr-transformer-aishell/resolve/703ce755c649ba09211202a35a7cff2eb6eadb8c/example_mandarin.wav",
        "language": "zh", "origin": "AISHELL-1 example shipped in speechbrain/asr-transformer-aishell", "license": "Apache-2.0 (repo; AISHELL-1)"},
    "aishell_example_mandarin2.flac": {
        "url": "https://huggingface.co/speechbrain/asr-transformer-aishell/resolve/703ce755c649ba09211202a35a7cff2eb6eadb8c/example_mandarin2.flac",
        "language": "zh", "origin": "AISHELL-1 example shipped in speechbrain/asr-transformer-aishell", "license": "Apache-2.0 (repo; AISHELL-1)"},
    "cv14_example_zh_CN.wav": {
        "url": "https://huggingface.co/speechbrain/asr-wav2vec2-commonvoice-14-zh-CN/resolve/92a28d6b46996c6a63b052b1933ddbfd5910dea5/example-zh-CN.wav",
        "language": "zh", "origin": "Common Voice 14 zh-CN example shipped in speechbrain/asr-wav2vec2-commonvoice-14-zh-CN",
        "license": "Apache-2.0 (repo; Common Voice CC0)"},
}
OPENED: list[str] = []
FORBIDDEN_PATH_MARKERS = ("role_D-", ".parquet", "/CS-Dialogue", "artifacts_dialogue", "/locked/", "srd2_g0/run1/gate",
                          "srd2_g0/run1/pulses", "evaluation_units")


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        OPENED.append(os.fsdecode(args[0]))


def git(*a: str) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
    tmp.replace(path)


def acl1_state() -> dict:
    freeze = subprocess.check_output([str(ACL1_PY), "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL)
    return {"python": str(ACL1_PY), "pip_freeze_sha256": "sha256:" + hashlib.sha256(freeze.encode()).hexdigest(),
            "pip_freeze_lines": len(freeze.splitlines())}


# ---- manifest -----------------------------------------------------------------------------------------------------

def cmd_manifest(args) -> None:
    import soundfile as sf
    import torch
    import transformers
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    audio = {}
    for name, meta in AUDIO.items():
        info = sf.info(str(AUDIO_DIR / name))
        audio[name] = {**meta, "sha256": pp.file_sha256(AUDIO_DIR / name), "sample_rate": info.samplerate,
                       "channels": info.channels, "frames": info.frames}
    body = {
        "schema": "dir_sprint0_d5_provider_manifest_v1", "status": "PROVIDER_FROZEN_ENGINEERING_ONLY",
        "created_unix": time.time(), "git_head_at_creation": git("rev-parse", "HEAD"),
        "model": {"repo": "facebook/wav2vec2-xlsr-53-espeak-cv-ft", "revision": REVISION, "license": "apache-2.0",
                  "architecture": "Wav2Vec2ForCTC (wav2vec2-large-xlsr-53 fine-tuned on Common Voice espeak phones)",
                  "paper": "arXiv:2109.11680", "dir": str(MODEL_DIR),
                  "download_url_template": f"https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft/resolve/{REVISION}/<file>",
                  "files": {f: pp.file_sha256(MODEL_DIR / f) for f in MODEL_FILES},
                  "sizes": {f: (MODEL_DIR / f).stat().st_size for f in MODEL_FILES},
                  "published_weights_lfs_sha256": PUBLISHED_WEIGHTS_LFS_SHA256,
                  "tokenizer_class_not_used": "Wav2Vec2PhonemeCTCTokenizer (would require phonemizer); ids map through vocab.json"},
        "environment": {
            "venv": str(ENV_DIR), "venv_base_python": str(ACL1_PY), "system_site_packages": True,
            "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__, "soundfile": md.version("soundfile"),
            "installed_into_venv": {"panphon": md.version("panphon"), "unicodecsv": md.version("unicodecsv")},
            "requirements_file": {"path": str(ENV_DIR / "requirements-d5.txt"), "sha256": pp.file_sha256(ENV_DIR / "requirements-d5.txt")},
            "wheelhouse": {p.name: pp.file_sha256(p) for p in sorted(WHEELHOUSE.iterdir())},
            "acl1_before_install": json.loads(Path(args.acl1_before).read_text()),
            "acl1_at_manifest": acl1_state()},
        "feature_table": {"path": str(TABLE.relative_to(ROOT)), "file_sha256": pp.file_sha256(TABLE),
                          "table_digest": table["table_digest"], "schema": table["schema"],
                          "panphon_version": table["panphon_version"], "panphon_data_sha256": table["panphon_data_sha256"],
                          "status_counts": table["status_counts"],
                          "builder": {"path": "experiments/inference_cf_dir_sprint0_d5_feature_table.py",
                                      "sha256": pp.file_sha256(ROOT / "experiments/inference_cf_dir_sprint0_d5_feature_table.py")}},
        "adapter": {"path": "src/csasr/inference_cf/phone_provider.py", "version": pp.VERSION,
                    "sha256": pp.file_sha256(ROOT / "src/csasr/inference_cf/phone_provider.py")},
        "frame_geometry": {"sample_rate": pp.SAMPLE_RATE, "frame_stride": pp.FRAME_STRIDE,
                           "receptive_field": pp.RECEPTIVE_FIELD, "n_frames": "0 if N < 400 else (N - 400) // 320 + 1",
                           "provider_frame_span": "[320 j, 320 j + 400)", "r2_encoder_frame_span": "[320 f, 320 f + 320)"},
        "licenses": {"model": "Apache-2.0", "panphon": "MIT", "unicodecsv": "BSD"},
        "engineering_audio": {"dir": str(AUDIO_DIR), "files": audio,
                              "use": "engineering plausibility only; not CS-Dialogue; not used for selection or accuracy claims"},
    }
    body["manifest_digest"] = pp.canonical_digest(body)
    atomic(MANIFEST, body)
    print(json.dumps({"manifest_digest": body["manifest_digest"], "weights": body["model"]["files"]["pytorch_model.bin"],
                      "weights_match_published": body["model"]["files"]["pytorch_model.bin"] == PUBLISHED_WEIGHTS_LFS_SHA256}))


# ---- engineering checks -------------------------------------------------------------------------------------------

EN_REQUIRED = ("p b t d k ɡ f v θ ð s z ʃ ʒ h tʃ dʒ m n ŋ l ɹ w j ɾ "
               "iː ɪ ɛ æ ʌ ɑː ɔː ʊ uː ə ɚ ɜː eɪ aɪ ɔɪ aʊ oʊ").split()
ZH_REQUIRED = "p ph t th k kh m n l f x ɕ tɕ tɕh s. ts. ts.h ɻ s ts tsh ŋ i5 u5 y5 ə5 a5 o5 i.5 i̪5 ər5 onɡ5 uei5 iou5".split()
CONTRASTS = [  # (a, b, feature, relation)
    ("p", "ph", "sg", "differ"), ("t", "th", "sg", "differ"), ("k", "kh", "sg", "differ"), ("ts", "tsh", "sg", "differ"),
    ("ts.", "ts.h", "sg", "differ"), ("tɕ", "tɕh", "sg", "differ"),
    ("p", "b", "voi", "differ"), ("t", "d", "voi", "differ"), ("k", "ɡ", "voi", "differ"), ("s", "z", "voi", "differ"),
    ("ʃ", "ʒ", "voi", "differ"), ("tʃ", "dʒ", "voi", "differ"), ("θ", "ð", "voi", "differ"),
    ("ts", "s", "cont", "differ"), ("ts.", "s.", "cont", "differ"), ("tɕ", "ɕ", "cont", "differ"), ("tʃ", "ʃ", "cont", "differ"),
    ("ts", "t", "delrel", "differ"), ("tʃ", "t", "delrel", "differ"),
    ("s", "s.", "ant", "differ"), ("ɕ", "ʃ", "hi", "differ"), ("s", "θ", "distr", "differ"),
    ("i", "iː", "long", "differ"), ("ɑ", "ɑ̃", "nas", "differ"), ("m", "b", "nas", "differ"),
    ("ɑ5", "ɑ2", "*", "equal_features_tone_differs"), ("i.5", "i.2", "*", "equal_features_tone_differs"),
    ("s.", "ʂ", "*", "equal_features"), ("tS", "tʃ", "*", "equal_features"),
]


def check_table(table: pp.FeatureTable, raw: dict) -> dict:
    idx = {s: i for i, s in enumerate(table.symbols)}
    rows = raw["rows"]
    ipa_only = [r["symbol"] for r in rows if r["status"] == "segmental" and any(ord(c) > 127 for c in r["ipa"])]
    A = {"segmental_with_ipa_specific_characters": len(ipa_only),
         "phone_level_distinctions_absent_from_latin_orthography": [p for p in (("θ", "ð"), ("ʃ", "ʒ"), ("tɕ", "ts"), ("s.", "s"), ("ɕ", "ʃ"))
                                                                    if p[0] in idx and p[1] in idx],
         "contrast_grapheme_model": "local mms-fa inventory = 26 Latin letters + apostrophe (0 IPA-specific symbols)"}
    A["pass"] = A["segmental_with_ipa_specific_characters"] >= 100 and len(A["phone_level_distinctions_absent_from_latin_orthography"]) == 5

    def cov(req):
        miss = [s for s in req if s not in idx]
        bad = [s for s in req if s in idx and not table.valid[idx[s]]]
        return {"required": req, "missing": miss, "not_segmental": bad, "pass": not miss and not bad}
    tones = {str(k): int(np.sum(table.valid & (table.tone == k))) for k in range(1, 6)}
    B = {"english_espeak_en_us": cov(EN_REQUIRED), "mandarin_espeak_cmn": cov(ZH_REQUIRED), "tone_bearing_symbols": tones}
    B["pass"] = B["english_espeak_en_us"]["pass"] and B["mandarin_espeak_cmn"]["pass"] and all(v > 0 for v in tones.values())
    D = {"status_counts": raw["status_counts"], "every_symbol_has_status": bool(all(s in pp.STATUS_KINDS for s in table.status)),
         "segmental_features_finite": bool(np.all(np.isfinite(table.F))), "feature_names": table.names}
    D["pass"] = D["every_symbol_has_status"] and D["segmental_features_finite"] and len(table) == 392
    excl = [r for r in rows if r["status"] == "excluded"]
    E = {"excluded": [{"id": r["id"], "symbol": r["symbol"], "reason": r["reason"]} for r in excl],
         "special_and_blank": [r["symbol"] for r in rows if r["status"] in ("blank", "special")],
         "tone_only": [r["symbol"] for r in rows if r["status"] == "tone_only"],
         "all_excluded_have_reason": all(r["reason"] for r in excl),
         "excluded_symbols_in_required_EN_ZH_sets": [s for s in EN_REQUIRED + ZH_REQUIRED if s in idx and table.status[idx[s]] == "excluded"]}
    E["pass"] = E["all_excluded_have_reason"] and not E["excluded_symbols_in_required_EN_ZH_sets"]
    res = []
    for a, b, feat, rel in CONTRASTS:
        fa, fb = table.F[idx[a]], table.F[idx[b]]
        if rel == "differ":
            k = table.names.index(feat)
            ok = bool(fa[k] != fb[k])
            res.append({"a": a, "b": b, "feature": feat, "a_value": fa[k], "b_value": fb[k], "pass": ok})
        else:
            ok = bool(np.array_equal(fa, fb)) and (rel != "equal_features_tone_differs" or table.tone[idx[a]] != table.tone[idx[b]])
            res.append({"a": a, "b": b, "relation": rel, "pass": ok})
    F = {"contrasts": res, "pass": all(r["pass"] for r in res),
         "convention_note": "Labels follow each language's espeak lexicon: English voiceless stops are unmarked for aspiration "
                            "(p: sg -1) although phonetically aspirated in onsets; Mandarin aspiration is marked (ph: sg +1); "
                            "Mandarin unaspirated b/d/g are labelled p/t/k (voi -1). Voicing and spread-glottis evidence therefore "
                            "partly encodes the provider's implicit language choice, not only acoustics. Faithfully mapped, not collapsed."}
    return {"A": A, "B": B, "D": D, "E": E, "F": F}


def load_audio(name: str):
    import soundfile as sf
    x, sr = sf.read(str(AUDIO_DIR / name), dtype="float32", always_2d=False)
    return x, sr


def cmd_engineering(args) -> None:
    sys.addaudithook(_audit)
    import torch
    torch.set_num_threads(int(args.threads))
    t0 = time.time()
    raw = json.loads(TABLE.read_text(encoding="utf-8"))
    prov = pp.PhoneProvider(MANIFEST, TABLE, device="cpu", verify_weights=True)
    load_sec = time.time() - t0
    table = prov.table
    out = {"schema": "dir_sprint0_d5_engineering_checks_v1", "manifest_digest": prov.manifest["manifest_digest"],
           "provenance": prov.provenance, "torch_threads": torch.get_num_threads(), "load_sec": load_sec}
    out.update(check_table(table, raw))

    # C / I: posteriors, normalization, blank fraction, determinism (repeat + fresh instance)
    samples, C = {}, {"files": {}}
    for name in AUDIO:
        x, sr = load_audio(name)
        r = prov.posteriors(x, sr)
        rec = {"sample_rate": sr, "n_samples": int(np.asarray(x).shape[0]), "status": r["status"], "reason": r["reason"]}
        if r["status"] == "ok":
            lp = r["log_probs"]
            t1 = time.time(); r2 = prov.posteriors(x, sr); dt = time.time() - t1
            lse = np.log(np.exp(lp.astype(np.float64)).sum(axis=1))
            ids = lp.argmax(axis=1)
            seg = np.exp(lp.astype(np.float64))[:, table.valid]
            cmn_like = np.array([table.tone[i] > 0 or any(k.startswith("cmn_") for k in raw["rows"][i]["rules"])
                                 for i in range(len(table))])
            p = np.exp(lp.astype(np.float64))
            rec.update({"n_frames": r["n_frames"], "frames_formula": pp.n_frames(rec["n_samples"]),
                        "logsumexp_max_abs": float(np.max(np.abs(lse))), "blank_argmax_fraction": float(np.mean(ids == table.blank_id)),
                        "segmental_mass_fraction": float(seg.sum() / p.sum()),
                        "cmn_convention_share_of_segmental_mass": float(p[:, cmn_like & table.valid].sum() / p[:, table.valid].sum()),
                        "greedy_phones": " ".join(pp.greedy_phones(lp, table)), "log_probs_sha256": r["log_probs_sha256"],
                        "repeat_bitwise": bool(np.array_equal(lp, r2["log_probs"])), "sec_per_audio_sec": dt / (rec["n_samples"] / 16000)})
            samples[name] = (x, lp)
        C["files"][name] = rec
    ok_files = [v for v in C["files"].values() if v["status"] == "ok"]
    C["pass"] = all(v["n_frames"] == v["frames_formula"] and v["logsumexp_max_abs"] <= 1e-5 for v in ok_files) and len(ok_files) >= 2
    out["C"] = C
    fresh = pp.PhoneProvider(MANIFEST, TABLE, device="cpu", verify_weights=False)
    rng = np.random.default_rng(240924)
    noise = (0.05 * rng.standard_normal(16000 * 3)).astype(np.float32)
    det = {"noise_3s": None, "fresh_instance": {}}
    a, b = prov.posteriors(noise, 16000), prov.posteriors(noise, 16000)
    det["noise_3s"] = {"bitwise": bool(np.array_equal(a["log_probs"], b["log_probs"])), "sha256": a["log_probs_sha256"]}
    for name, (x, lp) in samples.items():
        det["fresh_instance"][name] = bool(np.array_equal(fresh.posteriors(x, 16000)["log_probs"], lp))
    det["pass"] = det["noise_3s"]["bitwise"] and all(det["fresh_instance"].values()) and all(v["repeat_bitwise"] for v in ok_files)
    det["scope"] = "CPU float32, fixed thread count; GPU determinism must be re-verified in the authorized Job A environment"
    out["I"] = det
    del fresh

    # G: 16 kHz mono handling (no internal resampling; explicit refusals)
    silence = np.zeros(16000, dtype=np.float32)
    s_out = prov.posteriors(silence, 16000)
    G = {"refuse_44100": prov.posteriors(*load_audio("aishell_example_mandarin2.flac"))["status"],
         "refuse_48000": prov.posteriors(*load_audio("cv14_example_zh_CN.wav"))["status"],
         "refuse_stereo": prov.posteriors(np.zeros((16000, 2), np.float32), 16000)["status"],
         "refuse_int16": prov.posteriors(np.zeros(16000, np.int16), 16000)["status"],
         "refuse_nan": prov.posteriors(np.full(16000, np.nan, np.float32), 16000)["status"],
         "too_short_399": prov.posteriors(np.zeros(399, np.float32), 16000)["status"],
         "ok_400_frames": prov.posteriors(np.zeros(400, np.float32), 16000)["n_frames"],
         "silence_1s_blank_argmax_fraction": float(np.mean(s_out["log_probs"].argmax(1) == table.blank_id)),
         "float64_input_equals_float32": bool(np.array_equal(prov.posteriors(noise.astype(np.float64), 16000)["log_probs"], a["log_probs"]))}
    G["pass"] = (G["refuse_44100"] == G["refuse_48000"] == G["refuse_stereo"] == G["refuse_int16"] == G["refuse_nan"] == "invalid_input"
                 and G["too_short_399"] == "too_short" and G["ok_400_frames"] == 1 and G["float64_input_equals_float32"])
    out["G"] = G

    # H: frame geometry vs the model and vs the frozen R2 sample conventions
    lens = [400, 401, 719, 720, 1040, 16000, 16001, 32000]
    model_len = {n: int(prov.model._get_feat_extract_output_lengths(torch.tensor(n))) for n in lens}
    H = {"model_vs_formula": {str(n): [model_len[n], pp.n_frames(n)] for n in lens}}
    win = (16000, 32000)                                   # a 50-encoder-frame (1 s) R2-style window
    H["r2_window_1s_provider_frames"] = int(pp.frames_in_interval(*win, pp.n_frames(48000)).size)
    enc = np.zeros(150); enc[50:100] = 1.0
    H["encoder_weights_mass_interior"] = float(pp.encoder_frame_weights(enc, pp.n_frames(48000)).sum())
    x, lp_full = samples["sample1.flac"]
    s0, s1 = 64000, 80000
    crop = prov.posteriors(np.ascontiguousarray(x[s0:s1]), 16000)["log_probs"]
    j0 = s0 // pp.FRAME_STRIDE
    H["crop_vs_full_slice_max_abs_logprob_diff"] = float(np.max(np.abs(crop - lp_full[j0:j0 + crop.shape[0]])))
    H["crop_vs_full_slice_argmax_agreement"] = float(np.mean(crop.argmax(1) == lp_full[j0:j0 + crop.shape[0]].argmax(1)))
    H["recommendation"] = ("run the provider once on the full original waveform and select frames by sample overlap with the "
                           "frozen R2 window; a crop is re-normalized and loses context, so it is not interchangeable")
    H["pass"] = all(a_ == b_ for a_, b_ in H["model_vs_formula"].values()) and H["r2_window_1s_provider_frames"] == 50 \
        and abs(H["encoder_weights_mass_interior"] - 50.0 * pp.RECEPTIVE_FIELD / pp.ENCODER_FRAME) < 1e-12
    out["H"] = H

    # feature evidence on public samples (descriptive) + deterministic fallbacks
    ev = {}
    for name, (x, lp) in samples.items():
        w = pp.interval_weights(0, x.shape[0], lp.shape[0])
        e = pp.feature_evidence(lp, w, table)
        ev[name] = {"status": e["status"], "mass": e["mass"], "q": dict(zip(table.names, map(float, e["q"])))}
    ev["fallback_no_frames"] = pp.feature_evidence(samples["sample1.flac"][1], np.zeros(samples["sample1.flac"][1].shape[0]), table)["status"]
    only_blank = np.full((3, len(table)), -1e30, dtype=np.float32); only_blank[:, table.blank_id] = 0.0
    ev["fallback_zero_valid_mass"] = pp.feature_evidence(only_blank, np.ones(3), table)["status"]
    out["feature_evidence_descriptive"] = ev

    # J: firewall (static + opened paths)
    src = (ROOT / "src/csasr/inference_cf/phone_provider.py").read_text()
    tree = ast.parse(src)
    imports = sorted({(n.module or "") if isinstance(n, ast.ImportFrom) else a.name
                      for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                      for a in (n.names if isinstance(n, ast.Import) else [n])})
    params = sorted({a.arg for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) for a in n.args.args + n.args.kwonlyargs})
    banned_params = {"text", "transcript", "reference", "references", "labels", "label", "phonemes", "alignment", "timing", "language"}
    J = {"adapter_imports": imports, "adapter_parameters": params,
         "banned_parameters_present": sorted(banned_params & set(params)),
         "phonemizer_imported": any("phonemizer" in i for i in imports),
         "phonemizer_installed": _installed("phonemizer")}
    opened = sorted(set(OPENED))
    J["opened_forbidden"] = [p for p in opened if any(m in p for m in FORBIDDEN_PATH_MARKERS)]
    J["opened_audio"] = [p for p in opened if p.endswith((".wav", ".flac"))]
    J["pass"] = not J["banned_parameters_present"] and not J["phonemizer_imported"] and not J["opened_forbidden"] \
        and all(p.startswith(str(AUDIO_DIR)) for p in J["opened_audio"])
    out["J"] = J
    out["opened_paths"] = opened
    out["elapsed_sec"] = time.time() - t0
    out["all_pass"] = all(out[k]["pass"] for k in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J"))
    atomic(CHECKS, out)
    print(json.dumps({k: out[k]["pass"] for k in "ABCDEFGHIJ"} | {"all_pass": out["all_pass"]}))
    for name, v in C["files"].items():
        if v["status"] == "ok":
            print(name, v["n_frames"], round(v["blank_argmax_fraction"], 3), round(v["cmn_convention_share_of_segmental_mass"], 3),
                  round(v["sec_per_audio_sec"], 3), v["greedy_phones"][:160])


def _installed(name: str) -> bool:
    try:
        md.version(name)
        return True
    except md.PackageNotFoundError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--acl1-before", required=True)
    e = sub.add_parser("engineering")
    e.add_argument("--threads", default=4)
    a = ap.parse_args()
    {"manifest": cmd_manifest, "engineering": cmd_engineering}[a.cmd](a)


if __name__ == "__main__":
    main()
