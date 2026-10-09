#!/usr/bin/env python
"""Independent DIR-SPRINT0 D5 provider-compatibility audit (separate process; engineering only).

It does NOT import ``csasr.inference_cf.phone_provider`` or the feature-table builder. From the raw files it
independently re-derives:
* model and environment hashes, including the published Hugging Face LFS hash and PyPI digests (live, read-only);
* that the inventory is phonetic;
* every segmental feature vector (directly from panphon) and every tone assignment (own rule code);
* golden normalizations and contrasts for the EN/ZH-critical symbols;
* the provider posteriors (own model load and own waveform normalization) against the sealed engineering hashes;
* the frame geometry from the conv configuration;
* firewall properties, acl1 / Whisper / historical-file invariance, and the exposure addendum.

Output: ``results/inference_cf/dir_sprint0/d5_provider/provider_audit.json`` with verdict
``DIR_SPRINT0_D5_PROVIDER_AUDIT: PASS`` or ``... : FAIL``.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unicodedata
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
D5 = ROOT / "results/inference_cf/dir_sprint0/d5_provider"
OUT = D5 / "provider_audit.json"
BASE_COMMIT = "72727f347ae72e575019339d4bfb226c03875919"
ACL1_PY = "/home/tungnx/miniconda3/envs/acl1/bin/python"
MMS_VOCAB = Path("/mnt/data/tungnx/models/mms-fa/vocab.json")
GOLDEN = {"p": "p", "ph": "pʰ", "t": "t", "th": "tʰ", "k": "k", "kh": "kʰ", "b": "b", "d": "d", "ɡ": "ɡ",
          "ts": "t͡s", "tsh": "t͡sʰ", "ts.": "ʈ͡ʂ", "ts.h": "ʈ͡ʂʰ", "s.": "ʂ", "ʂ": "ʂ", "tɕ": "t͡ɕ", "tɕh": "t͡ɕʰ",
          "ɕ": "ɕ", "tʃ": "t͡ʃ", "dʒ": "d͡ʒ", "ʃ": "ʃ", "ʒ": "ʒ", "θ": "θ", "ð": "ð", "x": "x", "ɻ": "ɻ", "ʐ": "ʐ",
          "i.5": "ɻ̩", "i̪5": "ɹ̩", "ər5": "ə˞", "ɚ": "ə˞", "onɡ5": "oŋ", "uei5": "uei", "ɜ": "ɜ", "əɜ": "ə",
          "iː": "iː", "ɑ̃": "ɑ̃", "ŋ": "ŋ", "ɹ": "ɹ", "æ": "æ", "ʌ": "ʌ", "aɪ": "aɪ", "oʊ": "oʊ"}
GOLDEN_TONE = {"i.5": 5, "i̪5": 5, "ər5": 5, "onɡ5": 5, "uei5": 5, "əɜ": 3, "ɜ": None, "ɑ5": 5, "ɑ2": 2, "ɑɜ": 3,
               "ɜː": None, "iɜ": 3, "1": 1}   # "1" is the bare tone mark: status tone_only, tone 1, no features
CONTRASTS = [("pʰ", "p", "sg"), ("t͡sʰ", "t͡s", "sg"), ("ʈ͡ʂʰ", "ʈ͡ʂ", "sg"), ("t͡ɕʰ", "t͡ɕ", "sg"), ("b", "p", "voi"),
             ("ð", "θ", "voi"), ("d͡ʒ", "t͡ʃ", "voi"), ("t͡s", "s", "cont"), ("ʈ͡ʂ", "ʂ", "cont"), ("t͡ɕ", "ɕ", "cont"),
             ("t͡s", "t", "delrel"), ("ʂ", "s", "ant"), ("ɕ", "ʃ", "hi"), ("iː", "i", "long"), ("ɑ̃", "ɑ", "nas")]
FORBIDDEN = ("role_D-", ".parquet", "/CS-Dialogue", "artifacts_dialogue", "/locked/", "srd2_g0/run1/gate",
             "srd2_g0/run1/pulses", "evaluation_units")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def arr_sha(a) -> str:
    import numpy as np
    a = np.ascontiguousarray(a)
    return "sha256:" + hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def cdigest(o) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def fetch_json(url: str):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r), None
    except Exception as e:  # network is optional evidence; absence is recorded, never faked
        return None, repr(e)


def main() -> None:
    t0 = time.time()
    checks, notes = {}, {}
    man = json.loads((D5 / "provider_manifest.json").read_text(encoding="utf-8"))
    tab = json.loads((D5 / "feature_table.json").read_text(encoding="utf-8"))
    eng = json.loads((D5 / "engineering_checks.json").read_text(encoding="utf-8"))

    # 1. manifest / model / environment provenance
    checks["manifest_digest"] = man["manifest_digest"] == cdigest({k: v for k, v in man.items() if k != "manifest_digest"})
    mdir = Path(man["model"]["dir"])
    checks["model_files_match_manifest"] = all(sha(mdir / f) == h for f, h in man["model"]["files"].items())
    api, err = fetch_json(f"https://huggingface.co/api/models/facebook/wav2vec2-xlsr-53-espeak-cv-ft/revision/{man['model']['revision']}?blobs=true")
    if api is None:
        notes["hf_api"] = f"unavailable: {err}"
        checks["weights_match_published_lfs"] = man["model"]["files"]["pytorch_model.bin"] == man["model"]["published_weights_lfs_sha256"]
    else:
        lfs = {s["rfilename"]: (s.get("lfs") or {}).get("sha256") for s in api.get("siblings", [])}
        notes["hf_api"] = {"sha": api.get("sha"), "license_tags": [t for t in api.get("tags", []) if t.startswith("license")]}
        checks["weights_match_published_lfs"] = ("sha256:" + str(lfs.get("pytorch_model.bin"))) == man["model"]["files"]["pytorch_model.bin"] \
            and api.get("sha") == man["model"]["revision"] and "license:apache-2.0" in api.get("tags", [])
    wheels_ok = True
    for name, ver in (("panphon", "0.22.2"), ("unicodecsv", "0.14.1")):
        meta, err = fetch_json(f"https://pypi.org/pypi/{name}/{ver}/json")
        published = {u["filename"]: "sha256:" + u["digests"]["sha256"] for u in meta["urls"]} if meta else {}
        local = {k: v for k, v in man["environment"]["wheelhouse"].items() if k.startswith(name)}
        ok = bool(local) and all(published.get(k) == v for k, v in local.items()) if meta else None
        notes[f"pypi_{name}"] = "verified" if ok else (f"unavailable: {err}" if meta is None else "MISMATCH")
        wheels_ok = wheels_ok and ok is not False
    checks["wheels_match_pypi"] = wheels_ok
    import importlib.metadata as md
    checks["venv_packages_pinned"] = md.version("panphon") == "0.22.2" and md.version("unicodecsv") == "0.14.1"
    freeze = subprocess.check_output([ACL1_PY, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL)
    now = "sha256:" + hashlib.sha256(freeze.encode()).hexdigest()
    before = json.loads((D5 / "acl1_before_install.json").read_text())["pip_freeze_sha256"]
    checks["acl1_unchanged"] = now == before == man["environment"]["acl1_before_install"]["pip_freeze_sha256"]
    acl1_out = subprocess.check_output([ACL1_PY, "-c", "import importlib.util as u; print(u.find_spec('panphon') is None)"], text=True).strip()
    checks["acl1_has_no_panphon"] = acl1_out == "True"
    whisper = json.loads((ROOT / "results/inference_cf/srd2_g0/run1/manifest.json").read_text())["model"]
    checks["whisper_files_unchanged"] = all(sha(Path(whisper["dir"]) / f) == h for f, h in whisper["files"].items())

    # 2. phonetic inventory (A) vs the rejected grapheme aligner
    vocab = json.loads((mdir / "vocab.json").read_text(encoding="utf-8"))
    syms = [s for s, _ in sorted(vocab.items(), key=lambda kv: kv[1]) if not (s.startswith("<") and s.endswith(">"))]
    ipa_specific = [s for s in syms if any(ord(c) > 127 for c in s)]
    mms = json.loads(MMS_VOCAB.read_text()) if MMS_VOCAB.exists() else {}
    mms_ipa = [s for s in mms if not (s.startswith("<") and s.endswith(">")) and any(ord(c) > 127 for c in s)]
    notes["inventory"] = {"symbols": len(syms), "ipa_specific": len(ipa_specific), "mms_fa_ipa_specific": len(mms_ipa)}
    checks["inventory_is_phonetic"] = len(ipa_specific) >= 200 and len(mms_ipa) == 0 and len(vocab) == 392

    # 3. feature table, independently from panphon
    import panphon
    ft = panphon.FeatureTable()
    names = [n for n in ft.names if n not in ("hitone", "hireg")]
    rows = tab["rows"]
    checks["table_digest"] = tab["table_digest"] == "sha256:" + hashlib.sha256(json.dumps(
        {k: tab[k] for k in ("feature_names", "valid_mask", "rows")}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    checks["table_matches_vocab"] = [r["symbol"] for r in rows] == [s for s, _ in sorted(vocab.items(), key=lambda kv: kv[1])] \
        and tab["vocab_sha256"] == man["model"]["files"]["vocab.json"] and tab["feature_names"] == names
    bad_feat, bad_seg = [], []
    for r in rows:
        if r["status"] != "segmental":
            if r["features"] is not None or r["ipa"] is not None:
                bad_feat.append(r["symbol"])
            continue
        segs = ft.ipa_segs(r["ipa"])
        if segs != r["segments"] or "".join(segs) != r["ipa"] or not all(ft.seg_known(s) for s in segs):
            bad_seg.append(r["symbol"])
            continue
        vecs = [dict(zip(ft.names, ft.fts(s).numeric())) for s in segs]
        if [sum(v[n] for v in vecs) / len(vecs) for n in names] != r["features"]:
            bad_feat.append(r["symbol"])
    notes["table_recompute"] = {"segment_mismatch": bad_seg, "feature_mismatch": bad_feat}
    checks["table_features_recomputed_exactly"] = not bad_seg and not bad_feat
    vs = set(vocab)

    def tone_of(s):
        m = re.fullmatch(r"(.+)([1-5])", s)
        if m:
            return int(m.group(2))
        if len(s) > 1 and s.endswith("ɜ") and any(s[:-1] + d in vs for d in "1245"):
            return 3
        return None
    tone_bad = [r["symbol"] for r in rows if r["status"] == "segmental" and tone_of(r["symbol"]) != r["tone"]]
    tone_bad += [s for s, t in GOLDEN_TONE.items() if s in vs and next(r for r in rows if r["symbol"] == s)["tone"] != t]
    tone_bad += [r["symbol"] for r in rows if r["status"] == "tone_only" and (r["features"] is not None or r["tone"] is None)]
    checks["tones_independent"] = not tone_bad
    nfd = lambda s: unicodedata.normalize("NFD", s)
    byname = {r["symbol"]: r for r in rows}
    golden_bad = [s for s, ipa in GOLDEN.items() if byname[s]["status"] != "segmental" or byname[s]["ipa"] != nfd(ipa)]
    leftover = [r["symbol"] for r in rows if r["status"] == "segmental" and re.search(r"[.:\^\[\"0-9]", r["ipa"])]
    checks["golden_normalization"] = not golden_bad and not leftover
    notes["golden"] = {"bad": golden_bad, "leftover_mnemonics": leftover}
    contr = []
    for a, b, f in CONTRASTS:
        fa, fb = ft.fts(nfd(a)), ft.fts(nfd(b))
        contr.append({"a": a, "b": b, "feature": f, "panphon_differs": fa is not None and fb is not None and fa[f] != fb[f]})
    checks["contrasts_panphon"] = all(c["panphon_differs"] for c in contr)
    statuses = [r["status"] for r in rows]
    checks["every_symbol_one_status"] = len(statuses) == 392 and set(statuses) <= {"segmental", "blank", "special", "tone_only", "excluded"} \
        and statuses[0] == "blank" and all(r["reason"] for r in rows if r["status"] == "excluded") \
        and tab["valid_mask"] == [s == "segmental" for s in statuses]

    # 4. posteriors: own load + own normalization vs sealed engineering hashes (C, G, I)
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import Wav2Vec2ForCTC
    torch.set_num_threads(int(eng["torch_threads"]))
    model = Wav2Vec2ForCTC.from_pretrained(mdir, dtype=torch.float32).eval()
    pre = json.loads((mdir / "preprocessor_config.json").read_text())

    def run(x):
        x = np.asarray(x, dtype=np.float32)
        if pre["do_normalize"]:
            x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        with torch.inference_mode():
            z = model(torch.from_numpy(x)[None], attention_mask=torch.ones(1, x.size, dtype=torch.long)).logits[0].float()
            return torch.log_softmax(z, -1).numpy()
    post = {}
    adir = Path(man["engineering_audio"]["dir"])
    for name, rec in eng["C"]["files"].items():
        if rec["status"] != "ok":
            continue
        meta = man["engineering_audio"]["files"][name]
        if sha(adir / name) != meta["sha256"]:
            post[name] = "audio_hash_mismatch"
            continue
        x, sr = sf.read(str(adir / name), dtype="float32")
        lp = run(x)
        post[name] = {"bitwise_equal_sealed": arr_sha(lp) == rec["log_probs_sha256"], "frames": int(lp.shape[0])}
    rng = np.random.default_rng(240924)
    noise = (0.05 * rng.standard_normal(16000 * 3)).astype(np.float32)
    post["noise_3s"] = {"bitwise_equal_sealed": arr_sha(run(noise)) == eng["I"]["noise_3s"]["sha256"]}
    notes["posteriors"] = post
    checks["posteriors_reproduced_independently"] = all(isinstance(v, dict) and v["bitwise_equal_sealed"] for v in post.values())
    checks["engineering_checks_all_pass"] = bool(eng["all_pass"]) and eng["manifest_digest"] == man["manifest_digest"]
    cfg = model.config
    geo = {}
    for n in (400, 401, 719, 720, 1040, 16000, 16001, 48000):
        L = n
        for k, s in zip(cfg.conv_kernel, cfg.conv_stride):
            L = (L - k) // s + 1
        geo[n] = (L, int(run(np.zeros(n, np.float32) + 1e-3 * np.sin(np.arange(n))).shape[0]), (n - 400) // 320 + 1)
    checks["frame_geometry"] = all(a == b == c for a, b, c in geo.values())
    notes["frame_geometry"] = {str(k): v for k, v in geo.items()}

    # 5. firewall (J)
    def ast_info(path):
        tree = ast.parse(path.read_text())
        mods = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
            {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        params = {a.arg for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) for a in n.args.args + n.args.kwonlyargs}
        return mods, params
    amods, aparams = ast_info(ROOT / "src/csasr/inference_cf/phone_provider.py")
    bmods, _ = ast_info(ROOT / "experiments/inference_cf_dir_sprint0_d5_feature_table.py")
    banned = {"text", "transcript", "reference", "references", "label", "labels", "phonemes", "alignment", "timing", "language"}
    checks["adapter_text_free"] = not (aparams & banned) and not any("phonemizer" in m for m in amods) \
        and amods <= {"__future__", "hashlib", "json", "math", "pathlib", "typing", "numpy", "torch", "transformers"}
    checks["builder_inputs_only_vocab_and_panphon"] = bmods <= {"__future__", "argparse", "hashlib", "json", "pathlib", "re",
                                                                "unicodedata", "panphon", "importlib.metadata"}
    checks["phonemizer_absent"] = subprocess.run([sys.executable, "-c", "import phonemizer"], capture_output=True).returncode != 0
    opened = eng.get("opened_paths", [])
    checks["engineering_opened_no_forbidden"] = not [p for p in opened if any(m in p for m in FORBIDDEN)] \
        and all(p.startswith(man["engineering_audio"]["dir"]) for p in opened if p.endswith((".wav", ".flac")))
    checks["source_hashes_current"] = sha(ROOT / man["adapter"]["path"]) == man["adapter"]["sha256"] \
        and sha(ROOT / man["feature_table"]["builder"]["path"]) == man["feature_table"]["builder"]["sha256"] \
        and sha(D5 / "feature_table.json") == man["feature_table"]["file_sha256"]

    # 6. exposure addendum + historical invariance
    add = json.loads((ROOT / "docs/current/DATA_EXPOSURE_ADDENDA.json").read_text(encoding="utf-8"))
    pop = json.loads((ROOT / "docs/inference_cf/SRD2_G0_POPULATION.json").read_text())
    e = add["entries"][0]
    sel = [r["utterance_id"] for r in pop["selected"]]
    roster = {r["utterance_id"] for r in pop["roster"]}
    documented = {r["utterance_id"] for r in pop["roster"] if r["exclusion_reasons"]}
    ids_digest = "sha256:" + hashlib.sha256(json.dumps(e["utterance_ids"], separators=(",", ":")).encode()).hexdigest()
    pin = json.loads((ROOT / "configs/inference_cf/srd2_g0.json").read_text())["source_sha256"]["docs/current/DATA_EXPOSURE.md"]
    checks["addendum_srd2_400"] = e["utterance_ids"] == sel and ids_digest == e["utterance_ids_hash"] == pop["selected_ids_hash"] \
        and len(set(sel)) == 400 and set(sel) <= roster and not (set(sel) & documented) \
        and e["run_manifest"]["manifest_hash"] == json.loads((ROOT / e["run_manifest"]["path"]).read_text())["manifest_hash"] \
        and sha(ROOT / e["run_manifest"]["path"]) == e["run_manifest"]["sha256"]
    checks["pinned_ledger_unchanged"] = sha(ROOT / "docs/current/DATA_EXPOSURE.md") == pin == add["base_ledger"]["sha256"]
    porcelain = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    diff = subprocess.check_output(["git", "diff", "--name-status", BASE_COMMIT], cwd=ROOT, text=True).splitlines()
    notes["git"] = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "porcelain": porcelain, "diff_vs_base": diff}
    checks["only_new_files"] = all(l.startswith("??") for l in porcelain) and all(l.startswith("A") for l in diff)
    sq = subprocess.run(["squeue", "-u", os.environ.get("USER", "tungnx"), "-h"], capture_output=True, text=True)
    checks["no_slurm_jobs"] = sq.returncode == 0 and not sq.stdout.strip()

    verdict = "DIR_SPRINT0_D5_PROVIDER_AUDIT: PASS" if all(checks.values()) else "DIR_SPRINT0_D5_PROVIDER_AUDIT: FAIL"
    out = {"schema": "dir_sprint0_d5_provider_audit_v1", "verdict": verdict, "checks": checks, "notes": notes,
           "failed": [k for k, v in checks.items() if not v], "manifest_digest": man["manifest_digest"],
           "imports_adapter": "csasr.inference_cf.phone_provider" in sys.modules, "elapsed_sec": time.time() - t0}
    if out["imports_adapter"]:
        raise SystemExit("auditor must not import the adapter")
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    print(json.dumps({"verdict": verdict, "failed": out["failed"], "checks": len(checks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
