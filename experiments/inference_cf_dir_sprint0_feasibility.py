#!/usr/bin/env python
"""DIR-SPRINT0 CPU-only scientific feasibility audit (sprint instruction section 2).

Runs BEFORE any design freeze, runner, GPU job or new data use. It checks whether each new direction family (D3
acoustic-prior contrast, D4 transcription-fidelity contrast, D5 phonetic-concept steering) and every mandatory
historical control can be built from providers that already exist, and whether the 240-utterance population is
available. It writes ``results/inference_cf/dir_sprint0/feasibility_audit.json``.

What it reads: repository source files (hashed, parsed with ``ast``; nothing imported from them), the pinned Whisper
generation/tokenizer JSON, the config/vocabulary JSON of local auxiliary models, installed-package metadata, the
committed SRD2-G0 identity roster (utterance/dialogue IDs and exclusion reasons only), the committed P2-DIR D1 fold
record and vectors, and the committed SRD2-G0 runtime summaries. What it never does: load a model, run a forward,
read audio, read a role parquet, read a reference transcript or label, use a GPU, download anything or install
anything. Every opened path is logged through ``sys.addaudithook`` and saved in the output.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/inference_cf/dir_sprint0/feasibility_audit.json"
WHISPER = Path("/mnt/data/tungnx/whisper-large-v3")
HF_HUB = Path.home() / ".cache/huggingface/hub"
LOCAL_MODEL_ROOTS = [Path("/mnt/data/tungnx/models"), Path("/mnt/data/tungnx/cs-asr-steer/models"),
                     Path("/mnt/data/tungnx/cs-asr-steer/model_cache/huggingface/hub"), HF_HUB]
EXTRA_MODEL_DIRS = [Path("/mnt/data/tungnx/whisper-large-v3-turbo"), Path("/mnt/data/tungnx/Qwen3-ASR-0.6B"),
                    Path("/mnt/data/tungnx/Qwen3-ASR-1.7B")]
PHONETIC_DISTRIBUTIONS = ("panphon", "allosaurus", "phonemizer", "epitran", "espnet", "speechbrain", "nemo_toolkit",
                          "fairseq", "s3prl", "g2p-en", "g2pM", "phonecodes", "ipapy", "montreal-forced-aligner",
                          "charsiu", "phonecodes", "pyctcdecode")
PHONETIC_EXECUTABLES = ("espeak", "espeak-ng", "mfa", "festival")
LABEL = "DIR_SPRINT0_D5_PROVIDER_BLOCKED"

OPENED: list[str] = []


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        OPENED.append(os.fsdecode(args[0]))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def defs(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def source_check(rel: str, required: tuple[str, ...], text_markers: tuple[str, ...] = ()) -> dict:
    p = ROOT / rel
    found = defs(p)
    text = p.read_text()
    return {"path": rel, "sha256": sha(p), "required_definitions": list(required),
            "missing_definitions": [r for r in required if r not in found],
            "required_text": list(text_markers), "missing_text": [m for m in text_markers if m not in text]}


# ---- 1. Whisper task / language tokens (D3, D4, D0 prompts) ----------------------------------------------------

def whisper_tokens() -> dict:
    gen = json.loads((WHISPER / "generation_config.json").read_text())
    added = json.loads((WHISPER / "added_tokens.json").read_text())
    inv = {v: k for k, v in added.items()}
    want = {50258: "<|startoftranscript|>", 50259: "<|en|>", 50260: "<|zh|>", 50359: "<|translate|>",
            50360: "<|transcribe|>", 50364: "<|notimestamps|>", 50257: "<|endoftext|>"}
    checks = {
        "added_token_strings": {str(i): inv.get(i) for i in want},
        "added_token_strings_match": all(inv.get(i) == s for i, s in want.items()),
        "task_to_id": gen.get("task_to_id"),
        "task_to_id_match": gen.get("task_to_id") == {"transcribe": 50360, "translate": 50359},
        "lang_en_zh": [gen["lang_to_id"].get("<|en|>"), gen["lang_to_id"].get("<|zh|>")],
        "sot_notimestamps_eos": [gen.get("decoder_start_token_id"), gen.get("no_timestamps_token_id"), gen.get("eos_token_id")],
        "task_tokens_in_suppress_list": {"50359": 50359 in gen.get("suppress_tokens", []),
                                         "50360": 50360 in gen.get("suppress_tokens", [])},
        "files": {"generation_config.json": sha(WHISPER / "generation_config.json"),
                  "added_tokens.json": sha(WHISPER / "added_tokens.json")}}
    checks["prompts"] = {"c_TR (B0 forced-ZH transcribe)": [50258, 50260, 50360, 50364],
                         "c_TL (forced-ZH translate, D4)": [50258, 50260, 50359, 50364],
                         "c_EN (forced-EN transcribe, D0)": [50258, 50259, 50360, 50364]}
    checks["pass"] = bool(checks["added_token_strings_match"] and checks["task_to_id_match"]
                          and checks["lang_en_zh"] == [50259, 50260]
                          and checks["sot_notimestamps_eos"] == [50258, 50364, 50257])
    return checks


# ---- 2. D5 phonetic provider inventory -------------------------------------------------------------------------

def classify_model_dir(d: Path) -> dict:
    rec = {"path": str(d), "has_weights": False, "architectures": None, "model_type": None, "ctc_vocab": None,
           "class": "no_config", "is_phone_recognizer": False}
    files = {p.name for p in d.iterdir()} if d.is_dir() else set()
    rec["has_weights"] = any(f.endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".msgpack")) for f in files)
    if d.name.startswith("models--"):
        snaps = d / "snapshots"
        rec["hf_cache_snapshots"] = sorted(p.name for p in snaps.iterdir()) if snaps.is_dir() else []
        weights = [p for s in snaps.glob("*") for p in s.iterdir()] if snaps.is_dir() else []
        rec["has_weights"] = any(p.name.endswith((".safetensors", ".bin", ".pt")) for p in weights)
        cfgs = [p for p in weights if p.name == "config.json"]
        cfg_path = cfgs[0] if cfgs else None
    else:
        cfg_path = d / "config.json" if "config.json" in files else None
    if cfg_path is not None:
        cfg = json.loads(cfg_path.read_text())
        rec["config_sha256"] = sha(cfg_path)
        rec["architectures"] = cfg.get("architectures")
        rec["model_type"] = cfg.get("model_type")
        rec["class"] = "non_ctc_model"
        vocab = cfg_path.parent / "vocab.json"
        if rec["architectures"] and any("ForCTC" in a for a in rec["architectures"]) and vocab.exists():
            v = json.loads(vocab.read_text())
            symbols = sorted((k for k in v if not (k.startswith("<") and k.endswith(">"))), key=lambda k: v[k])
            ascii_letters = [s for s in symbols if len(s) == 1 and s.isascii() and s.isalpha()]
            ipa_like = [s for s in symbols if not s.isascii() or len(s) > 1]
            rec["ctc_vocab"] = {"sha256": sha(vocab), "size": len(v), "symbols": symbols,
                                "ascii_letter_symbols": len(ascii_letters), "ipa_or_multichar_symbols": len(ipa_like)}
            # A phone recognizer emits phone symbols (IPA / multi-character phone labels); a 26-letter Latin output
            # inventory is a romanized-grapheme CTC model whose letters have language-dependent phonetic values.
            rec["class"] = "phone_ctc" if ipa_like else "latin_grapheme_ctc"
            rec["is_phone_recognizer"] = bool(ipa_like)
    elif not rec["has_weights"]:
        rec["class"] = "no_weights"
    return rec


def d5_inventory() -> dict:
    dists = {}
    for name in PHONETIC_DISTRIBUTIONS:
        try:
            dists[name] = md.version(name)
        except md.PackageNotFoundError:
            dists[name] = None
    exes = {e: shutil.which(e) for e in PHONETIC_EXECUTABLES}
    dirs = []
    for root in LOCAL_MODEL_ROOTS:
        if root.is_dir():
            dirs += [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith((".", "datasets--"))]
    dirs += [d for d in EXTRA_MODEL_DIRS if d.is_dir()]
    models = [classify_model_dir(d) for d in dirs]
    notes = {
        "/mnt/data/tungnx/models/mms-fa": "MMS-300m CTC forced-alignment checkpoint (local copy; HF download metadata only). "
            "Output inventory = 26 lowercase Latin letters + apostrophe: romanized-grapheme posteriors, not phones. "
            "Letter-to-feature mapping is orthographic and language-dependent (e.g. pinyin b/p = unaspirated/aspirated "
            "voiceless stops vs English b/p = voicing contrast; pinyin x/q/c/z/j have no English letter value), so no "
            "reliable phoneme-to-articulatory-feature mapping exists for an EN/ZH code-switching study.",
        "/mnt/data/tungnx/cs-asr-steer/models/Qwen3-ForcedAligner-0.6B": "Text-conditioned forced aligner: needs an input "
            "transcript and returns unit timestamps; no phone inventory or phone posteriors.",
        "models--MahmoudAshraf--mms-300m-1130-forced-aligner": "HF cache entry with refs only (no snapshot / weights).",
    }
    phone_models = [m for m in models if m["is_phone_recognizer"]]
    feature_mapping = [n for n in ("panphon", "epitran", "ipapy", "phonecodes") if dists.get(n)]
    return {"phonetic_distributions": dists, "phonetic_executables": exes, "local_models": models, "notes": notes,
            "phone_recognizer_available": bool(phone_models), "phone_recognizers": [m["path"] for m in phone_models],
            "ipa_feature_mapping_available": bool(feature_mapping), "feature_mapping_packages": feature_mapping,
            "pass": bool(phone_models) and bool(feature_mapping)}


# ---- 3. existing providers for D3 / D4 / historical controls ----------------------------------------------------

def providers() -> dict:
    c = {
        "cached_branch_any_prompt": source_check("experiments/inference_cf_cached.py", ("Branch", "cached_greedy"),
                                                 ("def __init__(self, bundle, encoded, prompt: list[int], name: str)",)),
        "historical_null_audio_30s_zeros": source_check("experiments/inference_cf_p0_r2.py", ("native_lid",),
                                                        ("np.zeros(30 * bundle.sample_rate, dtype=np.float32)",)),
        "srd2_null_reuse": source_check("experiments/inference_cf_srd2_g0.py", ("capture_utterance", "cached_baseline"),
                                        ("np.zeros(30 * bundle.sample_rate, dtype=np.float32)",)),
        "d2_scratch_gradient_provider": source_check("src/csasr/inference_cf/readout.py",
                                                     ("readout_direction", "objective", "processed_logits", "ZeroProbe",
                                                      "clone_scratch", "tangent_unit", "cache_fingerprint")),
        "d0_prompt_direction": source_check("src/csasr/inference_cf/core_p1.py", ("direction", "processed_argmax")),
        "d1_crossfit_fit": source_check("src/csasr/inference_cf/unique.py", ("fit_fold",)),
        "r2_gate_g_old": source_check("src/csasr/inference_cf/srd2_g0.py",
                                      ("r2_gate", "native_preview", "CachedSolver", "NativeSite", "chord_guards")),
        "r2_primitives": source_check("src/csasr/inference_cf/core_r2.py",
                                      ("tokenizer_partition", "max_attention_window", "local_support", "conflict_from_logits")),
        "solver": source_check("experiments/inference_cf_p2r.py", ("solve_scale",)),
        "random_direction_pcg64": source_check("src/csasr/inference_cf/src_cf0_pilot.py", ("random_direction", "direction")),
        "v_ac_region_chain": source_check("src/csasr/inference_cf/r0_regions.py", ("window_grid", "classify_window", "query_mapping")),
        "v_ac_s1_association": source_check("src/csasr/inference_cf/s1_evidence.py", ("select_target", "select_offtarget")),
    }
    for v in c.values():
        v["pass"] = not v["missing_definitions"] and not v["missing_text"]
    readout = (ROOT / "src/csasr/inference_cf/readout.py").read_text()
    c["d2_scratch_gradient_provider"]["objective_is_hardcoded_J"] = "def objective(z: torch.Tensor, step: int, suppress, begin, partition: dict)" in readout
    return c


def d1_folds() -> dict:
    rec = json.loads((ROOT / "results/inference_cf/p2dir/folds_run1/folds.json").read_text())
    folds = rec["folds"]
    vec_ok = {d: sha(ROOT / f["vector_path"]) == f["vector_file_sha256"] for d, f in folds.items()}
    return {"folds_json_sha256": sha(ROOT / "results/inference_cf/p2dir/folds_run1/folds.json"),
            "folds_hash": rec.get("folds_hash"), "version": sorted({f["version"] for f in folds.values()}),
            "n_folds": len(folds), "status": dict(Counter(f["status"] for f in folds.values())),
            "vector_file_hash_matches": sum(vec_ok.values()), "dialogues": sorted(folds),
            "pass": len(folds) == 20 and all(vec_ok.values()) and all(f["status"] == "ok" for f in folds.values())}


# ---- 4. population headroom (identity only) ---------------------------------------------------------------------

def population() -> dict:
    p = json.loads((ROOT / "docs/inference_cf/SRD2_G0_POPULATION.json").read_text())
    srd2 = {r["utterance_id"] for r in p["selected"]}
    remaining, excluded_known = Counter(), Counter()
    dialogues = sorted({r["dialogue_id"] for r in p["roster"]})
    for r in p["roster"]:
        if r["exclusion_reasons"]:
            excluded_known["documented_exposure_registry"] += 1
        elif r["utterance_id"] in srd2:
            excluded_known["SRD2_G0_400"] += 1
        else:
            remaining[r["dialogue_id"]] += 1
    need = 12
    return {"source": "docs/inference_cf/SRD2_G0_POPULATION.json (committed identity roster; no role parquet read)",
            "source_sha256": sha(ROOT / "docs/inference_cf/SRD2_G0_POPULATION.json"),
            "role": p["role"], "roster_count": p["roster_count"], "roster_hash": p["roster_hash"], "dialogues": len(dialogues),
            "excluded": dict(excluded_known), "remaining_eligible_total": sum(remaining.values()),
            "remaining_per_dialogue_min": min(remaining.values()), "remaining_per_dialogue_max": max(remaining.values()),
            "remaining_per_dialogue": dict(sorted(remaining.items())), "required_per_dialogue": need,
            "pass": len(dialogues) == 20 and min(remaining.values()) >= need,
            "note": "Counts only. No utterance was selected; selection must be frozen in the design (identity-only hash rule)."}


# ---- 5. compute pre-forecast from measured SRD2-G0 runtimes ------------------------------------------------------

def compute() -> dict:
    run = ROOT / "results/inference_cf/srd2_g0/run1"
    desc = json.loads((run / "descriptive.json").read_text())
    cap = json.loads((run / "capture_runtime.json").read_text())
    pul = json.loads((run / "pulse_runtime.json").read_text())
    n_utt, n_struct = 400, desc["structural_queries"]
    pulse_mean, grad_mean = desc["timing"]["pulse_sec"]["mean"], desc["timing"]["d2_sec"]["mean"]
    srd2_overhead = (pul["elapsed_sec"] - pul["model_load_sec"] - desc["timing"]["pulse_sec"]["n"] * pulse_mean
                     - desc["timing"]["d2_sec"]["n"] * grad_mean) / n_struct
    s = round(n_struct * 240 / n_utt)
    # Assumptions (forecast, not measurement): structural queries scale with utterance count; every ungated arm
    # (D0, D1, D2, random, D3, D4, D5, D5-shuffled, v_AC) pulses every structural query (upper bound: D3/v_AC are
    # subsets); four gated arms fire on the SRD2 nonzero-gate fraction; two autograd calls per query (D2, D3).
    gate_frac = desc["gate_distribution_structural"]["g_positive_fraction"]
    pulses = s * 9 + s * 4 * gate_frac
    job_b = 15 + pulses * pulse_mean + 2 * s * grad_mean + 1.5 * s * srd2_overhead
    job_a = 15 + (cap["elapsed_sec"] - cap["model_load_sec"]) * 240 / n_utt * 2.0
    return {"basis": "measured SRD2-G0 Slurm 58300 / 58314 runtimes on the same MIG 3g.40gb type",
            "srd2_measured": {"structural_queries": n_struct, "capture_elapsed_sec": cap["elapsed_sec"],
                              "pulse_elapsed_sec": pul["elapsed_sec"], "pulse_mean_sec": pulse_mean,
                              "autograd_mean_sec": grad_mean, "per_query_overhead_sec": srd2_overhead},
            "expected_structural_queries_240": s, "forecast_pulses_upper": round(pulses),
            "job_a_forecast_sec": round(job_a), "job_b_forecast_sec": round(job_b), "limit_per_job_sec": 10800,
            "assumptions": "Job A = 2x the SRD2 per-utterance capture cost (extra null / translate / forced-EN branches, "
                           "v_AC region pass and phone provider); Job B = 9 ungated arms on every structural query "
                           "+ 4 gated arms at the SRD2 nonzero-gate fraction + 2 autograd calls per query + 1.5x SRD2 "
                           "per-query overhead (full-vocabulary outputs for more arms).",
            "status": "PRE-FORECAST ONLY (not a design-freeze projection; recost after Job A would be mandatory)"}


def main() -> None:
    sys.addaudithook(_audit)
    t0 = time.time()
    head, branch = git("rev-parse", "HEAD"), git("branch", "--show-current")
    porcelain = git("status", "--porcelain")
    tokens, d5, prov, d1, pop, comp = whisper_tokens(), d5_inventory(), providers(), d1_folds(), population(), compute()
    d3_ok = tokens["pass"] and all(prov[k]["pass"] for k in ("cached_branch_any_prompt", "historical_null_audio_30s_zeros",
                                                             "d2_scratch_gradient_provider", "solver"))
    d4_ok = tokens["pass"] and all(prov[k]["pass"] for k in ("cached_branch_any_prompt", "solver"))
    controls = {"D0": prov["d0_prompt_direction"]["pass"] and tokens["pass"], "D1": d1["pass"],
                "D2": prov["d2_scratch_gradient_provider"]["pass"],
                "R2_g_old": prov["r2_gate_g_old"]["pass"] and prov["r2_primitives"]["pass"],
                "random": prov["random_direction_pcg64"]["pass"],
                "v_AC": prov["v_ac_region_chain"]["pass"] and prov["v_ac_s1_association"]["pass"]}
    out = {
        "schema": "dir_sprint0_feasibility_v1", "label": LABEL if not d5["pass"] else "DIR_SPRINT0_FEASIBLE_TO_FREEZE",
        "git": {"head": head, "branch": branch, "clean_at_start": porcelain == "",
                "porcelain_at_start": porcelain.splitlines()},
        "families": {
            "D3_acoustic_prior_contrast": {"status": "IMPLEMENTABLE_WITH_EXISTING_PROVIDERS" if d3_ok else "BLOCKED",
                                           "new_code_needed": "objective z(c_AP)-z(b_t) through the existing ZeroProbe / "
                                           "clone_scratch / tangent_unit mechanics (readout.objective is hard-coded to J); "
                                           "null decoder branch fed the B0 content prefix on the historical 30 s zero waveform"},
            "D4_transcription_fidelity": {"status": "IMPLEMENTABLE_WITH_EXISTING_PROVIDERS" if d4_ok else "BLOCKED",
                                          "new_code_needed": "translate-prompt Branch fed the B0 content prefix; DG-02 L16 "
                                          "state capture; tangent projection as D2"},
            "D5_phonetic_concept": {"status": "PROVIDER_BLOCKED" if not d5["pass"] else "PROVIDER_AVAILABLE",
                                    "missing": [] if d5["pass"] else [
                                        "frozen acoustic phone-recognition model (no phone-output model is installed or cached)",
                                        "phone/IPA-to-articulatory-feature mapping resource (no panphon/epitran/ipapy/phonecodes, "
                                        "no espeak/espeak-ng executable)"]}},
        "historical_controls_available": controls,
        "whisper_tokens": tokens, "d5_provider_inventory": d5, "providers": prov, "d1_folds": d1,
        "population": pop, "compute_preforecast": comp,
        "data_access": {"audio_read": False, "role_parquet_read": False, "references_read": False, "model_forward": False,
                        "gpu": False, "downloads_or_installs": False},
        "elapsed_sec": None, "opened_paths": None}
    out["elapsed_sec"] = time.time() - t0
    out["opened_paths"] = sorted(set(OPENED))
    forbidden = [p for p in out["opened_paths"] if any(s in p for s in ("role_D-", ".parquet", ".wav", ".flac", "/locked/"))]
    out["forbidden_opened_paths"] = forbidden
    if forbidden:
        raise SystemExit(f"firewall violation: {forbidden}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(OUT)
    print(json.dumps({"label": out["label"], "families": {k: v["status"] for k, v in out["families"].items()},
                      "controls": controls, "population_min_per_dialogue": pop["remaining_per_dialogue_min"],
                      "job_a_forecast_sec": comp["job_a_forecast_sec"], "job_b_forecast_sec": comp["job_b_forecast_sec"],
                      "phone_recognizer_available": d5["phone_recognizer_available"],
                      "feature_mapping_available": d5["ipa_feature_mapping_available"]}, indent=1))


if __name__ == "__main__":
    main()
