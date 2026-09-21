#!/usr/bin/env python3
"""Freeze the Phase-B input/settings manifest before confirmation execution."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/a6_ott_upper_bound"

def fsha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

def atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)

def main() -> None:
    elig = OUT / "eligibility"
    settings = {}
    for model, filename in (("whisper", "WHISPER_SELECTION.json"), ("qwen3_asr_1p7b", "QWEN_SELECTION.json")):
        sel_path = OUT / "search" / filename
        sel = json.loads(sel_path.read_text())
        rows = []
        for family_side, candidates in sel["top2_by_family"].items():
            family, side = family_side.split("/")
            for item in candidates:
                rows.append({"family": family, "side": side, "layer": int(item["layer"]), "rho": float(item["rho"])})
        settings[model] = {"schema_version": "a6_ott_phase_b_settings_v1", "status": "FROZEN_FROM_PHASE_A_SELECTION", "model": model, "settings": rows, "phase_a_selection_sha256": fsha(sel_path)}
        atomic(OUT / "confirm" / f"PHASE_B_SETTINGS_{model}.json", settings[model])
    manifest = {
        "schema_version": "a6_ott_phase_b_manifest_v3",
        "status": "FROZEN_BEFORE_CONFIRM_EXECUTION",
        "phase": "B", "regime": "oracle_tt", "selection_outcomes_loaded": "phase_a_only",
        "phase_a_search_freeze": fsha(OUT / "phase_a" / "SEARCH_MANIFEST_FINGERPRINT.json"),
        "phase_a_selection": {m: settings[m]["phase_a_selection_sha256"] for m in settings},
        "eligibility": {p.name: fsha(p) for p in sorted(elig.glob("CS_CONFIRM_*.json")) + sorted(elig.glob("ASCEND_CONFIRM_*.json"))},
        "settings": settings,
        "datasets": {"cs_dialogue_confirm": "CS_CONFIRM_280", "ascend_confirm": "ASCEND_CONFIRM_200"},
        "decode_modes": ["greedy", "official_standard"],
        "qwen_official_equivalence": fsha(OUT / "preflight" / "PREFLIGHT_MANIFEST.json"),
        "forbidden_inputs": ["phase_c", "transfer", "seame", "d-test", "phase_b_outcomes"],
        "fixed_direction_artifact": None,
    }
    manifest["manifest_sha256"] = "sha256:" + hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    atomic(OUT / "confirm" / "PHASE_B_MANIFEST.json", manifest)
    atomic(OUT / "confirm" / "PHASE_B_FINGERPRINT.json", {"schema_version": "a6_ott_phase_b_fingerprint_v1", "status": "IMMUTABLE", "manifest_sha256": manifest["manifest_sha256"], "settings": {m: settings[m]["settings"] for m in settings}, "search_phase_immutable": True})
    print(json.dumps({"status": manifest["status"], "manifest_sha256": manifest["manifest_sha256"], "settings": {m: len(settings[m]["settings"]) for m in settings}}, indent=2))

if __name__ == "__main__":
    main()
