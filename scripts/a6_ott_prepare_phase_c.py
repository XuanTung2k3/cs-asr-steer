#!/usr/bin/env python3
"""Freeze Phase-C inputs after Phase-B final settings, before SEAME rows."""
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
    final = json.loads((OUT / "confirm" / "PHASE_B_FINAL_SETTINGS.json").read_text())
    if final.get("status") != "FROZEN_BEFORE_PHASE_C":
        raise RuntimeError("Phase B final settings are not frozen")
    elig = {p.name: fsha(p) for p in sorted((OUT / "eligibility").glob("SEAME_*.json"))}
    payload = {
        "schema_version": "a6_ott_phase_c_manifest_v1", "status": "FROZEN_BEFORE_TRANSFER_EXECUTION",
        "phase": "C", "regime": "oracle_tt",
        "phase_b_final_settings": fsha(OUT / "confirm" / "PHASE_B_FINAL_SETTINGS.json"),
        "seame_eligibility": elig,
        "datasets": {"seame_dev_man": "SEAME-man", "seame_dev_sge": "SEAME-sge"},
        "decode_modes": ["greedy", "official_standard"], "seame_outcomes_loaded": False,
        "fixed_direction_artifact": None,
    }
    payload["manifest_sha256"] = "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    atomic(OUT / "transfer" / "PHASE_C_MANIFEST.json", payload)
    atomic(OUT / "transfer" / "PHASE_C_FINGERPRINT.json", {"schema_version": "a6_ott_phase_c_fingerprint_v1", "status": "IMMUTABLE", "manifest_sha256": payload["manifest_sha256"], "phase_b_final_settings": final["settings"]})
    print(json.dumps({"status": payload["status"], "manifest_sha256": payload["manifest_sha256"]}, indent=2))

if __name__ == "__main__":
    main()
