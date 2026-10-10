"""DIR-SPRINT0 design-freeze contract (CPU; metadata, tokenizer and frozen artifacts only)."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from csasr.inference_cf.core import digest, file_hash  # noqa: E402

CONFIG = ROOT / "configs/inference_cf/dir_sprint0.json"
FREEZE = ROOT / "docs/inference_cf/DIR_SPRINT0_FREEZE.json"
POPULATION = ROOT / "docs/inference_cf/DIR_SPRINT0_POPULATION.json"
MODEL = Path("/mnt/data/tungnx/whisper-large-v3")


@pytest.fixture(scope="module")
def cfg():
    return json.loads(CONFIG.read_text())


@pytest.fixture(scope="module")
def pop():
    return json.loads(POPULATION.read_text())


def test_freeze_index_files_unchanged():
    fr = json.loads(FREEZE.read_text())
    assert fr["freeze_index_hash"] == digest(fr["files"])
    for rel, h in fr["files"].items():
        assert file_hash(ROOT / rel) == h, rel
    for must in ("configs/inference_cf/dir_sprint0.json", "docs/inference_cf/DIR_SPRINT0_SPEC.md", "docs/inference_cf/DIR_SPRINT0_DESIGN.md",
                 "docs/inference_cf/DIR_SPRINT0_FIREWALL.md", "docs/inference_cf/DIR_SPRINT0_COMPUTE_FEASIBILITY.md",
                 "docs/inference_cf/DIR_SPRINT0_POPULATION.json", "results/inference_cf/dir_sprint0/design/d5_concept_table.json"):
        assert must in fr["files"]


def test_population_contract(cfg, pop):
    pc = cfg["population"]
    assert file_hash(POPULATION) == pc["file_sha256"]
    assert digest({k: v for k, v in pop.items() if k != "manifest_hash"}) == pop["manifest_hash"] == pc["manifest_hash"]
    sel = pop["selected"]
    assert digest(sel) == pc["selected_hash"] and digest([r["utterance_id"] for r in sel]) == pc["selected_ids_hash"]
    assert digest(pop["roster"]) == pc["roster_hash"] and pop["roster_count"] == 7919
    assert len(sel) == 240 and set(Counter(r["dialogue_id"] for r in sel).values()) == {12} and len({r["dialogue_id"] for r in sel}) == 20
    assert [r["canonical_index"] for r in sel] == list(range(240))
    assert all(r["sample_rate"] == 16000 and r["channels"] == 1 for r in sel)
    bank = pop["calibration_bank"]["rows"]
    assert len(bank) == 300 and digest(bank) == pc["calibration_bank"]["rows_hash"]
    assert not {r["utterance_id"] for r in bank} & {r["utterance_id"] for r in sel}


def test_selection_recomputed_and_exposure_excluded(cfg, pop):
    from csasr.inference_cf.exposure_registry import load_addenda
    # the append-only registry as of the freeze: documented registry + the addenda entries named in by_source
    # (later appended entries, e.g. this sprint's own exposure, are allowed and ignored here)
    add = load_addenda(ROOT)
    reg = json.loads((ROOT / add["base_machine_registry"]["path"]).read_text())
    frozen_ids = {r["utterance_id"] for r in reg["roster"] if r["exclusion_reasons"]}
    names = set(pop["exclusion_registry"]["by_source"]) - {"documented_registry_at_srd2_freeze"}
    for e in add["entries"]:
        if e["addendum_id"] in names:
            frozen_ids |= set(e["utterance_ids"])
    assert digest(sorted(frozen_ids)) == pop["exclusion_registry"]["exposed_ids_hash"] == cfg["population"]["exclusion_registry"]["exposed_ids_hash"]
    assert len(frozen_ids) == pop["exclusion_registry"]["exposed_ids_count"] == 700
    sel = [r["utterance_id"] for r in pop["selected"]]
    assert not set(sel) & frozen_ids
    excluded = {r["utterance_id"] for r in pop["roster"] if r["exclusion_reasons"]}
    assert frozen_ids <= excluded
    elig = {}
    for r in pop["roster"]:
        if not r["exclusion_reasons"]:
            elig.setdefault(r["dialogue_id"], []).append(r["utterance_id"])
    own = []
    for d in sorted(elig):
        own += sorted(elig[d], key=lambda u: (hashlib.sha256(f"DIR-SPRINT0-population-v1|240924|{u}".encode()).hexdigest(), u))[:12]
    assert own == sel
    bank = sorted(r["utterance_id"] for r in pop["calibration_bank"]["rows"])
    assert digest(bank) == cfg["population"]["calibration_bank"]["ids_hash_sorted"]
    assert set(bank) == {r["utterance_id"] for r in reg["roster"] if r["exclusion_reasons"]}   # the bank is exactly the documented FULL300


def test_historical_and_design_pins(cfg):
    for rel, h in cfg["source_sha256"].items():
        assert file_hash(ROOT / rel) == h, rel
    d5 = cfg["families"]["D5"]
    for key in ("provider_manifest", "feature_table", "concept_table", "threshold_engineering"):
        assert file_hash(ROOT / d5[key]) == d5[f"{key}_sha256"], key
    ct = json.loads((ROOT / d5["concept_table"]).read_text())
    dig = "sha256:" + hashlib.sha256(json.dumps({k: ct[k] for k in ("classes", "rules", "rows", "feature_table_digest")},
                                                sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert dig == ct["concept_table_digest"] == d5["concept_table_digest"]
    assert ct["feature_table_digest"] == d5["feature_table_digest"]
    assert ct["primary"] == d5["primary_concepts"] == ["NAS", "STOP", "FRIC", "LAB", "COR", "DOR"]
    assert ct["descriptive_only"] == d5["descriptive_only"] == ["VOI_OBS", "ASP"]


@pytest.mark.skipif(not (MODEL / "vocab.json").exists(), reason="pinned Whisper tokenizer files unavailable")
def test_d3_legal_static_and_prompts(cfg):
    from transformers import WhisperTokenizer
    from transformers.models.whisper.tokenization_whisper import bytes_to_unicode
    tok = WhisperTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    gen = json.loads((MODEL / "generation_config.json").read_text())
    dec = {v: k for k, v in bytes_to_unicode().items()}
    sup, special = set(gen["suppress_tokens"]), set(tok.all_special_ids)
    legal = []
    for i in range(50257):
        if i in sup or i in special:
            continue
        b = bytes(dec[c] for c in tok.convert_ids_to_tokens(i))
        if not b or 0x80 <= b[0] <= 0xBF:
            continue
        try:
            b.decode("utf-8")
            legal.append(i)
        except UnicodeDecodeError as e:
            if e.reason == "unexpected end of data" and e.end == len(b):
                legal.append(i)
    ls = cfg["families"]["D3"]["legal_static"]
    assert len(legal) == ls["count"] and digest(legal) == ls["hash"]
    for name, br in cfg["branches"].items():
        if isinstance(br, dict) and "tokens" in br:
            assert tok.convert_ids_to_tokens(br["prompt"]) == br["tokens"], name
    assert gen["task_to_id"] == {"transcribe": 50360, "translate": 50359}


def test_arms_statistics_and_precedence(cfg):
    assert cfg["arm_order"] == ["D0", "D1", "D2", "VAC", "RND", "D3", "D4", "D5", "D5SH", "D2G", "D3G", "D4G", "D5G", "D5SHG", "RNDG"]
    assert set(cfg["arms"]) == set(cfg["arm_order"]) | {"B0", "D3CD"}
    assert all(cfg["arms"][a]["target"] == ("e_star*g_old" if a.endswith("G") else "e_star") for a in cfg["arm_order"])
    assert cfg["dose"]["e_star"] == 1.1260757575454359
    st = cfg["statistics"]
    assert st["family_size"] == 30 and math.isclose(st["family_quantiles"][0], 0.05 / 60) and st["bootstrap_replicates"] == 10000
    assert cfg["mechanistic"]["family_size"] == 6 and math.isclose(cfg["mechanistic"]["quantiles"][0], 0.05 / 12)
    assert cfg["terminal_precedence"][0] == "DIR_SPRINT0_INVALID" and cfg["terminal_precedence"][-1] == "DIR_SPRINT0_ALL_DIRECTIONS_INEFFECTIVE"
    assert len(cfg["terminal_precedence"]) == 7
    assert cfg["power"] == {"corrections_min": 8, "correction_dialogues_min": 4, "max_correction_dialogue_share": 0.5}
    assert cfg["opportunities"]["mapped_EN_confusion_min"] == 150 and cfg["opportunities"]["mapped_EN_confusion_dialogues_min"] == 10
    assert cfg["compute"]["max_scientific_jobs"] == 2 and cfg["compute"]["seconds_max_each"] == 10800
    assert cfg["firewall"]["authorization_B_fields"][-1] == "authorized"
    for role in ("D-dev-confirm", "D-test", "router-calib", "P3"):
        assert role in cfg["firewall"]["forbidden_roles"]
