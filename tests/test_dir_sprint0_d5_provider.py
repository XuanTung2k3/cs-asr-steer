"""DIR-SPRINT0 D5 provider engineering tests (CPU; synthetic inputs; no CS-Dialogue data, references or outcomes)."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from csasr.inference_cf import exposure_registry as er
from csasr.inference_cf import phone_provider as pp

D5 = ROOT / "results/inference_cf/dir_sprint0/d5_provider"
TABLE = D5 / "feature_table.json"
MANIFEST = D5 / "provider_manifest.json"


@pytest.fixture(scope="module")
def table():
    return pp.FeatureTable(TABLE)


def _row(table, sym):
    i = table.symbols.index(sym)
    return dict(zip(table.names, table.F[i])), int(table.tone[i]), table.status[i]


# ---- frozen table -------------------------------------------------------------------------------------------------

def test_table_covers_every_symbol_once(table):
    assert len(table) == 392 and table.blank_id == 0 and table.status[0] == "blank"
    assert set(table.status.tolist()) <= set(pp.STATUS_KINDS)
    raw = json.loads(TABLE.read_text(encoding="utf-8"))
    assert raw["status_counts"] == {"blank": 1, "excluded": 32, "segmental": 355, "special": 3, "tone_only": 1}
    assert all(r["reason"] for r in raw["rows"] if r["status"] == "excluded")
    assert not table.valid[[table.symbols.index(s) for s in ("<pad>", "<s>", "</s>", "<unk>", "1", "??", "ʲ")]].any()


@pytest.mark.parametrize("a,b,feat", [("p", "ph", "sg"), ("ts.", "ts.h", "sg"), ("tɕ", "tɕh", "sg"), ("p", "b", "voi"),
                                      ("θ", "ð", "voi"), ("ts", "s", "cont"), ("ts.", "s.", "cont"), ("tɕ", "ɕ", "cont"),
                                      ("tʃ", "ʃ", "cont"), ("ts", "t", "delrel"), ("s", "s.", "ant"), ("ɕ", "ʃ", "hi"),
                                      ("i", "iː", "long"), ("ɑ", "ɑ̃", "nas")])
def test_contrasts_not_collapsed(table, a, b, feat):
    assert _row(table, a)[0][feat] != _row(table, b)[0][feat]


def test_tone_is_separated_from_segmental_features(table):
    fa, ta, _ = _row(table, "ɑ5")
    fb, tb, _ = _row(table, "ɑ2")
    fc, tc, _ = _row(table, "ɑɜ")
    assert fa == fb == fc and (ta, tb, tc) == (5, 2, 3)
    assert _row(table, "ɜ")[1] == 0 and _row(table, "ɜ")[2] == "segmental"        # a real vowel, not tone 3
    assert _row(table, "əɜ")[1] == 3
    assert "hitone" not in table.names and "hireg" not in table.names


def test_espeak_mnemonics_normalized(table):
    raw = {r["symbol"]: r for r in json.loads(TABLE.read_text(encoding="utf-8"))["rows"]}
    import unicodedata
    nfd = lambda s: unicodedata.normalize("NFD", s)
    expect = {"ph": "pʰ", "th": "tʰ", "kh": "kʰ", "tsh": "t͡sʰ", "ts.": "ʈ͡ʂ", "ts.h": "ʈ͡ʂʰ", "s.": "ʂ", "tɕh": "t͡ɕʰ",
              "i.5": "ɻ̩", "ər2": "ə˞", "onɡ5": "oŋ", "tS": "t͡ʃ", "dZ": "d͡ʒ", "S": "ʃ", "N": "ŋ", "t[": "t̪", "ᵻ": "ɪ̈",
              "ts": "t͡s", "dʒ": "d͡ʒ", "u:": "uː"}
    for sym, ipa in expect.items():
        assert raw[sym]["ipa"] == nfd(ipa), sym
    for r in raw.values():
        if r["status"] == "segmental":
            assert not any(c in r["ipa"] for c in ".:^[\"0123456789"), r["symbol"]


def test_table_digest_tamper_rejected(tmp_path):
    t = json.loads(TABLE.read_text(encoding="utf-8"))
    t["rows"][5]["features"][0] = -t["rows"][5]["features"][0]
    p = tmp_path / "t.json"
    p.write_text(json.dumps(t, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(pp.ProviderError):
        pp.FeatureTable(p)


def test_table_rebuild_is_deterministic():
    pytest.importorskip("panphon")
    sys.path.insert(0, str(ROOT / "experiments"))
    import inference_cf_dir_sprint0_d5_feature_table as b
    vocab = Path(json.loads(MANIFEST.read_text())["model"]["dir"]) / "vocab.json"
    if not vocab.exists():
        pytest.skip("provider files not present")
    assert b.build(vocab)["table_digest"] == json.loads(TABLE.read_text(encoding="utf-8"))["table_digest"]


# ---- geometry -----------------------------------------------------------------------------------------------------

def test_frame_geometry():
    assert [pp.n_frames(n) for n in (0, 399, 400, 719, 720, 16000)] == [0, 0, 1, 1, 2, 49]
    sp = pp.frame_spans(3)
    assert sp.tolist() == [[0, 400], [320, 720], [640, 1040]]
    assert pp.frames_in_interval(16000, 32000, 149).tolist() == list(range(50, 100))
    w = pp.interval_weights(320, 720, 3)
    assert np.allclose(w, [80 / 400, 1.0, 80 / 400])
    enc = np.zeros(10); enc[3] = 1.0
    ew = pp.encoder_frame_weights(enc, pp.n_frames(3200))
    assert np.isclose(ew.sum(), 400 / 320) and np.isclose(ew[3], 1.0) and np.isclose(ew[2], 80 / 320)
    with pytest.raises(ValueError):
        pp.encoder_frame_weights([-1.0], 3)


# ---- feature evidence (hand-computed) -----------------------------------------------------------------------------

def _tiny_table(tmp_path):
    rows = [{"id": 0, "symbol": "<pad>", "status": "blank", "tone": None, "features": None},
            {"id": 1, "symbol": "A", "status": "segmental", "tone": 2, "features": [1.0, -1.0]},
            {"id": 2, "symbol": "B", "status": "segmental", "tone": None, "features": [-1.0, 0.5]},
            {"id": 3, "symbol": "C", "status": "excluded", "tone": None, "features": None}]
    t = {"feature_names": ["f1", "f2"], "valid_mask": [False, True, True, False], "rows": rows, "blank_id": 0}
    t["table_digest"] = pp.canonical_digest_table(t)
    p = tmp_path / "tiny.json"
    p.write_text(json.dumps(t), encoding="utf-8")
    return pp.FeatureTable(p)


def test_feature_evidence_hand_computed(tmp_path):
    tb = _tiny_table(tmp_path)
    P = np.array([[0.5, 0.3, 0.1, 0.1], [0.2, 0.0, 0.6, 0.2]])
    e = pp.feature_evidence(np.log(np.where(P > 0, P, 1e-300)), [1.0, 0.5], tb)
    mA, mB = 0.3 + 0.5 * 0.0, 0.1 + 0.5 * 0.6
    assert e["status"] == "ok"
    assert np.isclose(e["mass"]["segmental"], mA + mB) and np.isclose(e["mass"]["blank"], 0.6)
    assert np.isclose(e["mass"]["excluded"], 0.2)
    assert np.allclose(e["q"], [(mA * 1 + mB * -1) / (mA + mB), (mA * -1 + mB * 0.5) / (mA + mB)])
    assert np.isclose(e["tone_mass"]["2"], mA) and np.isclose(e["tone_mass"]["0"], mB)


def test_feature_evidence_fallbacks(tmp_path):
    tb = _tiny_table(tmp_path)
    lp = np.log(np.array([[1.0, 1e-300, 1e-300, 1e-300]]))
    assert pp.feature_evidence(lp, [0.0], tb)["status"] == "no_frames"
    z = np.full((1, 4), -1e30); z[0, 0] = 0.0                  # finite; exp underflows to exactly 0
    out = pp.feature_evidence(z, [1.0], tb)
    assert out["status"] == "zero_valid_mass" and out["q"] is None
    z[0, 1] = -np.inf                                            # -inf is never a real log-softmax output
    assert pp.feature_evidence(z, [1.0], tb)["status"] == "nonfinite_posteriors"
    assert pp.feature_evidence(np.full((1, 4), np.nan), [1.0], tb)["status"] == "nonfinite_posteriors"
    with pytest.raises(ValueError):
        pp.feature_evidence(lp, [1.0, 1.0], tb)


# ---- manifest / firewall ------------------------------------------------------------------------------------------

def test_manifest_digest_and_pins(tmp_path):
    m = pp.load_manifest(MANIFEST)
    assert m["model"]["revision"] == "2c733782da5604684829819a5eb744c193fe9398"
    assert m["model"]["files"]["pytorch_model.bin"] == m["model"]["published_weights_lfs_sha256"]
    assert m["feature_table"]["table_digest"] == json.loads(TABLE.read_text(encoding="utf-8"))["table_digest"]
    bad = dict(m); bad["model"] = dict(m["model"], revision="main")
    p = tmp_path / "m.json"; p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(pp.ProviderError):
        pp.load_manifest(p)


def test_adapter_is_text_free():
    src = (ROOT / "src/csasr/inference_cf/phone_provider.py").read_text()
    tree = ast.parse(src)
    params = {a.arg for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) for a in n.args.args + n.args.kwonlyargs}
    assert not params & {"text", "transcript", "reference", "references", "label", "labels", "phonemes", "alignment",
                         "timing", "language"}
    mods = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
        {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any("phonemizer" in m for m in mods)
    assert "Wav2Vec2PhonemeCTCTokenizer" not in src.split('"""', 2)[2]


# ---- model-backed (skip when the provider files are absent) -------------------------------------------------------

@pytest.fixture(scope="module")
def provider():
    m = json.loads(MANIFEST.read_text())
    if not (Path(m["model"]["dir"]) / "pytorch_model.bin").exists():
        pytest.skip("provider weights not present on this machine")
    return pp.PhoneProvider(MANIFEST, TABLE, device="cpu", verify_weights=False)


def test_provider_silence_invalid_and_determinism(provider):
    s = provider.posteriors(np.zeros(16000, np.float32), 16000)
    assert s["status"] == "ok" and s["n_frames"] == 49 and np.all(s["log_probs"].argmax(1) == 0)
    assert provider.posteriors(np.zeros(16000, np.float32), 44100)["status"] == "invalid_input"
    assert provider.posteriors(np.zeros((16000, 2), np.float32), 16000)["status"] == "invalid_input"
    assert provider.posteriors(np.zeros(399, np.float32), 16000)["status"] == "too_short"
    x = (0.05 * np.random.default_rng(0).standard_normal(8000)).astype(np.float32)
    a, b = provider.posteriors(x, 16000), provider.posteriors(x, 16000)
    assert np.array_equal(a["log_probs"], b["log_probs"]) and a["log_probs_sha256"] == b["log_probs_sha256"]
    assert np.allclose(np.log(np.exp(a["log_probs"].astype(np.float64)).sum(1)), 0, atol=1e-5)


# ---- exposure addenda ---------------------------------------------------------------------------------------------

def test_exposure_registry_unions_ledger_and_addendum():
    reg = er.exposed_ids(ROOT)
    pop = json.loads((ROOT / "docs/inference_cf/SRD2_G0_POPULATION.json").read_text())
    srd2 = {r["utterance_id"] for r in pop["selected"]}
    documented = {r["utterance_id"] for r in pop["roster"] if r["exclusion_reasons"]}
    assert reg["by_source"] == {"documented_registry_at_srd2_freeze": 300, "A1-SRD2-G0-400": 400}
    assert reg["ids"] == frozenset(srd2 | documented) and len(reg["ids"]) == 700 and reg["role"] == "D-dev-select"
    pin = json.loads((ROOT / "configs/inference_cf/srd2_g0.json").read_text())["source_sha256"]["docs/current/DATA_EXPOSURE.md"]
    assert reg["base_ledger_sha256"] == pin == "sha256:" + hashlib.sha256((ROOT / "docs/current/DATA_EXPOSURE.md").read_bytes()).hexdigest()


def test_exposure_registry_rejects_tampering(tmp_path):
    for rel in ("docs/current/DATA_EXPOSURE.md", er.ADDENDA, "docs/inference_cf/SRD2_G0_POPULATION.json"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    er.exposed_ids(tmp_path)
    add = json.loads((tmp_path / er.ADDENDA).read_text())
    add["entries"][0]["utterance_ids"] = add["entries"][0]["utterance_ids"][:-1]
    (tmp_path / er.ADDENDA).write_text(json.dumps(add))
    with pytest.raises(er.ExposureRegistryError):
        er.exposed_ids(tmp_path)
    shutil.copy(ROOT / er.ADDENDA, tmp_path / er.ADDENDA)
    with open(tmp_path / "docs/current/DATA_EXPOSURE.md", "a") as f:
        f.write("\nedit\n")
    with pytest.raises(er.ExposureRegistryError):
        er.exposed_ids(tmp_path)
