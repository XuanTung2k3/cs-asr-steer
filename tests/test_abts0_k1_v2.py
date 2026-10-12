"""ABTS-0 K1 evaluator amendments A1-A3 and pre-score gate G1 (synthetic, CPU, no references)."""
import itertools
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import experiments.abts0_k1_evaluate_v2 as E2  # noqa: E402
import experiments.abts0_k1_gate as G  # noqa: E402
from csasr.evaluation.mer import MATCH, SUB, align_tokens  # noqa: E402

CFG = json.loads((ROOT / "configs/inference_cf/abts0_k1.json").read_text())
GO = CFG["evaluator"]["go_MM"]


# ---- A1: co-optimal alignment geometry is exact -------------------------------------------------------------------

def _all_alignments(ref, hyp):
    """Brute force: every alignment as (cost, diagonal pairs, deleted ref indices)."""
    if not ref and not hyp:
        yield 0, frozenset(), frozenset()
        return
    n, m = len(ref), len(hyp)
    if n and m:
        for c, dg, de in _all_alignments(ref[:-1], hyp[:-1]):
            yield c + (ref[-1] != hyp[-1]), dg | {(n - 1, m - 1)}, de
    if n:
        for c, dg, de in _all_alignments(ref[:-1], hyp):
            yield c + 1, dg, de | {n - 1}
    if m:
        for c, dg, de in _all_alignments(ref, hyp[:-1]):
            yield c + 1, dg, de


@pytest.mark.parametrize("ref,hyp", [("ab", "ba"), ("abc", "xbz"), ("aab", "ab"), ("m", "xy"), ("amb", "axyb"),
                                     ("abca", "acba"), ("", "ab"), ("ab", "")])
def test_cooptimal_matches_brute_force_and_contains_frozen_path(ref, hyp):
    ref, hyp = list(ref), list(hyp)
    al = list(_all_alignments(ref, hyp))
    best = min(c for c, _, _ in al)
    diag = set().union(*[dg for c, dg, _ in al if c == best])
    dele = set().union(*[de for c, _, de in al if c == best])
    assert E2.cooptimal(ref, hyp) == (diag, dele)
    frozen = {(o.ref_idx, o.hyp_idx) for o in align_tokens(ref, hyp) if o.op in (MATCH, SUB)}
    assert frozen <= diag


# ---- A1: lexical labels ---------------------------------------------------------------------------------------------

def lab(rt, en, hyp, H, cu):
    return E2.lexical_label(rt, [bool(int(c)) for c in en], hyp, H, cu)


def test_transliterated_word_is_literal_onset_without_timing():
    r = lab(["我", "想", "meeting", "了"], "0010", ["我", "想", "米", "听", "了"], [2, 3], ["meeting"])
    assert r["label"] == "LITERAL" and r["ref_index"] == 2 and r["onset"] and r["block_b0_correct_mandarin"] == 0


def test_block_starting_on_misrecognised_mandarin_still_literal_but_later_word_is_not_onset():
    r = lab(["我", "们", "meeting", "吧"], "0010", ["我", "米", "听", "吧"], [1, 2], ["meeting"])
    assert r["label"] == "LITERAL" and r["onset"]
    r = lab(["the", "meeting", "了"], "110", ["的", "米", "了"], [0, 1], ["meeting"])
    assert r["label"] == "LITERAL" and not r["onset"]


def test_repeated_word_in_window_is_ambiguous():
    assert lab(["我", "meeting", "meeting", "了"], "0110", ["我", "米", "听", "了"], [1, 2], ["meeting"])["label"] \
        == "AMBIGUOUS_REPEAT"


def test_cooptimal_reassignment_is_ambiguous():
    # "meeting" -> 米 + ins 听 and ins 米 + 听 tie; a block of only 米 cannot claim the word
    assert lab(["meeting"], "1", ["米", "听"], [0], ["meeting"])["label"] == "AMBIGUOUS_ALIGNMENT"


def test_block_spanning_two_english_runs_is_ambiguous():
    assert lab(["ok", "的", "meeting"], "101", ["欧", "得", "米"], [0, 1, 2], ["meeting"])["label"] == "AMBIGUOUS_RUN"


def test_word_elsewhere_boundary_wrong_english_and_mandarin_speech():
    assert lab(["我", "去", "了", "school"], "0001", ["我", "区", "了", "school"], [1], ["school"])["label"] \
        == "WRONG_OCCURRENCE"
    assert lab(["我", "去", "school"], "001", ["我", "区", "school"], [1], ["school"])["label"] == "AMBIGUOUS_BOUNDARY"
    assert lab(["我", "想", "meeting", "了"], "0010", ["我", "想", "米", "听", "了"], [2, 3], ["meaning"])["label"] \
        == "WRONG_ENGLISH"
    r = lab(["我", "去", "学", "校"], "0000", ["我", "去", "学", "校"], [2, 3], ["school"])
    assert r["label"] == "EN_ON_MANDARIN_SPEECH" and r["block_b0_correct_mandarin"] == 2


def test_unmapped_or_empty_is_ambiguous():
    assert lab(["meeting"], "1", ["米"], [None], ["meeting"])["label"] == "AMBIGUOUS_UNMAPPED"
    assert lab(["meeting"], "1", ["米"], [0], [])["label"] == "AMBIGUOUS_EMPTY"


def test_label_pair_flags_proposals_over_correct_mandarin_and_rejects_latin_blocks():
    r = {"tokens": ["我", "去", "学", "校"], "is_en": [False] * 4}
    units = [{"kind": "HAN", "surface": s} for s in "我去学校"]
    out = E2.label_pair(r, [None] * 4, ["我", "去", "学", "校"], units, [0, 1, 2, 3], {"block": [2, 3], "surface": "school"})
    assert out["label"] == "EN_ON_MANDARIN_SPEECH" and out["over_correct_mandarin"] and not out["genuine"]
    units[2] = {"kind": "LATIN", "surface": "xue"}
    out = E2.label_pair(r, [None] * 4, ["我", "去", "学", "校"], units, [0, 1, 2, 3], {"block": [2, 3], "surface": "school"})
    assert out["label"] == "AMBIGUOUS_UNMAPPED"


def test_map_units_and_group_rule():
    canon = E2.canonical_units("我想 Meeting 了")
    assert E2.map_units(["我", "想", "Meeting", "了"], canon) == [0, 1, 2, 3]
    assert E2.group_of("我想去学校开个meeting")[0] == "MM"
    assert E2.group_of("hello world 我")[0] == "END"
    assert E2.run_index([True, False, True, True, False]) == [0, None, 1, 1, None]


# ---- A2/A3 ----------------------------------------------------------------------------------------------------------

BASE_M = {"reachable_genuine_runs": 39, "reachable_dialogues": 13, "scored": 20, "scored_dialogues": 8, "auroc_dialogues": 3,
          "accepted": 12, "accepted_dialogues": 6, "precision": 0.75, "genuine_recoverable_words": 8,
          "genuine_recoverable_dialogues": 4, "onset_recall": 8 / 39, "auroc_macro": 0.65}


def test_decide_v2_precedence_and_unchanged_go_numbers():
    def d(**kw):
        return E2.decide_v2(dict(BASE_M, **kw), GO, blocked=[], gate_status="PASS")
    assert d() == ("K1_PASS", [])
    assert E2.decide_v2(BASE_M, GO, blocked=["B0 != B0M"], gate_status="PASS")[0] == "K1_BLOCKED"
    assert E2.decide_v2(BASE_M, GO, blocked=[], gate_status="K1_INSUFFICIENT") == ("K1_INSUFFICIENT", ["STRUCTURAL"])
    assert d(scored=11) == ("K1_INSUFFICIENT", ["COVERAGE"])
    assert d(scored_dialogues=5) == ("K1_INSUFFICIENT", ["COVERAGE"])
    assert d(auroc_dialogues=2) == ("K1_INSUFFICIENT", ["CLASSES"])
    assert d(reachable_genuine_runs=11) == ("K1_INSUFFICIENT", ["REACHABLE"])
    assert d(onset_recall=7 / 39) == ("K1_FAIL", ["onset_recall"])
    assert d(precision=0.74) == ("K1_FAIL", ["precision"])
    assert d(accepted=11) == ("K1_FAIL", ["accepted"])
    assert d(auroc_macro=0.64) == ("K1_FAIL", ["auroc"])


# ---- G1 -------------------------------------------------------------------------------------------------------------

def _rows(spec):
    rows, groups = [], {}
    for k, (grp, dlg, valid) in enumerate(spec):
        uid = f"u{k}"
        groups[uid] = {"group": grp, "dialogue_id": dlg}
        rows.append({"utterance_id": uid, "chosen": None if valid is None else {"encoding_valid": valid}})
    return rows, groups


def test_capacity_gate_counts_only_mm_encoding_valid_chosen():
    spec = [("MM", f"d{k % 6}", True) for k in range(12)]
    rows, groups = _rows(spec)
    assert G.capacity(rows, groups, GO)["ok"]
    rows, groups = _rows(spec[:11] + [("MIX", "d0", True), ("MM", "d1", False), ("MM", "d2", None)])
    cap = G.capacity(rows, groups, GO)
    assert not cap["ok"] and cap["mm_encoding_valid_chosen"] == 11
    rows, groups = _rows([("MM", "d0", True)] * 7 + [("MM", f"d{k}", True) for k in range(1, 6)])
    cap = G.capacity(rows, groups, GO)
    assert cap["mm_dialogues"] == 6 and cap["mm_dialogues_with_2"] == 1 and not cap["ok"]


def test_amendment_seal_detects_any_change(tmp_path, monkeypatch):
    from csasr.inference_cf.core import digest, file_hash
    f = tmp_path / "x.py"
    f.write_text("a = 1\n")
    seal = {"files": {"x.py": file_hash(f)}, "plan_sources": {}, "status": "READY_FOR_K1_CAPTURE"}
    seal["seal_hash"] = digest(seal)
    (tmp_path / "seal.json").write_text(json.dumps(seal))
    monkeypatch.setattr(G, "ROOT", tmp_path)
    monkeypatch.setattr(G, "AMEND_SEAL", "seal.json")
    assert G.verify_amendment_seal()["seal_hash"] == seal["seal_hash"]
    f.write_text("a = 2\n")
    with pytest.raises(ValueError):
        G.verify_amendment_seal()


def test_gate_and_runner_stay_reference_free():
    gate = (ROOT / "experiments/abts0_k1_gate.py").read_text()
    for bad in ("load_references", "reference_context", "oracle_tracks", "transcript_raw", "unit_status", "evaluation_units"):
        assert bad not in gate, bad
    for p in ("experiments/abts0_k1.py", "src/csasr/inference_cf/abts0_k1.py"):
        src = (ROOT / p).read_text()
        for bad in ("abts0_k1_gate", "abts0_k1_evaluate_v2", "groups_prerun"):
            assert bad not in src, (p, bad)
    assert set(G.AMEND_FILES) >= {"experiments/abts0_k1_evaluate.py", "experiments/abts0_k1_evaluate_v2.py",
                                  "slurm/abts0_k1.sbatch", "results/inference_cf/abts0_k1/groups_prerun.json"}
    sb = (ROOT / "slurm/abts0_k1.sbatch").read_text()
    assert re.search(r"capture\)\n.*abts0_k1_gate.py verify-amendments", sb)
    assert re.search(r"score\)\n(.*\n){2}.*abts0_k1_gate.py verify", sb)


def test_runtime_constants_unchanged_by_amendments():
    from csasr.inference_cf import abts0_k1 as K
    assert K.ONSET_TOL == 3200 and E2.SENSITIVITY_TOL == 6400
    assert CFG["evaluator"]["go_MM"] == {"accepted_min": 12, "accepted_dialogues_min": 6, "precision_min": 0.75,
                                         "genuine_recoverable_words_min": 6, "genuine_recoverable_dialogues_min": 4,
                                         "reachable_recall_min": 0.2, "auroc_min": 0.65, "auroc_dialogues_min": 3}
