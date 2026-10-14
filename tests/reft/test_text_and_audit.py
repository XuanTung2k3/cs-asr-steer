"""CPU tests for ReFT text primitives, strata, error kinds, panel sampling and write-once I/O (no model, no data)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.common import write_json  # noqa: E402
from reft.data_audit import round_robin, split_dialogues  # noqa: E402
from reft.text import (en_error_kind, fd_category, first_divergence, one, target_text, token_is_lexical,  # noqa: E402
                       token_unit_map, tokens_format_equivalent, units, utterance_stratum)

FILL = {"嗯", "哦", "um", "uh", "yeah", "okay", "ok"}
EOS = 0


class Tok:
    """Toy tokenizer: each id maps to a byte string; id 0 is EOS."""

    def __init__(self, pieces):
        self.vocab = {i + 1: p.encode() for i, p in enumerate(pieces)}
        self.ids = {p: i for i, p in self.vocab.items()}

    def __call__(self, i):
        return b"" if i == EOS else self.vocab[i]

    def enc(self, *pieces):
        return [self.ids[p.encode()] for p in pieces]


def test_target_text_keeps_case_drops_punct_and_han_spaces():
    t = target_text("哦，你说的这些 <SPK/> ，我所了解的 A I，它 OK。")
    assert t == "哦你说的这些我所了解的A I它OK"
    assert units(t) == units("哦，你说的这些 <SPK/> ，我所了解的 A I，它 OK。")


def test_utterance_strata():
    assert utterance_stratum("我想 go home", FILL)["stratum"] == "CS"
    assert utterance_stratum("嗯 okay", FILL)["stratum"] == "OTHER"
    assert utterance_stratum("我想回家 yeah", FILL)["stratum"] == "ZH_ONLY"
    assert utterance_stratum("I want to go", FILL)["stratum"] == "EN_ONLY"


def test_first_divergence_lexical_token_and_category():
    tk = Tok(["我", "想", " go", " went", "去", " home"])
    ref = tk.enc("我", "想", " go", " home") + [EOS]
    base = tk.enc("我", "想", " went", " home") + [EOS]
    fd = first_divergence(ref, base, tk, EOS, lambda u: u in FILL)
    assert fd["cls"] == "LEXICAL_FIRST" and fd["sub"] == "TOKEN" and fd["k"] == 2
    assert fd["ref_unit"] == "go" and fd["base_unit"] == "went"
    assert fd_category(fd) == "en_lexical_substitution"
    base2 = tk.enc("我", "想", "去", " home") + [EOS]
    fd2 = first_divergence(ref, base2, tk, EOS, lambda u: u in FILL)
    assert fd_category(fd2) == "translation_like"
    base3 = tk.enc("我", "想", " home") + [EOS]
    fd3 = first_divergence(ref, base3, tk, EOS, lambda u: u in FILL)
    assert fd3["k"] == 2 and fd_category(fd3) == "en_omission"


def test_first_divergence_never_skips_format_first():
    tk = Tok(["我", "，", "想", " go", " went"])
    ref = tk.enc("我", "想", " go") + [EOS]
    base = tk.enc("我", "，", "想", " went") + [EOS]
    fd = first_divergence(ref, base, tk, EOS, lambda u: u in FILL)
    assert fd["cls"] == "FORMAT_FIRST" and fd["k"] == 1


def test_identical_and_format_only():
    tk = Tok(["我", "想", " Go", " go"])
    ref = tk.enc("我", "想", " go") + [EOS]
    assert first_divergence(ref, list(ref), tk, EOS, lambda u: False)["cls"] == "IDENTICAL"
    base = tk.enc("我", "想", " Go") + [EOS]
    assert first_divergence(ref, base, tk, EOS, lambda u: False)["cls"] == "FORMAT_ONLY"


def test_en_error_kinds():
    o = one("我 想 go home now", "我 想 去 home now")
    assert en_error_kind(o, 2) == "translation_like"
    o = one("我 想 go home now", "我 想 went home now")
    assert en_error_kind(o, 2) == "lexical_substitution"
    o = one("我 用 a i 写", "我 用 ai 写")
    kinds = {en_error_kind(o, i) for i in (2, 3) if o["by"][i][0] != "match"}
    assert kinds == {"segmentation_artifact"}
    o = one("这 is dangerous from 那", "这 is dangerousfrom 那")
    assert {en_error_kind(o, i) for i in (2, 3)} == {"segmentation_artifact"}
    o = one("我 想 go home 了", "我 想 home 了")
    assert en_error_kind(o, 2) in ("omission", "boundary")


def test_token_unit_map_han_partial_bytes_and_words():
    han = "中".encode()
    pieces = {1: han[:2], 2: han[2:], 3: b"A", 4: b" I", 5: "我".encode()}
    tb = lambda i: pieces[i]  # noqa: E731
    m = token_unit_map("中A I我", [1, 2, 3, 4, 5], tb)
    assert m == [[0], [0], [1], [2], [3]]
    assert token_unit_map("中A I我", [1, 2, 3, 4], tb) is None  # byte coverage mismatch


def test_token_predicates():
    assert token_is_lexical(b" go") and token_is_lexical("我".encode()) and token_is_lexical(b"\xe4")
    assert not token_is_lexical(b",") and not token_is_lexical(b" ") and not token_is_lexical(b"")
    assert tokens_format_equivalent(b" Go", b"go") and tokens_format_equivalent(b" because", b" be")
    assert not tokens_format_equivalent(b" go", b" went")


def test_round_robin_and_split_are_deterministic_and_dialogue_disjoint():
    rows = [{"utterance_id": f"u{d}_{i}", "dialogue_id": f"D{d}", "stratum": "CS" if i % 3 else "ZH_ONLY"}
            for d in range(8) for i in range(12)]
    a, short = round_robin(rows, {"CS": 40, "ZH_ONLY": 10}, 42)
    b, _ = round_robin(rows, {"CS": 40, "ZH_ONLY": 10}, 42)
    assert [r["utterance_id"] for r in a] == [r["utterance_id"] for r in b] and short == {"CS": 0, "ZH_ONLY": 0}
    assert len({r["dialogue_id"] for r in a}) == 8
    aud = split_dialogues(a, 0.2)
    n_aud = sum(r["dialogue_id"] in aud for r in a)
    assert n_aud >= 0.2 * len(a) and aud == split_dialogues(a, 0.2)
    con = {r["dialogue_id"] for r in a if r["dialogue_id"] not in aud}
    assert not (con & aud)


def test_write_json_is_write_once(tmp_path):
    p = tmp_path / "x.json"
    write_json(p, {"a": 1})
    with pytest.raises(FileExistsError):
        write_json(p, {"a": 2})
