"""CPU tests for the P2-1D audit helpers (synthetic strings only)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.p2_1d_audit import (alt_alignment, contingency, dist_bin, edit_cost, en_distances, ok_from_ops,  # noqa: E402
                              switch_distances, topk_share, transition, utterance_events, wilson)
from reft.text import alignment, one  # noqa: E402

FILL = {"嗯", "yeah", "um"}


def test_distance_bins():
    assert [dist_bin(d) for d in (0, 1, 2, 3, 4, 7, 8, 50)] == ["0-1", "0-1", "2-3", "2-3", "4-7", "4-7", ">=8", ">=8"]
    assert dist_bin(None) is None


def test_en_and_switch_distances():
    units = ["我", "想", "go", "home", "了", "吧"]
    tags = ["ZH", "ZH", "EN", "EN", "ZH", "ZH"]
    assert en_distances(tags, units, FILL) == [2, 1, 0, 0, 1, 2]
    # switches between 想|go (1,2) and home|了 (3,4)
    assert switch_distances(tags, units, FILL) == [1, 0, 0, 0, 0, 1]
    assert en_distances(["ZH", "ZH"], ["我", "想"], FILL) == [None, None]
    assert switch_distances(["ZH", "ZH"], ["我", "想"], FILL) == [None, None]
    # fillers never create an English anchor or a switch
    assert en_distances(["ZH", "EN", "ZH"], ["我", "yeah", "想"], FILL) == [None, None, None]
    assert switch_distances(["ZH", "EN", "ZH"], ["我", "yeah", "想"], FILL) == [None, None, None]


def test_transition_classes():
    assert transition(False, True) == "repair" and transition(True, False) == "corruption"
    assert transition(True, True) == "unchanged_correct" and transition(False, False) == "unchanged_wrong"


@pytest.mark.parametrize("a,b", [(list("abcabc"), list("abc")), (list("aab"), list("ab")), (list("xyz"), list("xqyz")),
                                 (list("我想去"), list("我去去")), ([], list("ab")), (list("ab"), [])])
def test_alt_alignment_is_valid_min_cost(a, b):
    alt, can = alt_alignment(a, b), alignment(a, b)
    assert edit_cost(alt) == edit_cost(can)
    assert [i for _, i, _ in alt if i is not None] == list(range(len(a)))   # every reference unit exactly once
    assert [j for _, _, j in alt if j is not None] == list(range(len(b)))   # every hypothesis unit exactly once
    assert all((o == "match") == (a[i] == b[j]) for o, i, j in alt if o in ("match", "sub"))


def test_alt_alignment_can_differ_on_ties():
    a, b = list("aab"), list("ab")
    assert ok_from_ops(alignment(a, b), 3) != ok_from_ops(alt_alignment(a, b), 3) or alignment(a, b) == alt_alignment(a, b)


def test_concentration_and_wilson():
    assert topk_share([5, 3, 1, 1], 1) == 0.5 and topk_share([5, 3, 1, 1], 3) == 0.9 and topk_share([], 1) is None
    lo, hi = wilson(3, 100)
    assert 0 < lo < 0.03 < hi < 0.1 and wilson(0, 0) is None


def test_contingency_cells():
    u = [{"dialogue": "d1", "en_repair": True, "zh_damage": True, "en_corr": 2, "en_corrupt": 0, "zh_new": 1, "zh_rep": 0, "d_mixed": -1},
         {"dialogue": "d2", "en_repair": False, "zh_damage": True, "en_corr": 0, "en_corrupt": 1, "zh_new": 3, "zh_rep": 1, "d_mixed": 3},
         {"dialogue": "d2", "en_repair": False, "zh_damage": False, "en_corr": 0, "en_corrupt": 0, "zh_new": 0, "zh_rep": 0, "d_mixed": 0}]
    c = contingency(u)
    assert c["EN_REPAIR&ZH_DAMAGE"]["utterances"] == 1 and c["no_EN_REPAIR&ZH_DAMAGE"]["zh_new"] == 3
    assert c["no_EN_REPAIR&no_ZH_DAMAGE"]["dialogues"] == 1 and c["EN_REPAIR&no_ZH_DAMAGE"]["utterances"] == 0
    assert sum(x["utterances"] for x in c.values()) == 3


def test_utterance_events_reconcile_with_scorer():
    ref = "我 想 go home 了 吧"
    b = {"o": one(ref, "我 想 去 home 了 吧")}
    s = {"o": one(ref, "我 go home 了 吧 吧")}
    ev, ins = utterance_events(b, s, FILL, 2, "beam5", "u", "d", "CS")
    assert [e["k"] for e in ev] == list(range(6))
    tr = {e["unit"]: e["transition"] for e in ev}
    assert tr["go"] == "repair" and tr["想"] == "corruption" and tr["home"] == "unchanged_correct"
    assert ev[2]["en_cat_b0"] == "translation_like" and ev[2]["en_cat_sys"] == "correct"
    assert sum(ins["sys"].values()) == 1 and sum(ins["b0"].values()) == 0  # extra 吧 is an insertion, not a ref unit
    # mixed-error change = change in wrong reference units + change in insertions
    d_ref = sum(not e["sys_ok"] for e in ev) - sum(not e["b0_ok"] for e in ev)
    assert s["o"]["counts"][2] - b["o"]["counts"][2] == d_ref + sum(ins["sys"].values()) - sum(ins["b0"].values())
