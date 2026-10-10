"""Focused test for DG-03 C4 count-matched matrix-continuation steps."""
from experiments.dg03_causal_screen import _matched_matrix_sets


def test_matched_matrix_sets_count_matched_and_disjoint():
    oracle = {"u1": [2, 5, 9], "u2": [2, 3], "u3": []}
    m = _matched_matrix_sets(oracle)
    for uid, steps in oracle.items():
        oset = set(steps)
        # count-matched to |oracle| per utterance
        assert len(m[uid]) == len(oset)
        # disjoint from oracle steps (genuine non-target locations)
        assert m[uid].isdisjoint(oset)
    # consecutive oracle still yields a matched, disjoint set
    assert m["u2"].isdisjoint({2, 3}) and len(m["u2"]) == 2
