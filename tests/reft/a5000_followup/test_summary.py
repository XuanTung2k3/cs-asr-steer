import pytest

from reft.a5000_followup.summarize import paired_bootstrap


def rows(errors):
    return [{'dialogue':f'd{i}','stratum':'CS','counts':[0,0,e,10,0,0,0,0,0,0,0]}
            for i,e in enumerate(errors)]


def test_paired_dialogue_bootstrap_identity():
    r=paired_bootstrap(rows([1,2,3]),rows([1,2,3]),100)
    assert r['delta_mer']==r['delta_cs_mer']==0 and r['mer_ci95']==[0.,0.]


def test_paired_dialogue_bootstrap_known_uniform_improvement():
    r=paired_bootstrap(rows([2,2,2]),rows([1,1,1]),100)
    assert r['delta_mer']==pytest.approx(-.1)
    assert r['cs_mer_ci95']==pytest.approx([-.1,-.1])
