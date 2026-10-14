"""Unit tests for the P2-1 terminal decision rules (docs/reft/P2_1_DECISION_RULES.md)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from reft.report_p2_1 import correction_but_damage, decide, preservation_ok, uncertain_gain, useful  # noqa: E402


def arm(**kw):
    a = {"d_mer": -0.01, "d_mer_ci": [-0.02, 0.0], "d_cs_mer": -0.02, "d_cs_mer_ci": [-0.03, -0.01], "net": 20, "corrections": 30,
         "corruptions": 10, "dialogues_with_corrections": 8, "max_share": 0.2, "mandarin_retention": 0.995, "zh_cer": 0.096,
         "zh_cer_b0": 0.0962, "loops": 0, "loops_b0": 0}
    a.update(kw)
    return a


NONE = arm(d_mer=0.0, d_cs_mer=0.0, d_cs_mer_ci=[-0.01, 0.01], net=0, corrections=2, dialogues_with_corrections=1, max_share=None)


def test_useful_requires_every_condition():
    assert useful(arm())
    for bad in (dict(d_cs_mer=-0.004), dict(d_cs_mer_ci=[-0.03, 0.001]), dict(d_mer=0.001), dict(net=0),
                dict(dialogues_with_corrections=3), dict(max_share=0.41), dict(mandarin_retention=0.989),
                dict(zh_cer=0.0983), dict(loops=2)):
        assert not useful(arm(**bad)), bad


def test_preservation_and_damage():
    assert preservation_ok(arm()) and not preservation_ok(arm(mandarin_retention=0.98))
    assert correction_but_damage(arm(mandarin_retention=0.98)) and correction_but_damage(arm(d_mer=0.01))
    assert not correction_but_damage(NONE)
    assert uncertain_gain(arm(d_cs_mer_ci=[-0.03, 0.002])) and not uncertain_gain(arm())


def test_verdict_order():
    assert decide(False, {"R2": arm(), "L1": arm()}) == "P2_1_ENGINEERING_BLOCKED"
    assert decide(True, {"R2": arm(), "L1": NONE}) == "P2_1_REPRESENTATION_FEASIBLE"
    assert decide(True, {"R2": arm(), "L1": arm()}) == "P2_1_REPRESENTATION_FEASIBLE"
    assert decide(True, {"R2": NONE, "L1": arm()}) == "P2_1_WEIGHT_PEFT_ONLY"
    assert decide(True, {"R2": arm(mandarin_retention=0.97), "L1": NONE}) == "P2_1_CORRECTION_PRESERVATION_FAILURE"
    assert decide(True, {"R2": arm(d_cs_mer_ci=[-0.03, 0.002]), "L1": NONE}) == "P2_1_INCONCLUSIVE"
    assert decide(True, {"R2": NONE, "L1": NONE}) == "P2_1_NO_FREE_DECODING_GAIN"
