from csasr.experiments import v2r3_hypothesis_substitution as H


def _target(result, index=1):
    return {x["reference_unit_index"]: x for x in result["targets"]}[index]


def test_partial_deletion_inserts_at_cursor_and_keeps_raw_text():
    result = H.construct_partial_hypothesis("我 like 茶", "我，茶", [1])
    assert result["validation_ok"]
    assert result["actual_units"] == ["我", "like", "茶"]
    assert "我，" in result["text"] and result["text"].endswith("茶")
    assert _target(result)["well_defined"]


def test_wrong_language_substitution_is_not_replaced():
    result = H.construct_partial_hypothesis("我 like 茶", "我喜茶", [1])
    assert result["actual_units"] == ["我", "like", "喜", "茶"]
    assert "喜" in result["text"]


def test_same_language_substitution_is_not_replaced():
    result = H.construct_partial_hypothesis("我 like 茶", "我 love 茶", [1])
    assert result["actual_units"] == ["我", "like", "love", "茶"]


def test_multiple_deletions_at_one_boundary_keep_reference_order():
    result = H.construct_partial_hypothesis(
        "我 social skills 很好", "我很好", [1, 2])
    assert result["actual_units"] == [
        "我", "social", "skills", "很", "好"]
    assert result["well_defined"] == 2
    assert result["skipped"] == 0


def test_non_english_requested_unit_is_visible_skip():
    result = H.construct_partial_hypothesis("我 like 茶", "我茶", [0])
    assert not _target(result, 0)["well_defined"]
    assert _target(result, 0)["reason"] == "reference_unit_not_english"


def test_transport_decomposition_is_ordered():
    assert H.transport_state(
        oracle_exists=False, hypothesis_has_english=False,
        has_candidate_region=False, abstained=True) == "no_step_at_all"
    assert H.transport_state(
        oracle_exists=True, hypothesis_has_english=False,
        has_candidate_region=False,
        abstained=True) == "step_exists_no_english_in_hypothesis"
    assert H.transport_state(
        oracle_exists=True, hypothesis_has_english=True,
        has_candidate_region=False,
        abstained=True) == "english_present_below_floor"
    assert H.transport_state(
        oracle_exists=True, hypothesis_has_english=True,
        has_candidate_region=True, abstained=False) == "localized"


def test_oracle_arms_are_not_methods_or_selections():
    assert H.FROZEN_CONFIG["action"]["site"] == "D"
    assert H.FROZEN_CONFIG["action"]["decoder_layer"] == 16
    assert H.FROZEN_CONFIG["action"]["rho"] == 1.0
    assert H.FROZEN_CONFIG["action"]["re_derived_here"] is False
    assert H.FROZEN_CONFIG["oracle_conditions_not_methods"] == [
        "B_gold_substituted", "C_partial_substitution"]
    assert H.FROZEN_CONFIG["selects_anything"] is False
    assert H.FROZEN_CONFIG["evaluates_gate"] is False
    assert H.FROZEN_CONFIG["held_out_roles_read"] == []
