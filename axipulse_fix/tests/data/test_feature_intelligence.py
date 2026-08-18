import pandas as pd

from core.data import (
    analyze_features,
    assess_features,
    detect_leakage,
    identify_target,
)


def test_dynamic_feature_assessment():
    frame = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "segment": ["A", "B", "A", "B"],
            "flag": [True, False, True, False],
            "record_id": ["a", "b", "c", "d"],
        }
    )

    result = assess_features(frame)

    by_name = {item.name: item for item in result}

    assert by_name["age"].eligible is True
    assert by_name["segment"].eligible is True
    assert by_name["flag"].eligible is True
    assert by_name["record_id"].eligible is False
    assert by_name["record_id"].is_identifier is True


def test_missing_data_threshold():
    frame = pd.DataFrame(
        {
            "good": [1.0, 2.0, 3.0, 4.0],
            "bad": [None, None, None, 4.0],
        }
    )

    result = assess_features(
        frame,
        max_missing_fraction=0.50,
    )

    by_name = {item.name: item for item in result}

    assert by_name["good"].eligible is True
    assert by_name["bad"].eligible is False


def test_explicit_target_is_respected():
    frame = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "outcome_value": [0, 1, 0],
        }
    )

    assert identify_target(
        frame,
        target="outcome_value",
    ) == "outcome_value"


def test_target_is_not_invented_when_ambiguous():
    frame = pd.DataFrame(
        {
            "value_a": [1, 2, 3],
            "value_b": [0, 1, 0],
        }
    )

    assert identify_target(frame) is None


def test_exact_target_copy_is_detected():
    frame = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
            "target_copy": [0, 1, 0],
        }
    )

    findings = detect_leakage(
        frame,
        target="target",
    )

    assert any(
        item.feature == "target_copy"
        and item.leakage_type == "target_copy"
        for item in findings
    )


def test_full_feature_analysis():
    frame = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
            "record_id": ["a", "b", "c"],
        }
    )

    result = analyze_features(
        frame,
        target="target",
    )

    assert result.target == "target"
    assert result.duplicate_rows == 0
    assert result.missing_cells == 0
    assert len(result.features) == 3


def test_free_text_is_distinguished_from_identifier():
    from core.data import profile_features

    frame = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "description": [
                "This is a long descriptive customer record.",
                "Another long descriptive customer record.",
                "A third long descriptive customer record.",
                "A fourth long descriptive customer record.",
            ],
        }
    )

    profiles = profile_features(frame)
    by_name = {item.name: item for item in profiles}

    assert by_name["id"].likely_identifier is True
    assert by_name["description"].likely_free_text is True
    assert by_name["description"].likely_identifier is False


def test_feature_profile_preserves_canonical_schema_kind():
    from core.data import profile_features

    frame = pd.DataFrame(
        {
            "score": [1.0, 2.0, 3.0],
            "category": ["A", "B", "A"],
            "flag": [True, False, True],
        }
    )

    profiles = profile_features(frame)
    by_name = {item.name: item for item in profiles}

    assert by_name["score"].kind == "numeric"
    assert by_name["category"].kind == "categorical"
    assert by_name["flag"].kind == "boolean"


def test_profile_features_does_not_mutate_dataframe():
    from core.data import profile_features

    frame = pd.DataFrame(
        {
            "value": [1, 2, 3],
            "text": ["a", "b", "c"],
        }
    )

    before = frame.copy(deep=True)

    profile_features(frame)

    pd.testing.assert_frame_equal(frame, before)
