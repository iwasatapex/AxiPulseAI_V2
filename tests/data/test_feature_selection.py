import pandas as pd

from core.data import select_features


def test_selects_normal_features():
    frame = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "segment": ["A", "B", "A", "B"],
            "flag": [True, False, True, False],
        }
    )

    result = select_features(frame)

    assert result.selected == (
        "age",
        "segment",
        "flag",
    )
    assert result.excluded == ()


def test_excludes_identifier():
    frame = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "record_id": ["a", "b", "c"],
        }
    )

    result = select_features(frame)

    assert "age" in result.selected
    assert "record_id" in result.excluded
    assert result.reasons["record_id"] == (
        "identifier excluded from features"
    )


def test_excludes_free_text():
    frame = pd.DataFrame(
        {
            "score": [1, 2, 3],
            "description": [
                "This is a long descriptive record containing text.",
                "Another long descriptive record containing information.",
                "A third long descriptive record containing natural language.",
            ],
        }
    )

    result = select_features(frame)

    assert "score" in result.selected
    assert "description" in result.excluded
    assert result.reasons["description"] == (
        "free-text excluded from features"
    )


def test_excludes_target():
    frame = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )

    result = select_features(
        frame,
        target="target",
    )

    assert result.selected == ("feature",)
    assert result.excluded == ("target",)
    assert result.reasons["target"] == "target column"


def test_excludes_high_missing_feature():
    frame = pd.DataFrame(
        {
            "good": [1, 2, 3, 4],
            "bad": [None, None, None, 4],
        }
    )

    result = select_features(
        frame,
        max_missing_fraction=0.50,
    )

    assert "good" in result.selected
    assert "bad" in result.excluded
    assert result.reasons["bad"] == (
        "exceeds missing-data threshold"
    )


def test_selection_does_not_mutate_dataframe():
    frame = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "record_id": ["a", "b", "c"],
        }
    )

    before = frame.copy(deep=True)

    select_features(frame)

    pd.testing.assert_frame_equal(
        frame,
        before,
    )


def test_selection_metadata():
    frame = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "record_id": ["a", "b", "c"],
        }
    )

    result = select_features(frame)

    assert result.metadata["rows"] == 3
    assert result.metadata["columns"] == 2
    assert result.metadata["selected_count"] == 1
    assert result.metadata["excluded_count"] == 1
