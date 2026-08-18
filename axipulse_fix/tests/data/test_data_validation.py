import pandas as pd

from core.data import (
    UniversalDataset,
    validate_dataset,
)


def test_valid_dataset():
    frame = pd.DataFrame(
        {
            "score": [1.0, 2.0, 3.0],
            "category": ["a", "b", "a"],
        }
    )

    dataset = UniversalDataset(
        data=frame,
        source_type="test",
    )

    result = validate_dataset(dataset)

    assert result.valid is True
    assert result.rows == 3
    assert result.columns == 2
    assert result.missing_cells == 0
    assert result.duplicate_rows == 0


def test_missing_values_are_reported_without_invalidating_structure():
    frame = pd.DataFrame(
        {
            "score": [1.0, None, 3.0],
            "category": ["a", "b", None],
        }
    )

    dataset = UniversalDataset(
        data=frame,
        source_type="test",
    )

    result = validate_dataset(dataset)

    assert result.valid is True
    assert result.missing_cells == 2


def test_duplicate_rows_are_reported():
    frame = pd.DataFrame(
        {
            "score": [1.0, 1.0],
            "category": ["a", "a"],
        }
    )

    dataset = UniversalDataset(
        data=frame,
        source_type="test",
    )

    result = validate_dataset(dataset)

    assert result.valid is True
    assert result.duplicate_rows == 1


def test_empty_dataset_is_invalid_by_default():
    frame = pd.DataFrame(columns=["score"])

    dataset = UniversalDataset(
        data=frame,
        source_type="test",
    )

    result = validate_dataset(dataset)

    assert result.valid is False
    assert "dataset contains no rows" in result.issues


def test_empty_dataset_can_be_explicitly_allowed():
    frame = pd.DataFrame(columns=["score"])

    dataset = UniversalDataset(
        data=frame,
        source_type="test",
    )

    result = validate_dataset(
        dataset,
        allow_empty=True,
    )

    assert result.valid is True


def test_duplicate_column_names_are_invalid():
    frame = pd.DataFrame(
        [[1, 2]],
        columns=["score", "score"],
    )

    dataset = UniversalDataset(
        data=frame,
        source_type="test",
    )

    result = validate_dataset(dataset)

    assert result.valid is False
    assert "dataset contains duplicate column names" in result.issues


def test_validation_does_not_mutate_dataset():
    frame = pd.DataFrame(
        {
            "score": [1.0, None],
            "category": ["a", "b"],
        }
    )

    before = frame.copy(deep=True)

    dataset = UniversalDataset(
        data=frame,
        source_type="test",
    )

    validate_dataset(dataset)

    pd.testing.assert_frame_equal(
        dataset.data,
        before,
    )
