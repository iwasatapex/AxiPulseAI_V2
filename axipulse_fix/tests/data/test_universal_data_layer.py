import pandas as pd

from core.data import (
    UniversalDataset,
    discover_schema,
    duplicate_rows,
    missingness,
    validate_dataset,
)


def test_universal_dataset_contract():
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

    assert dataset.rows == 3
    assert dataset.columns == ["score", "category"]
    assert dataset.shape == (3, 2)


def test_schema_discovery():
    frame = pd.DataFrame(
        {
            "score": [1.0, 2.0, 3.0],
            "category": ["a", "b", "a"],
            "flag": [True, False, True],
        }
    )

    schema = discover_schema(frame)

    kinds = {item.name: item.kind for item in schema}

    assert kinds["score"] == "numeric"
    assert kinds["category"] == "categorical"
    assert kinds["flag"] == "boolean"


def test_validation():
    frame = pd.DataFrame(
        {
            "score": [1, 2, 3],
        }
    )

    validate_dataset(frame)

    assert missingness(frame)["score"] == 0.0
    assert duplicate_rows(frame) == 0
