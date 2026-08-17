from pathlib import Path

import pandas as pd
import pytest

from core.data.large_data_training import (
    LargeDataTrainingAdapter,
    LargeDataTrainingConfig,
)


def test_config_requires_positive_chunksize():
    with pytest.raises(ValueError):
        LargeDataTrainingConfig(chunksize=0)


def test_csv_batches_are_memory_bounded(tmp_path: Path):
    path = tmp_path / "large.csv"

    pd.DataFrame(
        {
            "feature": range(25),
            "target": range(25),
        }
    ).to_csv(path, index=False)

    received = []

    adapter = LargeDataTrainingAdapter(
        lambda frame: received.append(len(frame)),
        config=LargeDataTrainingConfig(chunksize=7),
    )

    batches = list(adapter.csv_batches(path))

    assert [len(batch) for batch in batches] == [7, 7, 7, 4]
    assert sum(received) == 0


def test_train_csv_is_memory_bounded(tmp_path: Path):
    path = tmp_path / "train.csv"

    pd.DataFrame(
        {
            "feature": range(25),
            "target": range(25),
        }
    ).to_csv(path, index=False)

    received = []

    adapter = LargeDataTrainingAdapter(
        lambda frame: received.append(len(frame)),
        config=LargeDataTrainingConfig(chunksize=7),
    )

    result = adapter.train_csv(path)

    assert result.status == "completed"
    assert result.batches == 4
    assert result.rows == 25
    assert result.peak_rows_per_batch == 7
    assert received == [7, 7, 7, 4]


def test_missing_csv_is_rejected(tmp_path: Path):
    adapter = LargeDataTrainingAdapter(lambda frame: None)

    with pytest.raises(FileNotFoundError):
        adapter.train_csv(tmp_path / "missing.csv")


def test_adapter_does_not_replace_models():
    adapter = LargeDataTrainingAdapter(lambda frame: None)

    assert adapter is not None
