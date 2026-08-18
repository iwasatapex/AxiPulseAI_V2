import pandas as pd
import pytest

from core.data.features.batch import process_feature_batches


def test_feature_batches_are_processed_individually():
    batches = [
        pd.DataFrame({"value": [1, 2]}),
        pd.DataFrame({"value": [3, 4]}),
        pd.DataFrame({"value": [5]}),
    ]

    seen = []

    def transform(frame):
        seen.append(len(frame))

        result = frame.copy()
        result["double"] = result["value"] * 2
        return result

    stats = process_feature_batches(
        batches,
        transform,
    )

    assert seen == [2, 2, 1]
    assert stats.batches == 3
    assert stats.rows == 5
    assert stats.columns == 2
    assert stats.peak_rows_per_batch == 2


def test_feature_transform_must_return_dataframe():
    batches = [
        pd.DataFrame({"value": [1, 2]}),
    ]

    with pytest.raises(TypeError):
        process_feature_batches(
            batches,
            lambda frame: frame["value"],
        )


def test_feature_transform_must_preserve_rows():
    batches = [
        pd.DataFrame({"value": [1, 2]}),
    ]

    def bad_transform(frame):
        return frame.iloc[:1].copy()

    with pytest.raises(ValueError):
        process_feature_batches(
            batches,
            bad_transform,
        )


def test_empty_feature_stream_is_valid():
    stats = process_feature_batches(
        [],
        lambda frame: frame.copy(),
    )

    assert stats.batches == 0
    assert stats.rows == 0
    assert stats.columns == 0
    assert stats.peak_rows_per_batch == 0


def test_invalid_transformer_is_rejected():
    with pytest.raises(TypeError):
        process_feature_batches([], None)


def test_input_batch_is_not_mutated_by_reference():
    frame = pd.DataFrame({"value": [1, 2, 3]})
    before = frame.copy(deep=True)

    def transform(batch):
        result = batch.copy()
        result["derived"] = result["value"] + 1
        return result

    process_feature_batches([frame], transform)

    pd.testing.assert_frame_equal(frame, before)
