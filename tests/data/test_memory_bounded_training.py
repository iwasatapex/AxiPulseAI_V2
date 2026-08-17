import pandas as pd
import pytest

from core.data.training import (
    MemoryBoundedTrainer,
)


def test_batches_are_forwarded_without_accumulation():
    batches = [
        pd.DataFrame({"x": [1, 2]}),
        pd.DataFrame({"x": [3, 4]}),
        pd.DataFrame({"x": [5]}),
    ]

    seen = []

    def fit_batch(frame):
        seen.append(frame["x"].tolist())

    trainer = MemoryBoundedTrainer(fit_batch)

    stats = trainer.fit_batches(batches)

    assert seen == [
        [1, 2],
        [3, 4],
        [5],
    ]

    assert stats.batches == 3
    assert stats.rows == 5
    assert stats.peak_rows_per_batch == 2


def test_empty_stream_is_valid():
    trainer = MemoryBoundedTrainer(lambda frame: None)

    stats = trainer.fit_batches([])

    assert stats.batches == 0
    assert stats.rows == 0
    assert stats.peak_rows_per_batch == 0


def test_invalid_callback_is_rejected():
    with pytest.raises(TypeError):
        MemoryBoundedTrainer(None)


def test_invalid_batch_is_rejected():
    trainer = MemoryBoundedTrainer(lambda frame: None)

    with pytest.raises(TypeError):
        trainer.fit_batches([{"x": 1}])


def test_training_does_not_mutate_input():
    frame = pd.DataFrame({
        "x": [1, 2, 3],
    })

    before = frame.copy(deep=True)

    def fit_batch(batch):
        batch["temporary"] = batch["x"] * 2

    trainer = MemoryBoundedTrainer(fit_batch)

    trainer.fit_batches([frame])

    pd.testing.assert_frame_equal(frame, before)


def test_callback_receives_one_batch_at_a_time():
    batches = [
        pd.DataFrame({"x": range(10)}),
        pd.DataFrame({"x": range(10, 15)}),
    ]

    sizes = []

    def fit_batch(frame):
        sizes.append(len(frame))

    trainer = MemoryBoundedTrainer(fit_batch)

    stats = trainer.fit_batches(batches)

    assert sizes == [10, 5]
    assert stats.peak_rows_per_batch == 10
