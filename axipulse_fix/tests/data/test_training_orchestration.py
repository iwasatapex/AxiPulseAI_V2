import pandas as pd
import pytest

from core.data.training_orchestration import (
    TrainingOrchestrator,
)


def test_orchestrator_runs_batches():
    batches = [
        pd.DataFrame({"x": [1, 2]}),
        pd.DataFrame({"x": [3, 4]}),
        pd.DataFrame({"x": [5]}),
    ]

    received = []

    def fit_batch(frame):
        received.append(len(frame))

    orchestrator = TrainingOrchestrator(fit_batch)

    result = orchestrator.run(batches)

    assert result.status == "completed"
    assert result.batches == 3
    assert result.rows == 5
    assert result.peak_rows_per_batch == 2
    assert received == [2, 2, 1]


def test_orchestrator_preserves_memory_bounded_behavior():
    batches = [
        pd.DataFrame({"x": range(10)}),
        pd.DataFrame({"x": range(10, 15)}),
    ]

    sizes = []

    def fit_batch(frame):
        sizes.append(len(frame))

    result = TrainingOrchestrator(fit_batch).run(batches)

    assert result.rows == 15
    assert result.peak_rows_per_batch == 10
    assert sizes == [10, 5]


def test_orchestrator_rejects_invalid_callback():
    with pytest.raises(TypeError):
        TrainingOrchestrator(None)


def test_orchestrator_handles_empty_stream():
    result = TrainingOrchestrator(lambda frame: None).run([])

    assert result.status == "completed"
    assert result.batches == 0
    assert result.rows == 0
    assert result.peak_rows_per_batch == 0


def test_orchestrator_does_not_accumulate_batches():
    received_ids = []

    def fit_batch(frame):
        received_ids.append(id(frame))

    batches = [
        pd.DataFrame({"x": [1]}),
        pd.DataFrame({"x": [2]}),
        pd.DataFrame({"x": [3]}),
    ]

    result = TrainingOrchestrator(fit_batch).run(batches)

    assert result.batches == 3
    assert len(received_ids) == 3
