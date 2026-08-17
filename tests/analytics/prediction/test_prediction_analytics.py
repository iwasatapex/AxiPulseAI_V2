import pytest

from core.analytics.prediction import (
    PredictionRecord,
    calculate_prediction_metrics,
)


def test_unresolved_prediction_has_no_error():
    record = PredictionRecord(
        prediction_id="p1",
        predicted=82.0,
    )

    assert record.resolved is False
    assert record.error is None
    assert record.absolute_error is None
    assert record.squared_error is None


def test_resolved_prediction_metrics():
    record = PredictionRecord(
        prediction_id="p1",
        predicted=82.0,
        actual=80.0,
    )

    assert record.resolved is True
    assert record.error == 2.0
    assert record.absolute_error == 2.0
    assert record.squared_error == 4.0


def test_prediction_record_preserves_lineage():
    record = PredictionRecord(
        prediction_id="p1",
        predicted=82.0,
        actual=80.0,
        model_version="model-v1",
        dataset_version="dataset-v1",
        feature_version="features-v1",
    )

    assert record.model_version == "model-v1"
    assert record.dataset_version == "dataset-v1"
    assert record.feature_version == "features-v1"


def test_metrics():
    result = calculate_prediction_metrics(
        [80.0, 90.0, 100.0],
        [82.0, 88.0, 101.0],
    )

    assert result.count == 3
    assert result.mae == pytest.approx(5.0 / 3.0)
    assert result.rmse == pytest.approx(
        ((4.0 + 4.0 + 1.0) / 3.0) ** 0.5
    )
    assert result.bias == pytest.approx(-1.0 / 3.0)


def test_metrics_reject_length_mismatch():
    with pytest.raises(ValueError):
        calculate_prediction_metrics(
            [1.0, 2.0],
            [1.0],
        )


def test_metrics_reject_empty_input():
    with pytest.raises(ValueError):
        calculate_prediction_metrics([], [])
