import pytest

from core.analytics.drift import (
    DriftDetector,
    calculate_numeric_drift,
)


def test_mean_shift_detected():
    metrics = calculate_numeric_drift(
        [1.0, 2.0, 3.0],
        [3.0, 4.0, 5.0],
    )

    result = DriftDetector.detect(
        metrics,
        mean_threshold=0.5,
    )

    assert result.drift_detected is True
    assert result.mean_shift_detected is True
    assert result.std_shift_detected is False


def test_mean_shift_below_threshold_is_not_detected():
    metrics = calculate_numeric_drift(
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
    )

    result = DriftDetector.detect(
        metrics,
        mean_threshold=0.5,
    )

    assert result.drift_detected is False
    assert result.mean_shift_detected is False


def test_std_shift_detected():
    metrics = calculate_numeric_drift(
        [1.0, 2.0, 3.0],
        [1.0, 10.0, 19.0],
    )

    result = DriftDetector.detect(
        metrics,
        std_threshold=1.0,
    )

    assert result.drift_detected is True
    assert result.std_shift_detected is True


def test_no_drift():
    metrics = calculate_numeric_drift(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    )

    result = DriftDetector.detect(
        metrics,
        mean_threshold=0.1,
        std_threshold=0.1,
    )

    assert result.drift_detected is False


def test_negative_threshold_is_rejected():
    metrics = calculate_numeric_drift(
        [1.0],
        [2.0],
    )

    with pytest.raises(ValueError):
        DriftDetector.detect(
            metrics,
            mean_threshold=-0.1,
        )

    with pytest.raises(ValueError):
        DriftDetector.detect(
            metrics,
            std_threshold=-0.1,
        )


def test_detection_does_not_mutate_metrics():
    metrics = calculate_numeric_drift(
        [1.0, 2.0],
        [2.0, 3.0],
    )

    before = metrics

    DriftDetector.detect(
        metrics,
        mean_threshold=0.1,
        std_threshold=0.1,
    )

    assert metrics == before
