import pytest

from core.analytics.drift import detect_prediction_drift


def test_prediction_mean_drift_is_detected():
    result = detect_prediction_drift(
        [10.0, 20.0, 30.0],
        [30.0, 40.0, 50.0],
        mean_threshold=1.0,
        std_threshold=100.0,
        median_threshold=100.0,
    )

    assert result.drift_detected is True
    assert result.mean_shift == pytest.approx(20.0)


def test_prediction_distribution_is_stable():
    result = detect_prediction_drift(
        [10.0, 20.0, 30.0],
        [10.0, 20.0, 30.0],
        mean_threshold=0.1,
        std_threshold=0.1,
        median_threshold=0.1,
    )

    assert result.drift_detected is False
    assert result.mean_shift == pytest.approx(0.0)
    assert result.std_shift == pytest.approx(0.0)
    assert result.median_shift == pytest.approx(0.0)


def test_std_drift_is_detected():
    result = detect_prediction_drift(
        [10.0, 11.0, 12.0],
        [1.0, 11.0, 21.0],
        mean_threshold=100.0,
        std_threshold=1.0,
        median_threshold=100.0,
    )

    assert result.drift_detected is True
    assert abs(result.std_shift) > 1.0


def test_median_drift_is_detected():
    result = detect_prediction_drift(
        [1.0, 2.0, 3.0],
        [10.0, 11.0, 12.0],
        mean_threshold=100.0,
        std_threshold=100.0,
        median_threshold=1.0,
    )

    assert result.drift_detected is True
    assert result.median_shift == pytest.approx(9.0)


def test_threshold_suppresses_small_drift():
    result = detect_prediction_drift(
        [10.0, 20.0, 30.0],
        [10.1, 20.1, 30.1],
        mean_threshold=1.0,
        std_threshold=1.0,
        median_threshold=1.0,
    )

    assert result.drift_detected is False


def test_empty_predictions_are_rejected():
    with pytest.raises(ValueError):
        detect_prediction_drift([], [1.0])


def test_negative_thresholds_are_rejected():
    with pytest.raises(ValueError):
        detect_prediction_drift(
            [1.0],
            [2.0],
            mean_threshold=-0.1,
        )

    with pytest.raises(ValueError):
        detect_prediction_drift(
            [1.0],
            [2.0],
            std_threshold=-0.1,
        )

    with pytest.raises(ValueError):
        detect_prediction_drift(
            [1.0],
            [2.0],
            median_threshold=-0.1,
        )


def test_prediction_drift_does_not_mutate_inputs():
    reference = [1.0, 2.0, 3.0]
    current = [2.0, 3.0, 4.0]

    before_reference = list(reference)
    before_current = list(current)

    detect_prediction_drift(
        reference,
        current,
        mean_threshold=0.1,
        std_threshold=0.1,
        median_threshold=0.1,
    )

    assert reference == before_reference
    assert current == before_current


def test_single_value_standard_deviation_is_zero():
    result = detect_prediction_drift(
        [5.0],
        [5.0],
        mean_threshold=0.1,
        std_threshold=0.1,
        median_threshold=0.1,
    )

    assert result.reference_std == pytest.approx(0.0)
    assert result.current_std == pytest.approx(0.0)
    assert result.std_shift == pytest.approx(0.0)
