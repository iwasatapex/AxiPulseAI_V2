import pytest

from core.analytics.drift import detect_calibration_drift


def test_calibration_drift_is_detected():
    result = detect_calibration_drift(
        reference_probabilities=[0.9, 0.1, 0.8, 0.2],
        reference_outcomes=[1.0, 0.0, 1.0, 0.0],
        current_probabilities=[0.6, 0.4, 0.7, 0.3],
        current_outcomes=[1.0, 0.0, 1.0, 0.0],
        brier_threshold=0.01,
        probability_mae_threshold=0.01,
    )

    assert result.drift_detected is True
    assert result.brier_score_shift > 0.01
    assert result.probability_mae_shift > 0.01


def test_calibration_stable_when_metrics_are_equal():
    result = detect_calibration_drift(
        reference_probabilities=[0.8, 0.2],
        reference_outcomes=[1.0, 0.0],
        current_probabilities=[0.8, 0.2],
        current_outcomes=[1.0, 0.0],
        brier_threshold=0.01,
        probability_mae_threshold=0.01,
    )

    assert result.drift_detected is False
    assert result.brier_score_shift == pytest.approx(0.0)
    assert result.probability_mae_shift == pytest.approx(0.0)


def test_small_calibration_change_below_threshold():
    result = detect_calibration_drift(
        reference_probabilities=[0.8, 0.2],
        reference_outcomes=[1.0, 0.0],
        current_probabilities=[0.79, 0.21],
        current_outcomes=[1.0, 0.0],
        brier_threshold=0.1,
        probability_mae_threshold=0.1,
    )

    assert result.drift_detected is False


def test_negative_thresholds_are_rejected():
    with pytest.raises(ValueError):
        detect_calibration_drift(
            [0.8],
            [1.0],
            [0.7],
            [1.0],
            brier_threshold=-0.1,
        )

    with pytest.raises(ValueError):
        detect_calibration_drift(
            [0.8],
            [1.0],
            [0.7],
            [1.0],
            probability_mae_threshold=-0.1,
        )


def test_probability_bounds_are_rejected():
    with pytest.raises(ValueError):
        detect_calibration_drift(
            [1.2],
            [1.0],
            [0.8],
            [1.0],
        )


def test_outcomes_must_be_binary():
    with pytest.raises(ValueError):
        detect_calibration_drift(
            [0.8],
            [0.5],
            [0.7],
            [1.0],
        )


def test_lengths_must_match():
    with pytest.raises(ValueError):
        detect_calibration_drift(
            [0.8, 0.7],
            [1.0],
            [0.7, 0.6],
            [1.0, 0.0],
        )


def test_calibration_detection_is_observational():
    reference_probabilities = [0.8, 0.2]
    reference_outcomes = [1.0, 0.0]
    current_probabilities = [0.6, 0.4]
    current_outcomes = [1.0, 0.0]

    before = (
        list(reference_probabilities),
        list(reference_outcomes),
        list(current_probabilities),
        list(current_outcomes),
    )

    detect_calibration_drift(
        reference_probabilities,
        reference_outcomes,
        current_probabilities,
        current_outcomes,
    )

    after = (
        reference_probabilities,
        reference_outcomes,
        current_probabilities,
        current_outcomes,
    )

    assert after == before
