import pytest

from core.probabilistic.calibration import CalibrationEngine


def test_brier_score():
    result = CalibrationEngine.binary(
        [0.0, 0.5, 1.0],
        [0.0, 1.0, 1.0],
    )

    assert result.sample_count == 3
    assert result.brier_score == pytest.approx(
        (0.0 + 0.25 + 0.0) / 3
    )


def test_reliability_bins():
    result = CalibrationEngine.reliability(
        [0.1, 0.2, 0.8, 0.9],
        [0.0, 0.0, 1.0, 1.0],
        bins=2,
    )

    assert result.sample_count == 4
    assert len(result.bins) == 2

    assert result.bins[0].predicted_probability == pytest.approx(0.15)
    assert result.bins[0].observed_frequency == pytest.approx(0.0)

    assert result.bins[1].predicted_probability == pytest.approx(0.85)
    assert result.bins[1].observed_frequency == pytest.approx(1.0)


def test_interval_coverage():
    result = CalibrationEngine.interval_coverage(
        [0.0, 0.0, 2.0],
        [1.0, 1.0, 3.0],
        [0.5, 2.0, 2.5],
        nominal_level=0.95,
    )

    assert result.covered == 2
    assert result.total == 3
    assert result.coverage == pytest.approx(2 / 3)


def test_probability_validation():
    with pytest.raises(ValueError):
        CalibrationEngine.binary(
            [-0.1, 0.5],
            [0.0, 1.0],
        )


def test_outcome_validation():
    with pytest.raises(ValueError):
        CalibrationEngine.binary(
            [0.2, 0.8],
            [0.0, 2.0],
        )


def test_interval_ordering_validation():
    with pytest.raises(ValueError):
        CalibrationEngine.interval_coverage(
            [2.0],
            [1.0],
            [1.5],
        )


def test_length_validation():
    with pytest.raises(ValueError):
        CalibrationEngine.binary(
            [0.2],
            [0.0, 1.0],
        )
