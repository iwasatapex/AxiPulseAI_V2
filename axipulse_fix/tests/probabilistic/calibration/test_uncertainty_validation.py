import pytest

from core.probabilistic.calibration import UncertaintyValidator


def test_interval_validation():
    result = UncertaintyValidator.validate_intervals(
        [0.0, 0.0, 2.0],
        [1.0, 1.0, 3.0],
        [0.5, 2.0, 2.5],
        nominal_level=0.95,
    )

    assert result.covered == 2
    assert result.total == 3
    assert result.coverage == pytest.approx(2 / 3)
    assert result.mean_interval_width == pytest.approx(1.0)
    assert result.median_interval_width == pytest.approx(1.0)
    assert result.coverage_error == pytest.approx(
        abs((2 / 3) - 0.95)
    )


def test_interval_widths_are_non_negative():
    result = UncertaintyValidator.validate_intervals(
        [1.0, 2.0],
        [2.0, 5.0],
        [1.5, 3.0],
    )

    assert result.min_interval_width == pytest.approx(1.0)
    assert result.max_interval_width == pytest.approx(3.0)


def test_invalid_ordering():
    with pytest.raises(ValueError):
        UncertaintyValidator.validate_intervals(
            [2.0],
            [1.0],
            [1.5],
        )


def test_invalid_nominal_level():
    with pytest.raises(ValueError):
        UncertaintyValidator.validate_intervals(
            [0.0],
            [1.0],
            [0.5],
            nominal_level=1.0,
        )


def test_length_mismatch():
    with pytest.raises(ValueError):
        UncertaintyValidator.validate_intervals(
            [0.0],
            [1.0, 2.0],
            [0.5],
        )


def test_empty_inputs():
    with pytest.raises(ValueError):
        UncertaintyValidator.validate_intervals(
            [],
            [],
            [],
        )


def test_non_finite_inputs():
    with pytest.raises(ValueError):
        UncertaintyValidator.validate_intervals(
            [0.0],
            [float("nan")],
            [0.5],
        )
