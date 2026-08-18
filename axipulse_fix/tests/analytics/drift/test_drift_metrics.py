import pytest

from core.analytics.drift import calculate_numeric_drift


def test_numeric_drift_metrics():
    result = calculate_numeric_drift(
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
    )

    assert result.count_reference == 3
    assert result.count_current == 3
    assert result.reference_mean == pytest.approx(2.0)
    assert result.current_mean == pytest.approx(3.0)
    assert result.mean_shift == pytest.approx(1.0)
    assert result.mean_absolute_shift == pytest.approx(1.0)
    assert result.std_shift == pytest.approx(0.0)


def test_drift_does_not_mutate_inputs():
    reference = [1.0, 2.0, 3.0]
    current = [4.0, 5.0, 6.0]

    reference_before = list(reference)
    current_before = list(current)

    calculate_numeric_drift(reference, current)

    assert reference == reference_before
    assert current == current_before


def test_empty_reference_is_rejected():
    with pytest.raises(ValueError):
        calculate_numeric_drift([], [1.0])


def test_empty_current_is_rejected():
    with pytest.raises(ValueError):
        calculate_numeric_drift([1.0], [])


def test_non_finite_values_are_rejected():
    with pytest.raises(ValueError):
        calculate_numeric_drift(
            [1.0, float("nan")],
            [1.0, 2.0],
        )


def test_constant_series_has_zero_standard_deviation():
    result = calculate_numeric_drift(
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 5.0],
    )

    assert result.reference_std == pytest.approx(0.0)
    assert result.current_std == pytest.approx(0.0)
    assert result.mean_shift == pytest.approx(0.0)
    assert result.std_shift == pytest.approx(0.0)
