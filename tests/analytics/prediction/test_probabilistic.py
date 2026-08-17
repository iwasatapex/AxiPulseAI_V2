import pytest

from core.analytics.prediction import (
    calculate_probabilistic_analytics,
)


def test_probability_metrics():
    result = calculate_probabilistic_analytics(
        [0.9, 0.2, 0.7, 0.1],
        [1.0, 0.0, 1.0, 0.0],
    )

    assert result.count == 4
    assert result.brier_score == pytest.approx(
        (0.01 + 0.04 + 0.09 + 0.01) / 4
    )
    assert result.probability_mae == pytest.approx(
        (0.1 + 0.2 + 0.3 + 0.1) / 4
    )


def test_interval_metrics():
    result = calculate_probabilistic_analytics(
        [0.8, 0.7, 0.2],
        [1.0, 0.0, 1.0],
        lower_bounds=[0.5, -0.2, 0.5],
        upper_bounds=[1.2, 0.2, 1.5],
    )

    assert result.interval_coverage == pytest.approx(1.0)
    assert result.mean_interval_width == pytest.approx(
        (0.7 + 0.4 + 1.0) / 3
    )


def test_no_intervals_are_supported():
    result = calculate_probabilistic_analytics(
        [0.5, 0.5],
        [1.0, 0.0],
    )

    assert result.interval_coverage == 0.0
    assert result.mean_interval_width == 0.0


def test_probability_bounds_are_enforced():
    with pytest.raises(ValueError):
        calculate_probabilistic_analytics(
            [1.2],
            [1.0],
        )


def test_binary_outcomes_are_required():
    with pytest.raises(ValueError):
        calculate_probabilistic_analytics(
            [0.5],
            [0.5],
        )


def test_interval_bounds_must_be_supplied_together():
    with pytest.raises(ValueError):
        calculate_probabilistic_analytics(
            [0.5],
            [1.0],
            lower_bounds=[0.0],
        )


def test_interval_lengths_must_match():
    with pytest.raises(ValueError):
        calculate_probabilistic_analytics(
            [0.5, 0.5],
            [1.0, 0.0],
            lower_bounds=[0.0],
            upper_bounds=[1.0],
        )


def test_invalid_interval_order_is_rejected():
    with pytest.raises(ValueError):
        calculate_probabilistic_analytics(
            [0.5],
            [1.0],
            lower_bounds=[2.0],
            upper_bounds=[1.0],
        )


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        calculate_probabilistic_analytics([], [])
