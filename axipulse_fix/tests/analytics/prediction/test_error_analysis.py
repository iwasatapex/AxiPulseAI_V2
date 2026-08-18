import pytest

from core.analytics.prediction import calculate_error_bias


def test_balanced_predictions_have_zero_mean_bias():
    result = calculate_error_bias(
        [90.0, 110.0],
        [100.0, 100.0],
    )

    assert result.count == 2
    assert result.mean_error == pytest.approx(0.0)
    assert result.mean_absolute_error == pytest.approx(10.0)
    assert result.rmse == pytest.approx(10.0)
    assert result.max_absolute_error == pytest.approx(10.0)


def test_overprediction_rate():
    result = calculate_error_bias(
        [110.0, 120.0, 100.0],
        [100.0, 100.0, 100.0],
    )

    assert result.overprediction_rate == pytest.approx(2.0 / 3.0)
    assert result.underprediction_rate == pytest.approx(0.0)
    assert result.mean_error == pytest.approx(10.0)


def test_underprediction_rate():
    result = calculate_error_bias(
        [90.0, 80.0, 100.0],
        [100.0, 100.0, 100.0],
    )

    assert result.underprediction_rate == pytest.approx(2.0 / 3.0)
    assert result.overprediction_rate == pytest.approx(0.0)
    assert result.mean_error == pytest.approx(-10.0)


def test_zero_error_is_neither_over_nor_under_prediction():
    result = calculate_error_bias(
        [100.0, 100.0],
        [100.0, 100.0],
    )

    assert result.mean_error == pytest.approx(0.0)
    assert result.mean_absolute_error == pytest.approx(0.0)
    assert result.rmse == pytest.approx(0.0)
    assert result.max_absolute_error == pytest.approx(0.0)
    assert result.underprediction_rate == pytest.approx(0.0)
    assert result.overprediction_rate == pytest.approx(0.0)


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        calculate_error_bias(
            [1.0, 2.0],
            [1.0],
        )


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        calculate_error_bias([], [])
