from core.decision_intelligence.v3.integration import (
    UniversalProbabilisticAdapter,
)


def test_predictors_use_different_historical_distributions():
    adapter = UniversalProbabilisticAdapter()

    nps = adapter.analyze_prediction(
        predictor="nps",
        prediction=82.0,
        historical_values=[70, 78, 81, 89, 94],
        samples=3000,
    )

    quality = adapter.analyze_prediction(
        predictor="quality",
        prediction=87.0,
        historical_values=[86, 87, 87, 88, 87],
        samples=3000,
    )

    assert nps.uncertainty != quality.uncertainty
    assert nps.historical_samples == 5
    assert quality.historical_samples == 5


def test_nps_uses_nps_scale():
    adapter = UniversalProbabilisticAdapter()

    result = adapter.analyze_prediction(
        predictor="nps",
        prediction=82.0,
        historical_values=[70, 75, 82, 88, 92],
        samples=2000,
    )

    assert result.downside < result.expected < result.upside
    assert result.simulations == 2000


def test_transfer_uses_transfer_scale():
    adapter = UniversalProbabilisticAdapter()

    result = adapter.analyze_prediction(
        predictor="transfer",
        prediction=9.0,
        historical_values=[7, 8, 9, 11, 13],
        samples=2000,
    )

    assert result.downside < result.expected < result.upside
    assert result.uncertainty > 0


def test_explicit_uncertainty_overrides_derived():
    adapter = UniversalProbabilisticAdapter()

    result = adapter.analyze_prediction(
        predictor="quality",
        prediction=87.0,
        historical_values=[70, 80, 90, 95],
        uncertainty=2.5,
        samples=1000,
    )

    assert result.uncertainty == 2.5
    assert result.simulations == 1000
