from core.decision_intelligence.v3.integration import (
    UniversalProbabilisticAdapter,
)


def test_nps_predictor():
    adapter = UniversalProbabilisticAdapter()

    result = adapter.analyze_prediction(
        predictor="nps",
        prediction=82.0,
        historical_values=[0.80, 0.82, 0.81, 0.83],
        score_distribution={f"score_{i}": 1.0 / 11.0 for i in range(11)},
        total_surveys=250,
        samples=2000,
    )

    assert result.predictor == "nps"
    assert 0.0 <= result.probability <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.downside < result.upside


def test_kpi_predictor():
    adapter = UniversalProbabilisticAdapter()

    result = adapter.analyze_prediction(
        predictor="quality",
        prediction=87.0,
        historical_values=[0.85, 0.87, 0.88, 0.86],
        uncertainty=1.0,
        samples=2000,
    )

    assert result.predictor == "quality"
    assert result.expected > 0
    assert result.downside < result.upside


def test_operations_health_predictor():
    adapter = UniversalProbabilisticAdapter()

    result = adapter.analyze_prediction(
        predictor="operations_health",
        prediction=82.0,
        historical_values=[0.78, 0.81, 0.83, 0.82],
        uncertainty=2.0,
        samples=2000,
    )

    assert result.predictor == "operations_health"
    assert result.monte_carlo.samples == 2000


def test_multiple_predictors():
    adapter = UniversalProbabilisticAdapter()

    predictors = {
        "nps": 82.0,
        "quality": 87.0,
        "competency": 93.0,
        "attendance": 90.0,
        "release": 60.0,
        "transfer": 9.0,
        "operations_health": 82.0,
    }

    for name, prediction in predictors.items():
        kwargs = {}
        if name == "nps":
            kwargs = {
                "score_distribution": {f"score_{i}": 1.0 / 11.0 for i in range(11)},
                "total_surveys": 250,
            }
        result = adapter.analyze_prediction(
            predictor=name,
            prediction=prediction,
            historical_values=[0.80, 0.82, 0.81],
            samples=1000,
            **kwargs,
        )

        assert result.predictor == name
        assert 0.0 <= result.probability <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.downside <= result.upside
