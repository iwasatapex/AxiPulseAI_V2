import pytest

from core.decision_intelligence.v3.integration import UniversalProbabilisticAdapter


def test_nps_scalar_uncertainty_is_rejected():
    adapter = UniversalProbabilisticAdapter()
    with pytest.raises(ValueError, match="scalar NPS uncertainty is prohibited"):
        adapter.analyze_prediction(
            predictor="nps",
            prediction=82.0,
            historical_values=[70, 75, 82],
            samples=100,
        )


def test_nps_uncertainty_uses_score_distribution_not_scalar_prediction():
    adapter = UniversalProbabilisticAdapter()
    dist = {f"score_{i}": 0.0 for i in range(11)}
    dist[10] = 1.0

    high = adapter.analyze_prediction(
        predictor="nps",
        prediction=82.0,
        score_distribution=dist,
        total_surveys=100,
        samples=500,
        seed=7,
    )
    low = adapter.analyze_prediction(
        predictor="nps",
        prediction=-50.0,
        score_distribution=dist,
        total_surveys=100,
        samples=500,
        seed=7,
    )

    assert high.expected == low.expected
    assert high.downside == low.downside
    assert high.upside == low.upside
    assert high.monte_carlo.metadata["distribution_domain"] == "survey_scores_0_10"
    assert high.monte_carlo.metadata["nps_derived_from_score_counts"] is True


def test_nps_observed_counts_update_bayesian_score_distribution():
    adapter = UniversalProbabilisticAdapter()
    dist = {f"score_{i}": 1.0 / 11.0 for i in range(11)}
    counts = [0] * 11
    counts[10] = 100

    result = adapter.analyze_prediction(
        predictor="nps",
        prediction=0.0,
        score_distribution=dist,
        total_surveys=100,
        observed_score_counts=counts,
        samples=500,
        seed=1,
    )

    posterior = result.bayesian.posterior_score_distribution
    assert posterior["score_10"] > posterior["score_0"]
    assert result.expected > 0
