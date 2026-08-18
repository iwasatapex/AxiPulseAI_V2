import numpy as np

from core.probabilistic.categorical_nps import (
    N_SCORES,
    attach_probabilistic_analysis,
    from_monte_carlo,
    from_observed_counts,
    from_prior_only,
    nps_from_score_counts,
)


def test_bayesian_prior_operates_on_all_11_scores():
    dist = np.array([0.00, 0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.09, 0.18, 0.25, 0.32])
    result = from_prior_only(dist, prior_strength=20.0)
    assert result.posterior.shape == (N_SCORES,)
    assert result.posterior_alpha.shape == (N_SCORES,)
    np.testing.assert_allclose(result.posterior.sum(), 1.0)
    np.testing.assert_allclose(result.posterior, dist / dist.sum())


def test_bayesian_observed_counts_update_score_distribution_not_scalar_nps():
    prior = np.full(N_SCORES, 1.0 / N_SCORES)
    observed = np.zeros(N_SCORES, dtype=int)
    observed[10] = 80
    observed[9] = 20
    result = from_observed_counts(prior, observed, prior_strength=10.0)
    assert result.posterior[10] > result.posterior[0]
    assert result.posterior[9] > result.posterior[5]
    assert result.metadata["observation_count"] == 100


def test_monte_carlo_samples_score_distribution_then_derives_nps():
    prior = np.array([0.01] * 7 + [0.04, 0.05, 0.25, 0.28], dtype=float)
    bayes = from_prior_only(prior, prior_strength=20.0)
    result = from_monte_carlo(
        bayes,
        total_surveys=200,
        simulations=500,
        random_state=7,
    )
    assert result["distribution_domain"] == "survey_scores_0_10"
    assert result["bayesian_parameter_sampling"] is True
    assert len(result["simulation_nps"]) == 500
    assert -100.0 <= result["p05"] <= result["p50"] <= result["p95"] <= 100.0
    assert np.isclose(result["mean_score_counts"].sum(), 200.0)


def test_attach_uses_score_distribution_as_probabilistic_domain():
    dist = np.array([0.01] * 7 + [0.04, 0.05, 0.25, 0.28], dtype=float)
    result = {
        "bayesian_score_distribution": {
            f"score_{i}": float(dist[i]) for i in range(N_SCORES)
        },
        "nps": 10.0,
    }
    enriched = attach_probabilistic_analysis(
        result,
        total_surveys=250,
        simulations=300,
        seed=11,
        prior_strength=20.0,
    )
    assert enriched["probabilistic_domain"] == "survey_scores_0_10"
    assert len(enriched["monte_carlo_nps"]) == 300
    assert sum(enriched["probabilistic_score_counts"].values()) == 250
    score_dist = np.array(
        [enriched["bayesian_score_distribution"][f"score_{i}"] for i in range(11)]
    )
    assert np.isclose(score_dist.sum(), 1.0)


def test_nps_is_derived_from_score_counts():
    counts = np.array([5, 0, 0, 0, 0, 0, 5, 10, 10, 30, 40])
    result = nps_from_score_counts(counts)
    assert result["total_surveys"] == 100
    assert result["promoters"] == 70
    assert result["passives"] == 20
    assert result["detractors"] == 10
    assert result["nps"] == 60.0
