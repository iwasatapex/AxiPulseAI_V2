from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adapter import UniversalProbabilisticAdapter
from .categorical_nps import attach_probabilistic_analysis, nps_from_score_counts
from .result import BayesianInfo, MonteCarloInfo, ProbabilisticResult


@dataclass(frozen=True)
class UniversalPredictionEnvelope:
    """
    Additive wrapper for an existing scalar prediction.

    The original prediction is preserved exactly.
    Probabilistic information is attached separately.
    """

    prediction: Any
    probabilistic: ProbabilisticResult
    metadata: dict[str, Any]


def wrap_prediction(
    prediction: float,
    *,
    target: float | None = None,
    uncertainty: float = 0.05,
    observations: list[float] | None = None,
    samples: int = 10000,
    seed: int = 0,
    metadata: dict[str, Any] | None = None,
) -> UniversalPredictionEnvelope:
    """
    Convert an existing scalar prediction into the universal
    probabilistic representation without changing the prediction.
    """

    adapter = UniversalProbabilisticAdapter()

    result = adapter.infer(
        observations=observations or [],
        baseline=float(prediction),
        target=target,
        uncertainty=float(uncertainty),
        samples=int(samples),
        seed=int(seed),
    )

    return UniversalPredictionEnvelope(
        prediction=prediction,
        probabilistic=result,
        metadata=dict(metadata or {}),
    )


def wrap_nps_prediction(
    prediction: float,
    *,
    score_distribution: dict[str, float] | list[float],
    total_surveys: int,
    observed_score_counts: list[int] | None = None,
    simulations: int = 10000,
    seed: int = 0,
    prior_strength: float = 20.0,
    metadata: dict[str, Any] | None = None,
) -> UniversalPredictionEnvelope:
    """Wrap an NPS prediction using ONLY the canonical 0..10 survey path.

    The scalar ``prediction`` is preserved as the point forecast, but it is
    never used as the Bayesian or Monte Carlo uncertainty source.
    """
    if total_surveys <= 0:
        raise ValueError("total_surveys must be greater than zero for NPS uncertainty")

    if isinstance(score_distribution, dict):
        distribution = {
            f"score_{i}": float(score_distribution.get(f"score_{i}", 0.0))
            for i in range(11)
        }
    else:
        if len(score_distribution) != 11:
            raise ValueError("score_distribution must contain exactly 11 buckets")
        distribution = {f"score_{i}": float(score_distribution[i]) for i in range(11)}

    analysis = attach_probabilistic_analysis(
        {
            "nps": float(prediction),
            "bayesian_score_distribution": distribution,
        },
        total_surveys=int(total_surveys),
        observed_counts=observed_score_counts,
        simulations=int(simulations),
        seed=int(seed),
        prior_strength=float(prior_strength),
    )

    posterior = analysis["bayesian_score_distribution"]
    mc_samples = analysis["monte_carlo_nps"]
    mc_p05 = float(analysis["monte_carlo_nps_p05"])
    mc_p50 = float(analysis["monte_carlo_nps_p50"])
    mc_p95 = float(analysis["monte_carlo_nps_p95"])

    posterior_expected_nps = float(
        nps_from_score_counts(
            list(analysis["monte_carlo_score_distribution"].values())
        )["nps"]
    )

    probabilistic = ProbabilisticResult(
        most_likely=mc_p50,
        likely_range_lower=mc_p05,
        likely_range_upper=mc_p95,
        range_confidence=0.90,
        expected_value=float(sum(mc_samples) / len(mc_samples)) if mc_samples else posterior_expected_nps,
        uncertainty=mc_p95 - mc_p05,
        confidence=0.95,
        bayesian=BayesianInfo(
            posterior_mean=posterior_expected_nps,
            posterior_std=max((mc_p95 - mc_p05) / 3.289707253, 0.0),
            credible_interval_lower=mc_p05,
            credible_interval_upper=mc_p95,
            credible_level=0.95,
            prior_type="dirichlet",
            metadata={
                "distribution_domain": "survey_scores_0_10",
                "score_distribution": posterior,
                "scalar_nps_prediction_not_used_for_uncertainty": True,
                "observed_score_counts_used": observed_score_counts is not None,
            },
        ),
        monte_carlo=MonteCarloInfo(
            num_simulations=int(simulations),
            distribution_samples=[float(v) for v in mc_samples],
            percentile_5=mc_p05,
            percentile_50=mc_p50,
            percentile_95=mc_p95,
            metadata={
                "distribution_domain": "survey_scores_0_10",
                "nps_derived_from_score_counts": True,
                "scalar_nps_prediction_not_used_for_uncertainty": True,
            },
        ),
        metadata={
            **dict(metadata or {}),
            "predictor": "nps",
            "metric": "nps",
            "distribution_authoritative": True,
            "uncertainty_domain": "survey_scores_0_10",
        },
    )

    return UniversalPredictionEnvelope(
        prediction=float(prediction),
        probabilistic=probabilistic,
        metadata={
            **dict(metadata or {}),
            "predictor": "nps",
            "metric": "nps",
            "distribution_authoritative": True,
        },
    )
