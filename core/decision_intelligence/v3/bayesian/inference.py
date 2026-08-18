from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BayesianResult:
    distribution: Any


class BayesianInferenceEngine:
    """
    V3 compatibility boundary for Bayesian inference.

    This class uses the canonical ``core.probabilistic`` adapter
    for all Bayesian inference.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._args = args
        self._kwargs = kwargs

    def run(self, *args: Any, **kwargs: Any) -> BayesianResult:
        if args or kwargs:
            return self.infer(*args, **kwargs)
        return self.infer(*self._args, **self._kwargs)

    def infer(
        self,
        observations: list[float] | None = None,
        prior_mean: float = 0.5,
        prior_strength: float = 2.0,
        credible_level: float = 0.95,
    ) -> BayesianResult:
        """
        Perform Bayesian inference using the canonical probabilistic adapter.

        Uses the canonical ``core.probabilistic`` adapter for scalar inference.
        """
        from core.probabilistic.adapter import UniversalProbabilisticAdapter

        adapter = UniversalProbabilisticAdapter()

        # For scalar Bayesian inference on [0,1] observations
        obs = [float(v) for v in (observations or [])]

        result = adapter.from_bayesian(
            observations=obs,
            prior_mean=prior_mean,
            prior_strength=prior_strength,
            credible_level=credible_level,
        )

        return BayesianResult(
            distribution=result.bayesian,
        )


def infer(*args: Any, **kwargs: Any) -> BayesianResult:
    """Module-level convenience for the V3 Bayesian inference."""
    engine_type = BayesianInferenceEngine
    return engine_type().infer(*args, **kwargs)


def expected_nps_from_distribution(distribution: Any) -> float:
    """Return the expected 0-10 NPS score from a score distribution.

    This preserves the V3 compatibility API. The expected score is the
    probability-weighted mean of scores 0 through 10.
    """
    if not distribution:
        return 0.0

    if isinstance(distribution, dict):
        probabilities = [
            float(distribution.get(f"score_{score}", 0.0))
            for score in range(11)
        ]
    else:
        probabilities = [float(value) for value in distribution]

    if len(probabilities) != 11:
        raise ValueError("NPS distribution must contain exactly 11 scores")

    total = sum(probabilities)
    if total <= 0.0:
        return 0.0

    probabilities = [value / total for value in probabilities]

    return float(
        sum(
            score * probability
            for score, probability in enumerate(probabilities)
        )
    )


def expected_nps_business(distribution: Any) -> float:
    """Return business-scale NPS from a score_0..score_10 distribution.

    NPS is defined canonically as:
        (P(score 9 or 10) - P(score 0 through 6)) * 100

    Scores 7 and 8 are passives and contribute zero.
    """
    if not distribution:
        return 0.0

    if isinstance(distribution, dict):
        probabilities = [
            float(distribution.get(f"score_{score}", 0.0))
            for score in range(11)
        ]
    else:
        probabilities = [float(value) for value in distribution]

    if len(probabilities) != 11:
        raise ValueError("NPS distribution must contain exactly 11 scores")

    total = sum(probabilities)
    if total <= 0.0:
        return 0.0

    probabilities = [value / total for value in probabilities]

    detractors = sum(probabilities[0:7])
    promoters = sum(probabilities[9:11])

    return float((promoters - detractors) * 100.0)


def score_distribution_probability_at_or_above(
    distribution: Any,
    threshold: float,
) -> float:
    """Return P(score >= threshold) from an 11-point score distribution.

    ``threshold`` is on the 0..10 score scale. Pure arithmetic over the
    supplied distribution — no predictor call, no Monte Carlo.
    """
    if not distribution:
        return 0.0

    if isinstance(distribution, dict):
        probabilities = [
            float(distribution.get(f"score_{score}", 0.0))
            for score in range(11)
        ]
    else:
        probabilities = [float(value) for value in distribution]

    if len(probabilities) != 11:
        raise ValueError("NPS distribution must contain exactly 11 scores")

    total = sum(probabilities)
    if total <= 0.0:
        return 0.0

    probabilities = [value / total for value in probabilities]

    target = int(round(float(threshold)))
    return max(
        0.0,
        min(1.0, sum(probabilities[target:])),
    )


def promoter_probability(distribution: Any) -> float:
    """Return P(score >= 9), i.e. the promoter probability.

    Promoters are survey scores 9 and 10.
    """
    return score_distribution_probability_at_or_above(distribution, 9)


def expected_nps_from_distribution(distribution: Any) -> float:
    """Return the expected 0..10 survey score from a score distribution."""
    if not distribution:
        return 0.0

    if isinstance(distribution, dict):
        probabilities = [
            float(distribution.get(f"score_{score}", 0.0))
            for score in range(11)
        ]
    else:
        probabilities = [float(value) for value in distribution]

    if len(probabilities) != 11:
        raise ValueError("NPS distribution must contain exactly 11 scores")

    total = sum(probabilities)
    if total <= 0.0:
        return 0.0

    probabilities = [value / total for value in probabilities]

    return float(
        sum(score * probability for score, probability in enumerate(probabilities))
    )


def nps_score_percentiles(distribution: Any) -> tuple[float, float]:
    """Return 5th/95th percentile survey scores on the 0..10 scale."""
    if not distribution:
        return (0.0, 0.0)

    if isinstance(distribution, dict):
        probabilities = [
            float(distribution.get(f"score_{score}", 0.0))
            for score in range(11)
        ]
    else:
        probabilities = [float(value) for value in distribution]

    if len(probabilities) != 11:
        raise ValueError("NPS distribution must contain exactly 11 scores")

    total = sum(probabilities)
    if total <= 0.0:
        return (0.0, 0.0)

    probabilities = [value / total for value in probabilities]

    cumulative = 0.0
    p05 = None
    p95 = None

    for score, probability in enumerate(probabilities):
        cumulative += probability

        if p05 is None and cumulative >= 0.05:
            p05 = float(score)

        if p95 is None and cumulative >= 0.95:
            p95 = float(score)
            break

    return (
        0.0 if p05 is None else p05,
        10.0 if p95 is None else p95,
    )


def nps_monte_carlo_percentiles(
    distribution: Any,
    samples: int = 10000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return 5th/95th percentile NPS from one categorical Monte Carlo draw.

    Each sampled survey score is converted to its business NPS contribution:
      0..6 -> -100
      7..8 -> 0
      9..10 -> +100
    """
    if not distribution:
        return (0.0, 0.0)

    if isinstance(distribution, dict):
        probabilities = [
            float(distribution.get(f"score_{score}", 0.0))
            for score in range(11)
        ]
    else:
        probabilities = [float(value) for value in distribution]

    if len(probabilities) != 11:
        raise ValueError("NPS distribution must contain exactly 11 scores")

    total = sum(probabilities)
    if total <= 0.0:
        return (0.0, 0.0)

    probabilities = [value / total for value in probabilities]

    import numpy as np

    rng = np.random.default_rng(seed)
    scores = rng.choice(
        np.arange(11),
        size=int(samples),
        p=probabilities,
    )

    nps_values = np.where(
        scores <= 6,
        -100.0,
        np.where(scores >= 9, 100.0, 0.0),
    )

    return (
        float(np.percentile(nps_values, 5)),
        float(np.percentile(nps_values, 95)),
    )


__all__ = [
    "BayesianResult",
    "BayesianInferenceEngine",
    "infer",
    "score_distribution_probability_at_or_above",
    "expected_nps_business",
    "expected_nps_from_distribution",
    "promoter_probability",
    "nps_score_percentiles",
    "nps_monte_carlo_percentiles",
]
