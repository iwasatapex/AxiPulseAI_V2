from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.nps_predictor.inference import _axi_bayesian_update_0_10


@dataclass(frozen=True)
class BayesianResult:
    distribution: Any


class BayesianInferenceEngine:
    """
    V3 compatibility boundary for the existing AxiPulseAI
    Bayesian 0–10 implementation.

    This class delegates to the existing implementation and does
    not operate on or optimize the aggregated NPS scalar.
    """

    @staticmethod
    def infer(*args: Any, **kwargs: Any) -> BayesianResult:
        result = _axi_bayesian_update_0_10(*args, **kwargs)
        return BayesianResult(distribution=result)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._args = args
        self._kwargs = kwargs

    def run(self, *args: Any, **kwargs: Any) -> BayesianResult:
        if args or kwargs:
            return self.infer(*args, **kwargs)
        return self.infer(*self._args, **self._kwargs)


def infer(*args: Any, **kwargs: Any) -> BayesianResult:
    return BayesianInferenceEngine.infer(*args, **kwargs)


def _distribution_as_scores(
    distribution: Any,
) -> list[float]:
    """
    Extract an 11-length score-probability vector from either a dict keyed
    ``score_0..score_10`` or a length-11 sequence. Missing/unusable entries
    become 0.0 (never fabricated as evidence).
    """
    scores = [0.0] * 11

    if isinstance(distribution, dict):
        for i in range(11):
            value = distribution.get(f"score_{i}")
            if value is not None:
                try:
                    scores[i] = float(value)
                except (TypeError, ValueError):
                    scores[i] = 0.0
    else:
        try:
            seq = list(distribution)
        except TypeError:
            return scores
        for i, value in enumerate(seq):
            if i >= 11:
                break
            try:
                scores[i] = float(value)
            except (TypeError, ValueError):
                scores[i] = 0.0

    total = sum(max(0.0, s) for s in scores)
    if total <= 0.0:
        return [0.0] * 11
    return [max(0.0, s) / total for s in scores]


def score_distribution_probability_at_or_above(
    distribution: Any,
    target: float,
) -> float:
    """
    P(score >= target) from an 11-point NPS posterior distribution.

    ``target`` is on the 0..10 score scale. Returned as a probability in
    [0, 1]; 0.0 when no usable distribution is provided. Purely arithmetic on
    the model-produced distribution — no predictor call, no Monte Carlo.
    """
    scores = _distribution_as_scores(distribution)
    total = sum(scores)
    if total <= 0.0:
        return 0.0
    target_score = int(round(float(target)))
    mass = sum(
        s for i, s in enumerate(scores) if i >= target_score
    )
    return max(0.0, min(1.0, mass))


def expected_nps_from_distribution(
    distribution: Any,
) -> float:
    """Expected NPS score (0..10) from an 11-point score distribution."""
    scores = _distribution_as_scores(distribution)
    total = sum(scores)
    if total <= 0.0:
        return 0.0
    return sum(i * s for i, s in enumerate(scores)) / total


def expected_nps_business(
    distribution: Any,
) -> float:
    """
    Expected business NPS (-100..100) from an 11-point score distribution.

    business NPS = (%promoters - %detractors) * 100, where promoters are
    scores 9..10 and detractors are scores 0..6.
    """
    scores = _distribution_as_scores(distribution)
    total = sum(scores)
    if total <= 0.0:
        return 0.0
    promoter_mass = sum(scores[9:11])
    detractor_mass = sum(scores[0:7])
    return (promoter_mass - detractor_mass) * 100.0


def promoter_probability(
    distribution: Any,
) -> float:
    """P(a survey score is a promoter) = P(score >= 9)."""
    return score_distribution_probability_at_or_above(distribution, 9)


def nps_score_percentiles(
    distribution: Any,
    percentiles: tuple[float, float] = (0.05, 0.95),
) -> tuple[float | None, float | None]:
    """
    Discrete (5th, 95th) percentiles of the NPS score distribution via the
    cumulative distribution function. Returns (None, None) for an empty
    distribution.
    """
    scores = _distribution_as_scores(distribution)
    total = sum(scores)
    if total <= 0.0:
        return None, None

    cumulative = 0.0
    lower: float | None = None
    upper: float | None = None
    for score, prob in enumerate(scores):
        cumulative += prob
        if lower is None and cumulative >= percentiles[0]:
            lower = float(score)
        if upper is None and cumulative >= percentiles[1]:
            upper = float(score)
    return lower, upper


def nps_monte_carlo_percentiles(
    distribution: Any,
    n_surveys: int = 100,
    n_simulations: int = 2000,
    percentiles: tuple[float, float] = (0.05, 0.95),
    random_state: int = 42,
) -> tuple[float | None, float | None]:
    """
    Monte Carlo NPS quantiles (-100..100) from an 11-point score distribution.

    For each simulation, sample ``n_surveys`` survey scores from the score
    distribution and compute the business NPS of that simulated survey
    population:

        NPS = (promoters - detractors) / n_surveys * 100

    where promoters are scores 9..10 and detractors are scores 0..6 (the
    universal AxiPulseAI 0..10 definition). The (p05, p95) percentiles of the
    resulting NPS distribution are returned on the real -100..100 NPS scale.

    This is distinct from :func:`nps_score_percentiles`, which returns 0..10
    *score* quantiles. Returns (None, None) for an empty distribution.
    """
    import numpy as np

    scores = _distribution_as_scores(distribution)
    total = sum(scores)
    if total <= 0.0:
        return None, None

    probs = np.asarray([max(0.0, s) / total for s in scores], dtype=float)
    probs = probs / probs.sum()

    n_surveys = max(1, int(n_surveys))
    n_simulations = max(1, int(n_simulations))

    rng = np.random.default_rng(random_state)
    draws = rng.choice(11, size=(n_simulations, n_surveys), p=probs)

    detractors = (draws <= 6).sum(axis=1)
    promoters = (draws >= 9).sum(axis=1)
    nps = (promoters - detractors) / float(n_surveys) * 100.0

    return (
        float(np.percentile(nps, percentiles[0] * 100.0)),
        float(np.percentile(nps, percentiles[1] * 100.0)),
    )


__all__ = [
    "infer",
    "BayesianResult",
    "BayesianInferenceEngine",
    "score_distribution_probability_at_or_above",
    "expected_nps_from_distribution",
    "expected_nps_business",
    "promoter_probability",
    "nps_score_percentiles",
    "nps_monte_carlo_percentiles",
]

