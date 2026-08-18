from dataclasses import dataclass

from core.probabilistic.adapter import UniversalProbabilisticAdapter


@dataclass
class ProbabilisticDecision:
    """V3 interpretation of one canonical probabilistic evaluation."""
    bayesian: object
    monte_carlo: object


class _DecisionBayesian:
    def __init__(self, info):
        self._info = info
        self.posterior_mean = info.posterior_mean
        self.posterior_std = info.posterior_std
        self.credible_interval_lower = info.credible_interval_lower
        self.credible_interval_upper = info.credible_interval_upper
        self.credible_level = info.credible_level
        self.confidence = float(info.credible_level or 0.0)
        self.probability = self.confidence


class _DecisionMonteCarlo:
    def __init__(self, info, result):
        self._info = info
        self.mean = result.expected_value
        self.p05 = info.percentile_5
        self.p50 = info.percentile_50
        self.p95 = info.percentile_95
        self.probability_positive = result.probability_of_target
        self.samples = info.num_simulations


class ADIEProbabilisticEngine:
    """V3 decision adapter; execution remains exclusively in core.probabilistic."""

    def __init__(self, probabilistic=None):
        self.probabilistic = probabilistic or UniversalProbabilisticAdapter()

    def analyze(
        self,
        observations: list[float],
        baseline: float,
        uncertainty: float = 0.05,
        samples: int = 10000,
    ) -> ProbabilisticDecision:
        result = self.probabilistic.infer(
            observations=[float(v) for v in observations],
            baseline=float(baseline),
            uncertainty=float(uncertainty),
            samples=int(samples),
        )
        if result.bayesian is None or result.monte_carlo is None:
            raise ValueError("Canonical probabilistic result must contain Bayesian and Monte Carlo evidence")
        return ProbabilisticDecision(
            bayesian=_DecisionBayesian(result.bayesian),
            monte_carlo=_DecisionMonteCarlo(result.monte_carlo, result),
        )


def analyze(
    observations: list[float],
    baseline: float,
    uncertainty: float = 0.05,
    samples: int = 10000,
) -> ProbabilisticDecision:
    return ADIEProbabilisticEngine().analyze(
        observations=observations, baseline=baseline,
        uncertainty=uncertainty, samples=samples,
    )


__all__ = ["ADIEProbabilisticEngine", "ProbabilisticDecision", "analyze"]
