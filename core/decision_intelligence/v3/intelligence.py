from dataclasses import dataclass
from core.bayesian import (
    BayesianInferenceEngine,
)
from core.monte_carlo import (
    MonteCarloEngine,
)


@dataclass
class ProbabilisticDecision:
    bayesian: object
    monte_carlo: object


class ADIEProbabilisticEngine:

    def __init__(self):
        self.bayesian = BayesianInferenceEngine()
        self.monte_carlo = MonteCarloEngine()

    def analyze(
        self,
        observations: list[float],
        baseline: float,
        uncertainty: float = 0.05,
        samples: int = 10000,
    ) -> ProbabilisticDecision:

        bayesian_result = self.bayesian.infer(
            observations=observations
        )

        monte_carlo_result = self.monte_carlo.simulate(
            baseline=baseline,
            uncertainty=uncertainty,
            samples=samples,
            # Decision-level baseline is a normalized probability (0..1); bound
            # the single draw to the probability domain so no surfaced value
            # (mean/p05/p50/p95/bins/counts) can exceed [0, 1].
            bounds=(0.0, 1.0),
        )

        return ProbabilisticDecision(
            bayesian=bayesian_result,
            monte_carlo=monte_carlo_result,
        )


def analyze(
    observations: list[float],
    baseline: float,
    uncertainty: float = 0.05,
    samples: int = 10000,
) -> ProbabilisticDecision:
    """Module-level convenience for the V3 probabilistic engine.

    Runs exactly one Bayesian inference and exactly one Monte Carlo simulation
    (the single-decision invariant) and returns the combined ``ProbabilisticDecision``.
    """
    return ADIEProbabilisticEngine().analyze(
        observations=observations,
        baseline=baseline,
        uncertainty=uncertainty,
        samples=samples,
    )


__all__ = ["ADIEProbabilisticEngine", "ProbabilisticDecision", "analyze"]

