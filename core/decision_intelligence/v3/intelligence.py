from dataclasses import dataclass
from core.probabilistic.adapter import UniversalProbabilisticAdapter


@dataclass
class ProbabilisticDecision:
    bayesian: object
    monte_carlo: object


class ADIEProbabilisticEngine:

    def __init__(self, adapter: UniversalProbabilisticAdapter | None = None):
        self.probabilistic = adapter or UniversalProbabilisticAdapter()

    def analyze(
        self,
        observations: list[float],
        baseline: float,
        uncertainty: float = 0.05,
        samples: int = 10000,
    ) -> ProbabilisticDecision:

        combined = self.probabilistic.from_combined(
            observations=observations,
            baseline=float(baseline),
            uncertainty=float(uncertainty),
            samples=int(samples),
            bounds=(0.0, 1.0),
            metadata={
                "scope": "v3_decision_level",
                "probability_domain": True,
            },
        )

        return ProbabilisticDecision(
            bayesian=combined.bayesian,
            monte_carlo=combined.monte_carlo,
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

