from __future__ import annotations

from math import erf, sqrt
from typing import Any

from core.bayesian import BayesianInferenceEngine
from core.monte_carlo import MonteCarloEngine

from core.probabilistic.result import (
    BayesianInfo,
    MonteCarloInfo,
    ProbabilisticResult,
)


class UniversalProbabilisticAdapter:
    """
    Universal bridge between Bayesian / Monte Carlo engines and the
    Phase 1 ProbabilisticResult contract.

    This class contains no NPS, OH, ADIE, or predictor-specific logic.
    """

    def __init__(
        self,
        bayesian: BayesianInferenceEngine | None = None,
        monte_carlo: MonteCarloEngine | None = None,
    ) -> None:
        self.bayesian = bayesian or BayesianInferenceEngine()
        self.monte_carlo = monte_carlo or MonteCarloEngine()

    def from_bayesian(
        self,
        observations: list[float],
        prior_mean: float = 0.5,
        prior_strength: float = 2.0,
        credible_level: float = 0.95,
        metadata: dict[str, Any] | None = None,
    ) -> ProbabilisticResult:

        result = self.bayesian.infer(
            observations=observations,
            prior_mean=prior_mean,
            prior_strength=prior_strength,
            credible_level=credible_level,
        )

        return ProbabilisticResult(
            most_likely=result.posterior_mean,
            likely_range_lower=result.credible_interval_lower,
            likely_range_upper=result.credible_interval_upper,
            range_confidence=result.credible_level,
            expected_value=result.posterior_mean,
            uncertainty=result.posterior_std,
            confidence=result.confidence,
            bayesian=BayesianInfo(
                posterior_mean=result.posterior_mean,
                posterior_std=result.posterior_std,
                credible_interval_lower=result.credible_interval_lower,
                credible_interval_upper=result.credible_interval_upper,
                credible_level=result.credible_level,
                prior_type="beta",
                metadata=result.metadata,
            ),
            metadata=metadata or {},
        )

    def from_monte_carlo(
        self,
        baseline: float,
        uncertainty: float = 0.05,
        samples: int = 10000,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> ProbabilisticResult:

        result = self.monte_carlo.simulate(
            baseline=baseline,
            uncertainty=uncertainty,
            samples=samples,
            seed=seed,
        )

        return ProbabilisticResult(
            most_likely=result.p50,
            likely_range_lower=result.p05,
            likely_range_upper=result.p95,
            range_confidence=0.90,
            probability_of_target=result.probability_positive,
            expected_value=result.mean,
            uncertainty=result.p95 - result.p05,
            confidence=1.0,
            monte_carlo=MonteCarloInfo(
                num_simulations=result.samples,
                percentile_5=result.p05,
                percentile_50=result.p50,
                percentile_95=result.p95,
                metadata=result.metadata,
            ),
            metadata=metadata or {},
        )

    @staticmethod
    def _probability_at_or_above(
        value: float,
        target: float,
        uncertainty: float,
    ) -> float:

        if uncertainty <= 0.0:
            return 1.0 if value >= target else 0.0

        z = (value - target) / uncertainty

        probability = 0.5 * (
            1.0 + erf(z / sqrt(2.0))
        )

        return max(0.0, min(1.0, probability))

    def from_combined(
        self,
        *,
        observations: list[float],
        baseline: float | None = None,
        target: float | None = None,
        prior_mean: float = 0.5,
        prior_strength: float = 2.0,
        credible_level: float = 0.95,
        uncertainty: float = 0.05,
        samples: int = 10000,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> ProbabilisticResult:

        bayesian = self.bayesian.infer(
            observations=observations,
            prior_mean=prior_mean,
            prior_strength=prior_strength,
            credible_level=credible_level,
        )

        if baseline is None:
            baseline = bayesian.posterior_mean

        monte_carlo = self.monte_carlo.simulate(
            baseline=float(baseline),
            uncertainty=uncertainty,
            samples=samples,
            seed=seed,
        )

        # Bayesian posterior and Monte Carlo baseline are deliberately
        # kept as separate pieces of evidence. The adapter combines
        # their output only at the universal result boundary.

        most_likely = float(monte_carlo.p50)

        probability_of_target = None
        probability_of_failure = None

        if target is not None:
            # Convert percentile spread into an approximate standard
            # deviation for the normal Monte Carlo distribution.
            mc_std = max(
                (monte_carlo.p95 - monte_carlo.p05) / 3.289707253,
                0.0,
            )

            probability_of_target = self._probability_at_or_above(
                value=monte_carlo.mean,
                target=target,
                uncertainty=mc_std,
            )

            probability_of_failure = (
                1.0 - probability_of_target
            )

        combined_metadata = dict(metadata or {})

        combined_metadata.update(
            {
                "adapter": "UniversalProbabilisticAdapter",
                "bayesian_information": {
                    "posterior_mean": bayesian.posterior_mean,
                    "posterior_std": bayesian.posterior_std,
                    "samples": bayesian.samples,
                    "credible_interval_lower":
                        bayesian.credible_interval_lower,
                    "credible_interval_upper":
                        bayesian.credible_interval_upper,
                    "credible_level":
                        bayesian.credible_level,
                },
                "monte_carlo_information": {
                    "mean": monte_carlo.mean,
                    "p05": monte_carlo.p05,
                    "p50": monte_carlo.p50,
                    "p95": monte_carlo.p95,
                    "samples": monte_carlo.samples,
                    "seed": monte_carlo.metadata.get("seed"),
                    "distribution":
                        monte_carlo.metadata.get("distribution"),
                },
            }
        )

        return ProbabilisticResult(
            most_likely=most_likely,
            likely_range_lower=monte_carlo.p05,
            likely_range_upper=monte_carlo.p95,
            range_confidence=0.90,
            probability_of_target=probability_of_target,
            probability_of_failure=probability_of_failure,
            expected_value=monte_carlo.mean,
            uncertainty=monte_carlo.p95 - monte_carlo.p05,
            risk=(
                probability_of_failure
                if probability_of_failure is not None
                else None
            ),
            confidence=bayesian.confidence,
            bayesian=BayesianInfo(
                posterior_mean=bayesian.posterior_mean,
                posterior_std=bayesian.posterior_std,
                credible_interval_lower=
                    bayesian.credible_interval_lower,
                credible_interval_upper=
                    bayesian.credible_interval_upper,
                credible_level=bayesian.credible_level,
                prior_type="beta",
                metadata=bayesian.metadata,
            ),
            monte_carlo=MonteCarloInfo(
                num_simulations=monte_carlo.samples,
                percentile_5=monte_carlo.p05,
                percentile_50=monte_carlo.p50,
                percentile_95=monte_carlo.p95,
                metadata=monte_carlo.metadata,
            ),
            metadata=combined_metadata,
        )

    def infer(
        self,
        observations: list[float],
        *,
        baseline: float | None = None,
        target: float | None = None,
        prior_mean: float = 0.5,
        prior_strength: float = 2.0,
        credible_level: float = 0.95,
        uncertainty: float = 0.05,
        samples: int = 10000,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> ProbabilisticResult:

        return self.from_combined(
            observations=observations,
            baseline=baseline,
            target=target,
            prior_mean=prior_mean,
            prior_strength=prior_strength,
            credible_level=credible_level,
            uncertainty=uncertainty,
            samples=samples,
            seed=seed,
            metadata=metadata,
        )


def adapt(
    *,
    bayesian_result: Any | None = None,
    monte_carlo_result: Any | None = None,
    target: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> ProbabilisticResult:

    if bayesian_result is None and monte_carlo_result is None:
        raise ValueError(
            "at least one probabilistic engine result is required"
        )

    adapter = UniversalProbabilisticAdapter()

    if bayesian_result is not None and monte_carlo_result is None:
        return ProbabilisticResult(
            most_likely=bayesian_result.posterior_mean,
            likely_range_lower=(
                bayesian_result.credible_interval_lower
            ),
            likely_range_upper=(
                bayesian_result.credible_interval_upper
            ),
            range_confidence=bayesian_result.credible_level,
            expected_value=bayesian_result.posterior_mean,
            uncertainty=bayesian_result.posterior_std,
            confidence=bayesian_result.confidence,
            bayesian=BayesianInfo(
                posterior_mean=bayesian_result.posterior_mean,
                posterior_std=bayesian_result.posterior_std,
                credible_interval_lower=(
                    bayesian_result.credible_interval_lower
                ),
                credible_interval_upper=(
                    bayesian_result.credible_interval_upper
                ),
                credible_level=bayesian_result.credible_level,
                prior_type="beta",
                metadata=bayesian_result.metadata,
            ),
            metadata=metadata or {},
        )

    if monte_carlo_result is not None and bayesian_result is None:
        probability_target = None

        if target is not None:
            mc_std = max(
                (
                    monte_carlo_result.p95
                    - monte_carlo_result.p05
                ) / 3.289707253,
                0.0,
            )

            probability_target = (
                adapter._probability_at_or_above(
                    value=monte_carlo_result.mean,
                    target=target,
                    uncertainty=mc_std,
                )
            )

        return ProbabilisticResult(
            most_likely=monte_carlo_result.p50,
            likely_range_lower=monte_carlo_result.p05,
            likely_range_upper=monte_carlo_result.p95,
            range_confidence=0.90,
            probability_of_target=probability_target,
            probability_of_failure=(
                None
                if probability_target is None
                else 1.0 - probability_target
            ),
            expected_value=monte_carlo_result.mean,
            uncertainty=(
                monte_carlo_result.p95
                - monte_carlo_result.p05
            ),
            confidence=1.0,
            monte_carlo=MonteCarloInfo(
                num_simulations=monte_carlo_result.samples,
                percentile_5=monte_carlo_result.p05,
                percentile_50=monte_carlo_result.p50,
                percentile_95=monte_carlo_result.p95,
                metadata=monte_carlo_result.metadata,
            ),
            metadata=metadata or {},
        )

    raise ValueError(
        "combined adaptation requires UniversalProbabilisticAdapter.infer "
        "or from_combined"
    )


__all__ = [
    "UniversalProbabilisticAdapter",
    "adapt",
]
