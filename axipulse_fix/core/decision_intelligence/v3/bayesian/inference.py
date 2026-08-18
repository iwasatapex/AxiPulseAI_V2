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
    for all Bayesian inference and does NOT import from
    NPS predictor internals.
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

        Uses ``core.probabilistic.adapter.UniversalProbabilisticAdapter``
        for the scalar Beta-Bernoulli inference. For NPS 0–10 categorical
        inference, use ``core.probabilistic.categorical_nps``.
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
    observations = kwargs.pop("observations", args[0] if args else None)
    prior_mean = kwargs.pop("prior_mean", 0.5)
    prior_strength = kwargs.pop("prior_strength", 2.0)
    credible_level = kwargs.pop("credible_level", 0.95)
    if kwargs:
        raise TypeError(f"Unexpected Bayesian arguments: {sorted(kwargs)}")
    from core.probabilistic.adapter import UniversalProbabilisticAdapter
    result = UniversalProbabilisticAdapter().from_bayesian(
        observations=[float(v) for v in (observations or [])],
        prior_mean=prior_mean,
        prior_strength=prior_strength,
        credible_level=credible_level,
    )
    return BayesianResult(distribution=result.bayesian)


__all__ = ["BayesianInferenceEngine", "ProbabilisticDecision", "analyze"]
