from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BayesianFeedbackState:
    """
    Persistent Bayesian belief state.

    The state represents prior information only. Observed outcomes
    are incorporated through explicit update() calls.

    No predictor-specific logic belongs here.
    """

    prior_mean: float = 0.5
    prior_strength: float = 2.0
    observations: int = 0
    successes: float = 0.0
    failures: float = 0.0
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.prior_mean <= 1.0:
            raise ValueError("prior_mean must be between 0 and 1")

        if self.prior_strength <= 0.0:
            raise ValueError("prior_strength must be greater than 0")

        if self.observations < 0:
            raise ValueError("observations must be non-negative")

        if self.successes < 0.0 or self.failures < 0.0:
            raise ValueError("successes and failures must be non-negative")

        if not np.isfinite(self.prior_mean):
            raise ValueError("prior_mean must be finite")

        if not np.isfinite(self.prior_strength):
            raise ValueError("prior_strength must be finite")

    @property
    def alpha(self) -> float:
        return (
            self.prior_mean * self.prior_strength
            + self.successes
        )

    @property
    def beta(self) -> float:
        return (
            (1.0 - self.prior_mean) * self.prior_strength
            + self.failures
        )

    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def posterior_strength(self) -> float:
        return self.alpha + self.beta

    def update(self, outcome: float) -> "BayesianFeedbackState":
        """
        Return a new state incorporating one observed Bernoulli outcome.

        outcome must be within [0, 1]. Fractional observations are
        supported for aggregated evidence.
        """
        if not np.isfinite(outcome):
            raise ValueError("outcome must be finite")

        if not 0.0 <= outcome <= 1.0:
            raise ValueError("outcome must be between 0 and 1")

        return BayesianFeedbackState(
            prior_mean=self.prior_mean,
            prior_strength=self.prior_strength,
            observations=self.observations + 1,
            successes=self.successes + float(outcome),
            failures=self.failures + float(1.0 - outcome),
            metadata=dict(self.metadata or {}),
        )

    def update_many(
        self,
        outcomes: list[float],
    ) -> "BayesianFeedbackState":
        """
        Return a new state incorporating observed outcomes.

        The original state is never mutated.
        """
        state = self

        for outcome in outcomes:
            state = state.update(outcome)

        return state

    def posterior(self) -> dict[str, float]:
        """
        Return the current posterior summary.
        """
        return {
            "mean": float(self.posterior_mean),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "strength": float(self.posterior_strength),
        }
