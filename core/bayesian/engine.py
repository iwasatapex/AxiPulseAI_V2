from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class BayesianResult:
    probability: float
    confidence: float
    posterior_mean: float
    posterior_std: float
    samples: int
    credible_interval_lower: float | None = None
    credible_interval_upper: float | None = None
    credible_level: float = 0.95
    prior_mean: float = 0.5
    prior_strength: float = 2.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BayesianInferenceEngine:
    """
    Universal Bayesian inference engine for probability estimation.

    The default model is Beta-Bernoulli inference. Observations are expected
    to be probabilities/success indicators in the [0, 1] interval.

    The engine is domain-neutral and contains no NPS, OH, ADIE, or predictor
    specific logic.
    """

    @staticmethod
    def _validate_prior(
        prior_mean: float,
        prior_strength: float,
        credible_level: float,
    ) -> None:
        if not np.isfinite(prior_mean):
            raise ValueError("prior_mean must be finite")

        if not 0.0 <= prior_mean <= 1.0:
            raise ValueError("prior_mean must be between 0 and 1")

        if not np.isfinite(prior_strength) or prior_strength <= 0.0:
            raise ValueError("prior_strength must be greater than 0")

        if not np.isfinite(credible_level):
            raise ValueError("credible_level must be finite")

        if not 0.0 < credible_level < 1.0:
            raise ValueError("credible_level must be between 0 and 1")

    @staticmethod
    def _validate_observations(
        observations: Iterable[float] | None,
    ) -> np.ndarray:
        if observations is None:
            raise ValueError("observations must not be None")

        values = np.asarray(list(observations), dtype=float)

        if values.ndim != 1:
            raise ValueError("observations must be one-dimensional")

        if values.size == 0:
            return values

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "observations must contain only finite values"
            )

        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(
                "observations must be between 0 and 1"
            )

        return values

    @staticmethod
    def _posterior_parameters(
        observations: np.ndarray,
        prior_mean: float,
        prior_strength: float,
    ) -> tuple[float, float]:
        successes = float(observations.sum())
        failures = float(observations.size - successes)

        alpha = prior_mean * prior_strength + successes
        beta_parameter = (
            (1.0 - prior_mean) * prior_strength + failures
        )

        return alpha, beta_parameter

    def infer(
        self,
        observations: list[float] | Iterable[float],
        prior_mean: float = 0.5,
        prior_strength: float = 2.0,
        credible_level: float = 0.95,
    ) -> BayesianResult:
        """
        Perform Beta-Bernoulli Bayesian inference.
        """
        self._validate_prior(
            prior_mean,
            prior_strength,
            credible_level,
        )

        values = self._validate_observations(observations)

        if values.size == 0:
            return BayesianResult(
                probability=float(prior_mean),
                confidence=0.0,
                posterior_mean=float(prior_mean),
                posterior_std=0.0,
                samples=0,
                credible_interval_lower=float(prior_mean),
                credible_interval_upper=float(prior_mean),
                credible_level=credible_level,
                prior_mean=prior_mean,
                prior_strength=prior_strength,
                metadata={
                    "source": "prior_only",
                    "model": "beta_bernoulli",
                    "observation_count": 0,
                },
            )

        alpha, beta_parameter = self._posterior_parameters(
            values,
            prior_mean,
            prior_strength,
        )

        posterior_mean = alpha / (alpha + beta_parameter)

        variance = (
            alpha
            * beta_parameter
            / (
                (alpha + beta_parameter) ** 2
                * (alpha + beta_parameter + 1.0)
            )
        )

        posterior_std = float(np.sqrt(variance))

        tail = (1.0 - credible_level) / 2.0

        credible_interval_lower = float(
            beta.ppf(tail, alpha, beta_parameter)
        )
        credible_interval_upper = float(
            beta.ppf(1.0 - tail, alpha, beta_parameter)
        )

        confidence = float(
            np.clip(
                1.0 - (posterior_std / 0.5),
                0.0,
                1.0,
            )
        )

        return BayesianResult(
            probability=float(posterior_mean),
            confidence=confidence,
            posterior_mean=float(posterior_mean),
            posterior_std=posterior_std,
            samples=int(values.size),
            credible_interval_lower=credible_interval_lower,
            credible_interval_upper=credible_interval_upper,
            credible_level=credible_level,
            prior_mean=float(prior_mean),
            prior_strength=float(prior_strength),
            metadata={
                "source": "observations_plus_prior",
                "model": "beta_bernoulli",
                "observation_count": int(values.size),
                "success_mass": float(values.sum()),
                "failure_mass": float(values.size - values.sum()),
            },
        )

    def update(
        self,
        prior_mean: float,
        prior_strength: float,
        observations: list[float] | Iterable[float],
        credible_level: float = 0.95,
    ) -> BayesianResult:
        """
        Explicit incremental Bayesian update.

        The returned posterior can be used as the prior for a subsequent
        update without coupling the engine to any application domain.
        """
        return self.infer(
            observations=observations,
            prior_mean=prior_mean,
            prior_strength=prior_strength,
            credible_level=credible_level,
        )

    def posterior_parameters(
        self,
        observations: list[float] | Iterable[float],
        prior_mean: float = 0.5,
        prior_strength: float = 2.0,
    ) -> tuple[float, float]:
        """
        Return posterior Beta(alpha, beta) parameters.
        """
        self._validate_prior(
            prior_mean,
            prior_strength,
            0.95,
        )

        values = self._validate_observations(observations)

        return self._posterior_parameters(
            values,
            prior_mean,
            prior_strength,
        )
