"""Universal probabilistic result contract for AxiPulseAI.

This module defines the shared probabilistic output contract only.
It does not implement Bayesian inference or Monte Carlo simulation.
Existing project implementations remain responsible for those calculations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class BayesianInfo(BaseModel):
    """Optional Bayesian posterior information."""

    posterior_mean: Optional[float] = None
    posterior_std: Optional[float] = Field(None, ge=0)
    credible_interval_lower: Optional[float] = None
    credible_interval_upper: Optional[float] = None
    credible_level: Optional[float] = Field(None, ge=0, le=1)
    prior_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def probability(self) -> Optional[float]:
        """Backward-compatible probability view of the Bayesian result."""
        return self.metadata.get("probability", self.posterior_mean)

    @property
    def confidence(self) -> Optional[float]:
        """Backward-compatible confidence view."""
        return self.credible_level

    @property
    def samples(self) -> Optional[int]:
        """Number of Bayesian observations represented by the posterior.

        Compatibility mapping:
        ``samples`` is sourced from the canonical observation count.
        Older producers may expose ``samples`` directly.
        """
        value = self.metadata.get("observation_count")
        if value is None:
            value = self.metadata.get("observation_count")
        if value is None:
            value = self.metadata.get("samples")
        return int(value) if value is not None else None

    @model_validator(mode="after")
    def validate_credible_interval(self) -> "BayesianInfo":
        if (
            self.credible_interval_lower is not None
            and self.credible_interval_upper is not None
            and self.credible_interval_lower > self.credible_interval_upper
        ):
            raise ValueError(
                "credible_interval_lower must be <= credible_interval_upper"
            )
        return self


class MonteCarloInfo(BaseModel):
    """Optional Monte Carlo distribution information."""

    num_simulations: Optional[int] = Field(None, ge=1)
    distribution_samples: Optional[List[float]] = None

    percentile_5: Optional[float] = None
    percentile_50: Optional[float] = None
    percentile_95: Optional[float] = None

    other_percentiles: Dict[float, float] = Field(default_factory=dict)

    # Legacy V3 compatibility field: fraction of the single Monte Carlo
    # sample draw that landed strictly above zero. It is derived from the
    # existing sample summary (never a second simulation) and exposed here
    # so V3 wrappers can keep reading ``monte_carlo.probability_positive``.
    probability_positive: Optional[float] = Field(None, ge=0, le=1)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def samples(self) -> Optional[int]:
        """Backward-compatible alias for num_simulations."""
        return self.num_simulations

    @property
    def p05(self) -> Optional[float]:
        """Backward-compatible alias for percentile_5."""
        return self.percentile_5

    @property
    def p50(self) -> Optional[float]:
        """Backward-compatible alias for percentile_50."""
        return self.percentile_50

    @property
    def p95(self) -> Optional[float]:
        """Backward-compatible alias for percentile_95."""
        return self.percentile_95

    @property
    def mean(self) -> Optional[float]:
        """Canonical Monte Carlo mean.

        Prefer the explicit canonical mean; fall back to the median only
        when a producer supplied no mean at all.
        """
        value = self.metadata.get("mean")
        if value is not None:
            return float(value)

        if self.distribution_samples:
            return float(sum(self.distribution_samples) / len(self.distribution_samples))

        return self.percentile_50

    @property
    def success_count(self) -> int:
        """Decision-level success count when supplied by the engine."""
        return int(self.metadata.get("success_count", 0))

    @property
    def failure_count(self) -> int:
        """Decision-level failure count when supplied by the engine."""
        return int(self.metadata.get("failure_count", 0))

    @property
    def distribution(self) -> list:
        """Decision-level distribution when supplied by the engine."""
        return list(self.metadata.get("distribution", []))

    @field_validator("other_percentiles")
    @classmethod
    def validate_percentile_keys(
        cls,
        value: Dict[float, float],
    ) -> Dict[float, float]:
        for percentile in value:
            if not 0 <= percentile <= 1:
                raise ValueError("Percentile keys must be between 0 and 1")
        return value

    @model_validator(mode="after")
    def validate_percentiles(self) -> "MonteCarloInfo":
        values = []

        if self.percentile_5 is not None:
            values.append((0.05, self.percentile_5))

        if self.percentile_50 is not None:
            values.append((0.50, self.percentile_50))

        if self.percentile_95 is not None:
            values.append((0.95, self.percentile_95))

        values.extend(self.other_percentiles.items())
        values.sort(key=lambda item: item[0])

        for (_, previous), (_, current) in zip(values, values[1:]):
            if previous > current:
                raise ValueError(
                    "Percentile values must be non-decreasing"
                )

        return self


class ProbabilisticResultBase(BaseModel):
    """Universal extensible probabilistic output contract."""

    most_likely: Optional[float] = Field(
        None,
        description="Central or most-likely predicted outcome.",
    )

    likely_range_lower: Optional[float] = Field(
        None,
        description="Lower bound of the likely outcome range.",
    )

    likely_range_upper: Optional[float] = Field(
        None,
        description="Upper bound of the likely outcome range.",
    )

    range_confidence: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Confidence level associated with the likely range.",
    )

    probability_of_target: Optional[float] = Field(
        None,
        ge=0,
        le=1,
    )

    probability_of_failure: Optional[float] = Field(
        None,
        ge=0,
        le=1,
    )

    expected_value: Optional[float] = None

    uncertainty: Optional[float] = Field(
        None,
        ge=0,
    )

    risk: Optional[float] = Field(
        None,
        ge=0,
    )

    confidence: Optional[float] = Field(
        None,
        ge=0,
        le=1,
    )

    bayesian: Optional[BayesianInfo] = None
    monte_carlo: Optional[MonteCarloInfo] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.now)

    contract_version: str = "1.0.0"

    @model_validator(mode="after")
    def validate_range(self) -> "ProbabilisticResultBase":
        if (
            self.likely_range_lower is not None
            and self.likely_range_upper is not None
            and self.likely_range_lower > self.likely_range_upper
        ):
            raise ValueError(
                "likely_range_lower must be <= likely_range_upper"
            )
        return self


ProbabilisticResult = ProbabilisticResultBase
