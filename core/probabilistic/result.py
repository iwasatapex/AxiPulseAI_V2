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

    metadata: Dict[str, Any] = Field(default_factory=dict)

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
