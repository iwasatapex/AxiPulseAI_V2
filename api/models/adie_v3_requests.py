"""Validated ADIE V3 decision request contract."""
from __future__ import annotations

from math import isfinite
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_SAMPLES = 10_000
MIN_SAMPLES = 1
MAX_SAMPLES = 100_000


class ADIEV3DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    observations: list[float] = Field(min_length=1)
    baseline: float
    uncertainty: float = Field(default=0.05, ge=0)
    samples: int = Field(default=DEFAULT_SAMPLES, ge=MIN_SAMPLES, le=MAX_SAMPLES)
    scenarios: list[dict[str, Any]] | None = None
    cutoff: Any = None
    provenance: Mapping[str, Any] | None = None
    observed: float | None = None
    observed_metrics: list[str] | None = None
    observed_nps: float | None = None
    observed_state: Mapping[str, Any] | None = None

    @field_validator("observations", mode="before")
    @classmethod
    def validate_observations(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("observations must be a non-empty sequence")
        result = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError("observations must contain numeric values")
            if not isfinite(float(item)):
                raise ValueError("observations must contain finite values")
            result.append(float(item))
        return result

    @field_validator("baseline", "uncertainty", mode="before")
    @classmethod
    def validate_numeric(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("value must be numeric")
        if not isfinite(float(value)):
            raise ValueError("value must be finite")
        return value
