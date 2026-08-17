"""Validated request contracts for predictor endpoints."""
from __future__ import annotations

from datetime import date
from math import isfinite
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NPSPredictRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    operational_health: float = Field(ge=0, le=200.0)
    target_quality: float = Field(ge=0, le=100)
    actual_quality: float = Field(ge=0, le=100)
    target_competency: float = Field(ge=0, le=100)
    actual_competency: float = Field(ge=0, le=100)
    target_attendance: float = Field(ge=0, le=100)
    actual_attendance: float = Field(ge=0, le=100)
    target_release_rate: float = Field(ge=0, le=100)
    actual_release_rate: float = Field(ge=0, le=100)
    target_transfer_rate: float = Field(ge=0, le=100)
    actual_transfer_rate: float = Field(ge=0, le=100)
    total_calls_received: int = Field(gt=0)
    operational_intelligence_factor: float | None = None
    business_intelligence_factor: float | None = None
    member_intelligence_factor: float | None = None
    total_surveys: int | None = Field(default=None, gt=0)
    date: date | str | None = None

    @field_validator(
        "operational_health",
        "target_quality",
        "actual_quality",
        "target_competency",
        "actual_competency",
        "target_attendance",
        "actual_attendance",
        "target_release_rate",
        "actual_release_rate",
        "target_transfer_rate",
        "actual_transfer_rate",
        "operational_intelligence_factor",
        "business_intelligence_factor",
        "member_intelligence_factor",
        mode="before",
    )
    @classmethod
    def validate_finite(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("value must be numeric")
        if not isfinite(float(value)):
            raise ValueError("value must be finite")
        return value


def validate_release_transfer_sum(release, transfer):
    """Compatibility validation helper retained for API compatibility."""
    if release is None or transfer is None:
        return True
    try:
        return float(release) + float(transfer) <= 100.0
    except (TypeError, ValueError):
        return False


class HealthPredictRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class HealthBatchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class NPSBatchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class DashboardRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class SystemStatusRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
