"""Response contracts for predictor endpoints."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class NPSPredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    data: dict[str, Any]
    metadata: dict[str, Any]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class NPSResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class DashboardResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class HealthPredictResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class HealthBatchResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class NPSBatchResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
