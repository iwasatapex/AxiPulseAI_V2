"""ADIE V3 response contract."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ADIEDecisionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    data: Any
    metadata: dict[str, Any]
