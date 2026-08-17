"""API-key authentication with production fail-closed configuration."""
from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader


_ENVIRONMENT = os.getenv("AXIPULSE_ENV", "development").lower()
_configured_key = os.getenv("AXIPULSE_API_KEY")

if _configured_key:
    API_KEY = _configured_key
elif _ENVIRONMENT in {"production", "prod"}:
    raise RuntimeError("AXIPULSE_API_KEY must be configured in production")
else:
    API_KEY = "dev-key-change-me"


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Security(api_key_header)) -> bool:
    """Validate the configured API key using constant-time comparison."""
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_api_key",
                "message": "Valid X-API-Key required",
            },
        )
    return True
