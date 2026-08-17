"""JWT creation and validation with fail-closed production configuration."""
from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone

from jose import jwt


ALGORITHM = "HS256"
_ENVIRONMENT = os.getenv("AXIPULSE_ENV", "development").lower()
_configured_secret = os.getenv("AXIPULSE_JWT_SECRET")

if _configured_secret:
    SECRET_KEY = _configured_secret
elif _ENVIRONMENT in {"production", "prod"}:
    raise RuntimeError("AXIPULSE_JWT_SECRET must be configured in production")
else:
    SECRET_KEY = secrets.token_urlsafe(48)


def create_token(username: str, role: str = "user") -> str:
    """Create an eight-hour bearer token for an authenticated user."""
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=8),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a bearer token."""
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
