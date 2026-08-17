"""Authentication endpoints."""
from __future__ import annotations

import hashlib
import hmac
import os

from fastapi import APIRouter, HTTPException

from api.auth.jwt import create_token
from api.auth.models import TokenRequest, TokenResponse


router = APIRouter()

_ENVIRONMENT = os.getenv("AXIPULSE_ENV", "development").lower()
_ADMIN_USERNAME = os.getenv("AXIPULSE_ADMIN_USERNAME")
_ADMIN_PASSWORD = os.getenv("AXIPULSE_ADMIN_PASSWORD")
_ADMIN_PASSWORD_HASH = os.getenv("AXIPULSE_ADMIN_PASSWORD_SHA256")

if not _ADMIN_USERNAME and _ENVIRONMENT not in {"production", "prod"}:
    _ADMIN_USERNAME = "test"
if not _ADMIN_PASSWORD and not _ADMIN_PASSWORD_HASH and _ENVIRONMENT not in {"production", "prod"}:
    _ADMIN_PASSWORD = "test"


def _password_matches(password: str) -> bool:
    if _ADMIN_PASSWORD_HASH:
        digest = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return hmac.compare_digest(digest, _ADMIN_PASSWORD_HASH)
    if _ADMIN_PASSWORD is not None:
        return hmac.compare_digest(password, _ADMIN_PASSWORD)
    return False


@router.post("/token", response_model=TokenResponse)
def login(request: TokenRequest):
    if not _ADMIN_USERNAME or not _password_matches(request.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not hmac.compare_digest(request.username, _ADMIN_USERNAME):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(request.username, "admin")
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "username": request.username,
            "role": "admin",
            "active": True,
        },
    }
