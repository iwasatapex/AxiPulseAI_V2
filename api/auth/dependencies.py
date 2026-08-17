from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from api.auth.jwt import decode_token


security = HTTPBearer()


def current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    try:
        payload = decode_token(credentials.credentials)
        username = payload.get("sub")
        role = payload.get("role", "user")
        if not username:
            raise JWTError("token subject missing")
        return {"username": username, "role": role}
    except JWTError as exc:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
        ) from exc


def require_admin(user=Depends(current_user)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
