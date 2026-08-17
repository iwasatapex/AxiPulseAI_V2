from pydantic import BaseModel
from typing import Optional


class User(BaseModel):
    username: str
    role: str = "user"
    active: bool = True


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: User
