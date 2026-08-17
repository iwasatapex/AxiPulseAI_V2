"""AxiPulseAI FastAPI application."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.auth import routes as auth_routes
from api.database import models  # noqa: F401
from api.middleware.metrics import MetricsMiddleware
from api.middleware.rate_limit import check_rate_limit
from api.middleware.request_id import RequestIDMiddleware
from api.middleware.security_headers import SecurityHeadersMiddleware
from api.routes import (
    adie_v3_routes,
    dashboard_routes,
    health_routes,
    history_routes,
    metrics_routes,
    nps_routes,
    system_routes,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("AxiPulseAI")

API_VERSION = "1.1.0"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply the existing limiter to API requests without changing route semantics."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/") and not check_rate_limit(
            request.client.host if request.client else "unknown"
        ):
            return JSONResponse(
                status_code=429,
                content={"status": "error", "message": "Rate limit exceeded"},
            )
        return await call_next(request)


app = FastAPI(
    title="AxiPulseAI API",
    description="Complete API for AxiPulseAI - Predictors, Simulator, ADIE Decision Intelligence",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(MetricsMiddleware)
app.add_middleware(RateLimitMiddleware)

allowed_origins = os.getenv(
    "AXIPULSE_CORS_ORIGINS",
    "http://localhost,http://localhost:3000",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def initialize_database() -> None:
    """Initialize schema at application startup instead of import time."""
    from api.database.connection import Base, engine
    Base.metadata.create_all(bind=engine)

app.include_router(health_routes.router, prefix="/api/v1/health", tags=["Health"])
app.include_router(nps_routes.router, prefix="/api/v1/nps", tags=["NPS"])
app.include_router(dashboard_routes.router, prefix="/api/v1/dashboard", tags=["Dashboard"])
app.include_router(system_routes.router, prefix="/api/v1/system", tags=["System"])
app.include_router(adie_v3_routes.router, prefix="/api/v1/adie/v3", tags=["ADIE V3"])
app.include_router(auth_routes.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(history_routes.router, prefix="/api/v1/history", tags=["History"])
app.include_router(metrics_routes.router, prefix="/metrics", tags=["Metrics"])

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": str(exc.detail),
            "path": request.url.path,
        },
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "status": "validation_error",
            "details": exc.errors(),
            "path": request.url.path,
        },
    )

@app.get("/")
async def root():
    return {
        "name": "AxiPulseAI API",
        "version": API_VERSION,
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
