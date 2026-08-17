"""Legacy standalone compatibility API.

The canonical production API remains ``api.main`` and exposes only V3.
This module is retained for compatibility with existing integrations.
"""
from datetime import datetime, timezone

from fastapi import Depends, FastAPI

from api.security.api_key import verify_api_key
from api.services.adie_v3_service import ADIEV3Service


app = FastAPI(title="AxiPulseAI Legacy Compatibility API")
service = ADIEV3Service()


@app.get("/")
async def root():
    return {
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/adie/decision", dependencies=[Depends(verify_api_key)])
def adie_decision(payload: dict):
    oh = float(payload.get("operations_health", 82))
    observations = [
        min(1.0, max(0.0, oh / 100.0)),
        min(1.0, max(0.0, float(payload.get("quality", 85)) / 100.0)),
        min(1.0, max(0.0, float(payload.get("competency", 88)) / 100.0)),
    ]
    result = service.analyze_scenarios(
        scenarios=[{"name": "current_state", "expected": oh / 100.0}],
        observations=observations,
        baseline=oh / 100.0,
        samples=5000,
    )
    package = result
    return {
        "recommendation": package.recommendation,
        "risk": package.risk,
        "probability": package.probability,
        "confidence": package.confidence,
        "expected": package.expected,
        "downside": package.downside,
        "upside": package.upside,
        "status": "success",
    }
