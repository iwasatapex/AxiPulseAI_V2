from datetime import datetime, timezone
from fastapi import APIRouter, Depends

from api.security.api_key import verify_api_key

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/predict")
async def predict_health():
    return {
        "status": "success",
        "message": "Health predictor ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/status")
async def health_status():
    return {
        "engine": "Operational Health Predictor",
        "status": "ready",
        "version": "10.10",
    }
