import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from api.models.requests import NPSPredictRequest
from api.models.responses import NPSPredictResponse
from api.security.api_key import verify_api_key
from api.services.nps_service import NPSService, NPSServiceUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key)])

_service = None


def _get_service():
    """Return the shared NPSService instance (created once)."""
    global _service
    if _service is None:
        _service = NPSService()
    return _service


@router.post("/predict", response_model=NPSPredictResponse)
async def predict_nps(request: NPSPredictRequest):
    service = _get_service()
    try:
        result = service.predict(request.model_dump())
    except NPSServiceUnavailableError as e:
        logger.error("NPS model unavailable: %s", e)
        raise HTTPException(status_code=503, detail="NPS model unavailable")
    except Exception as e:
        logger.error("NPS prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="NPS prediction failed")

    return {
        "status": "success",
        "data": result,
        "metadata": {
            "engine": "nps",
            "version": "2.1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


@router.get("/status")
async def nps_status():
    service = _get_service()
    return {
        "engine": "NPS Predictor",
        "status": "ready" if service.is_loaded() else "model_unavailable",
        "version": "2.1.0",
    }

