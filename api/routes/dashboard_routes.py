from datetime import datetime, timezone
from fastapi import APIRouter, Depends

from api.security.api_key import verify_api_key

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/full")
async def get_dashboard():
    return {
        "status": "success",
        "message": "Dashboard ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
