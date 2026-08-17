from datetime import datetime, timezone
from fastapi import APIRouter, Depends

from api.security.api_key import verify_api_key

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get("/status")
async def system_status():
    return {
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
