from fastapi import APIRouter

from src.utils.metrics import snapshot

router = APIRouter()


@router.get("/metrics")
async def metrics():
    return snapshot()