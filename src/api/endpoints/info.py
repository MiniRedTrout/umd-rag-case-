from fastapi import APIRouter

from src.schemas.info import InfoResponse
from src.structure import structure

router = APIRouter()


@router.get("/info")
async def info() -> InfoResponse:
    return await structure.system_info_service.get_info()