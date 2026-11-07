from fastapi import APIRouter

from src.api.endpoints import chat, health, info, models
from src.api.endpoints.upload import upload_router
from src.api.endpoints import metrics as metrics_endpoint


api_router = APIRouter()

prefix = "/api/system/v1"

api_router.include_router(health.router, prefix=prefix, tags=["Service"])
api_router.include_router(info.router, prefix=prefix, tags=["Service"])
api_router.include_router(upload_router, prefix="/internal/knowledge", tags=["knowledge"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(
    metrics_endpoint.router, prefix=prefix, tags=["Service"]
)
api_router.include_router(models.router, prefix=prefix, tags=["Service"])