from fastapi import APIRouter, Request, Response, status

from src.memory.engine import MemoryDatabaseUnavailable, MemoryEngine
from src.structure import structure
from src.utils.logger import logger


router = APIRouter()


@router.get("/health")
async def health_check():
    """Checks the health status of the application.

    Returns:
        A dictionary with the status message indicating the application's health.
    """
    return await structure.health_check.check()


@router.get("/readiness")
async def readiness(request: Request, response: Response) -> dict[str, str]:
    """Say whether this pod should be handed traffic.

    Distinct from ``/health``, which reports on the components this process
    talks to. This answers a narrower question the orchestrator asks over and
    over: is the memory schema in front of *this build* the one it was written
    against? A TCP probe cannot ask it, and the failure it misses is specific —
    a pod rolled out ahead of its migration, or rolled back behind one, accepts
    connections perfectly well and then fails every request that touches
    memory.

    ``MemoryEngine.verify`` is the same check startup runs, so readiness and
    startup can never disagree about what "ready" means.

    Returns:
        dict[str, str] — ``{"state": ...}``, with 503 when it is not ready.
    """
    engine: MemoryEngine | None = getattr(request.app.state, "memory_engine", None)
    if engine is None:
        return {"state": "READY", "detail": "conversation memory is not in use"}
    try:
        await engine.verify()
    except MemoryDatabaseUnavailable as error:
        logger.warning(f"[readiness] not ready: {error}")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"state": "NOT_READY", "detail": str(error)}
    return {"state": "READY", "detail": "memory schema is at the expected revision"}