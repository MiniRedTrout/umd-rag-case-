from __future__ import annotations

from collections.abc import AsyncIterator
from http import HTTPStatus

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.sse import SSE_HEADERS, sse_frame
from src.api.streaming import run_event_stream
from src.graph.runner import produce_agent_events
from src.schemas import QueryChat
from src.utils.logger import logger

router = APIRouter()


@router.post("/stream", status_code=200)
async def stream_answer(query_extended: QueryChat):
    """Stream structured SSE events; reject an empty conversation with HTTP 400."""
    if not query_extended.messages:
        logger.error("No messages in request")
        return JSONResponse(status_code=400, content="No messages in request")

    logger.info(
        f"Received query: messages={len(query_extended.messages)}, "
        f"model={query_extended.model_name}"
    )

    async def frames() -> AsyncIterator[bytes]:
        async for event in run_event_stream(
            lambda: produce_agent_events(query_extended)
        ):
            yield sse_frame(event)

    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        frames(),
        media_type="text/event-stream; charset=utf-8",
        headers=SSE_HEADERS,
    )