from __future__ import annotations

import json

from src.schemas.events import SSEEventPayload


def sse_frame(payload: SSEEventPayload) -> bytes:
    """Render an event as an SSE frame.

    Args:
        payload: SSEEventPayload — Event payload to frame.

    Returns:
        bytes — Encoded SSE frame.
    """
    data = json.dumps(payload.model_dump(exclude_none=True), ensure_ascii=False)
    return f"event: {payload.type}\ndata: {data}\n\n".encode()


#: Headers that keep intermediaries from buffering or caching the stream.
SSE_HEADERS = {
    "Cache-Control": "no-store",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
