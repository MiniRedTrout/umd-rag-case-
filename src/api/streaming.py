from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

from src.schemas.events import (
    SSEEventPayload,
    SSEEventType,
    classify_exception,
    done_event,
    error_event,
    message_end_event,
    public_exception_message,
    response_start_event,
)
from src.schemas.usage import zero_usage
from src.utils.logger import logger

#: Shown when a stream dies before the producer could say anything useful.
STREAM_ERROR_TEXT = "Не удалось сформировать ответ. Попробуйте повторить запрос."

_NON_TERMINAL_FINISH = "tool_calls"

EventProducer = Callable[[], AsyncIterator[SSEEventPayload]]


async def run_event_stream(producer: EventProducer) -> AsyncIterator[SSEEventPayload]:
    """Run the agent graph and emit its events as SSE frames."""
    yield response_start_event()

    response_terminated = False
    upstream = producer()
    try:
        async for event in upstream:
            event_type = event.type
            if event_type in (
                SSEEventType.RESPONSE_START.value,
                SSEEventType.DONE.value,
            ):
                logger.error(
                    f"Producer emitted {event_type!r}, which the runner owns; dropping"
                )
                continue
            if event_type == SSEEventType.MESSAGE_END.value:
                response_terminated = event.finish_reason != _NON_TERMINAL_FINISH
            yield event
    except (asyncio.CancelledError, GeneratorExit):
        raise
    except Exception as exc:
        logger.exception("Event producer failed")
        if not response_terminated:
            error_type, provider_status = classify_exception(exc)
            yield error_event(
                error_type=error_type,
                message=public_exception_message(exc),
                recoverable=False,
                provider_status=provider_status,
            )
            yield message_end_event(
                text=STREAM_ERROR_TEXT, finish_reason="error", usage=zero_usage()
            )
            response_terminated = True
    else:
        if not response_terminated:
            logger.warning("Producer finished without a terminal message_end")
            yield error_event(
                error_type="api_error",
                message=STREAM_ERROR_TEXT,
                recoverable=False,
            )
            yield message_end_event(
                text=STREAM_ERROR_TEXT, finish_reason="error", usage=zero_usage()
            )
    finally:
        await _safe_aclose(upstream)

    yield done_event()


async def _safe_aclose(agen: object) -> None:
    aclose = getattr(agen, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except (asyncio.CancelledError, GeneratorExit):
        raise
    except Exception:
        logger.exception("Failed to close the event producer")