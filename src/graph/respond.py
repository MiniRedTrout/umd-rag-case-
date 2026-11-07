from __future__ import annotations

from typing import Any

from src.schemas.events import (
    message_delta_event,
    message_end_event,
    message_start_event,
)
from src.schemas.usage import emitted_usage, zero_usage
from src.utils.logger import logger


FALLBACK = "Здравствуйте! Задайте вопрос по документам, и я постараюсь помочь."


async def respond_node(state, config=None) -> dict[str, Any]:
    writer = _writer()
    text = (state.get("smalltalk_answer") or "").strip() or FALLBACK
    logger.info(f"[respond] smalltalk: {text!r}")
    writer(message_start_event())
    writer(message_delta_event(text=text))
    writer(
        message_end_event(
            text=text,
            finish_reason="stop",
            usage=emitted_usage(zero_usage(), zero_usage()),
        )
    )
    return {"answer_text": text, "finish_reason": "stop", "references": []}


def _writer():
    try:
        from langgraph.config import get_stream_writer
        return get_stream_writer()
    except RuntimeError:
        return lambda _payload: None