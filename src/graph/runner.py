from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langfuse import get_client, propagate_attributes  # <-- LANGFUSE

from src.graph.graph import build_chat_graph
from src.graph.prompt import prompt_version_hash  
from src.schemas.events import SSEEventPayload
from src.schemas.query import QueryChat
from src.utils.logger import logger

_graph: Any = None


def get_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_chat_graph()
        logger.info("Graph compiled")
    return _graph


def initial_state(query: QueryChat) -> dict[str, Any]:
    history = [message.model_dump() for message in query.messages[:-1]]
    return {
        "question": query.messages[-1].content.rstrip(),
        "chat_history": history,
        "conversation_id": (query.conversation_id or "").strip(),
        "model_name": query.model_name,
        "llm_params": query.model_dump(
            include={"temperature", "top_p", "max_tokens", "repetition_penalty"}
        ),
    }


async def produce_agent_events(query: QueryChat) -> AsyncIterator[SSEEventPayload]:
    """Yield events emitted while processing one request."""
    graph = get_graph()
    state0 = initial_state(query)
    final_state: dict[str, Any] = {}

    prompt_hash = prompt_version_hash()
    session_id = (query.conversation_id or "").strip() or "anonymous"

    try:

        with propagate_attributes(
            trace_name=f"chat_{query.model_name}",
            session_id=session_id,
            version=prompt_hash,
        ):
            async for mode, event in graph.astream(
                state0, stream_mode=["custom", "values"]
            ):
                if mode == "values":
                    final_state = event
                    continue
                yield event
    except asyncio.CancelledError:
        logger.warning("[runner] request cancelled")
        raise