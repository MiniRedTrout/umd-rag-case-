from __future__ import annotations

import time
from typing import Any

from langchain_core.runnables import RunnableConfig

from src.core.config.agent_config import get_agent_config
from src.graph.memory import recall, render_session_memory
from src.graph.state import ChatGraphState
from src.schemas.usage import zero_usage
from src.utils.logger import logger


def clamp_history(
    history: list[Any], limits, *, max_total_chars: int = 8000
) -> list[Any]:

    trimmed = history[-limits.max_messages :] if limits.max_messages > 0 else history

    clamped = []
    for message in trimmed:
        content = message.get("content", "") if isinstance(message, dict) else ""
        if isinstance(content, str) and len(content) > limits.max_message_chars:
            content = content[-limits.max_message_chars :]
            message = {**message, "content": content}
        clamped.append(message)

    total = sum(len(m.get("content", "")) for m in clamped if isinstance(m, dict))
    while total > max_total_chars and len(clamped) > 1:
        oldest = clamped.pop(0)
        total -= len(oldest.get("content", "")) if isinstance(oldest, dict) else 0

    return clamped


def make_normalize_node():
    async def normalize_node(
        state: ChatGraphState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        agent_config = get_agent_config()
        question = (state.get("question") or "").strip()
        memory = await recall(str(state.get("conversation_id") or "").strip())
        history = clamp_history(state.get("chat_history") or [], agent_config.history)
        return {
            "chat_history": history,
            "session_memory_block": render_session_memory(
                memory, agent_config.memory
            ),
            "total_usage": state.get("total_usage") or zero_usage(),
            "peak_usage": state.get("peak_usage") or zero_usage(),
        }

    return normalize_node