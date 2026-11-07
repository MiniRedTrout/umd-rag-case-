from __future__ import annotations

from typing import Any, TypedDict

from src.schemas.events import FinishReason


class ChatGraphState(TypedDict, total=False):
    question: str
    chat_history: list[dict[str, Any]]
    model_name: str
    llm_params: dict[str, Any]
    conversation_id: str
    session_memory_block: str
    context: list[dict[str, Any]]
    references: list[dict[str, Any]]
    answer_text: str
    finish_reason: FinishReason
    total_usage: dict[str, int]
    peak_usage: dict[str, int]
    request_kind: str 
    smalltalk_question: str
    smalltalk_answer: str