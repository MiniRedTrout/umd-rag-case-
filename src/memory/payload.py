"""Compact turn encoding for storage."""

from __future__ import annotations

import math
from typing import Any

from src.graph.memory import TurnMemory


PAYLOAD_SCHEMA_VERSION = 1


def to_payload(turn: TurnMemory) -> dict[str, Any]:
    """Render a turn as the JSON document stored in one row."""
    return {
        "question": _clip(str(turn.question or ""), 2000),
        "answer": _clip(str(turn.answer or ""), 8000),
        "created_at": _finite_float(turn.created_at),
    }


def from_payload(payload: Any) -> TurnMemory | None:
    """Rebuild a turn from a stored document."""
    if not isinstance(payload, dict):
        return None
    try:
        return TurnMemory(
            question=str(payload.get("question") or ""),
            answer=str(payload.get("answer") or ""),
            created_at=_finite_float(payload.get("created_at")),
        )
    except (TypeError, ValueError):
        return None


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 0:
        return ""
    if limit == 1:
        return "…"
    return value[: limit - 1].rstrip() + "…"


def _finite_float(value: Any) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return number if math.isfinite(number) else 0.0