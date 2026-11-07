"""Persist every chat turn to ``memory_shared.chat_history``.

The ``conversation_memory_turns`` table answers "what did this dialog
discuss", and keeps only the last few turns. This one answers "what did
the service ever say", and keeps everything: model, finish reason, tokens.
Used for analytics and exports, never read back into the prompt.
"""

from __future__ import annotations

from loguru import logger
from sqlalchemy import text

from src.memory.engine import MemoryDatabaseUnavailable, MemoryEngine


_engine: MemoryEngine | None = None


def set_engine(engine: MemoryEngine) -> None:
    global _engine
    _engine = engine


def reset_engine() -> None:
    global _engine
    _engine = None


async def persist_chat_history(
    *,
    conversation_id: str,
    question: str,
    answer: str,
    model_name: str,
    finish_reason: str,
    usage: dict[str, int],
) -> None:
    """Insert one row. Failures are logged, never raised — analytics must
    not break the request path."""
    engine = _engine
    if engine is None:
        return

    try:
        async with engine.sessionmaker() as session:
            await session.execute(
                text(
                    "INSERT INTO memory_shared.chat_history "
                    "(conversation_id, question, answer, model_name, "
                    " finish_reason, input_tokens, output_tokens, total_tokens) "
                    "VALUES (:cid, :q, :a, :m, :fr, :it, :ot, :tt)"
                ),
                {
                    "cid": conversation_id,
                    "q": question,
                    "a": answer,
                    "m": model_name,
                    "fr": finish_reason,
                    "it": int(usage.get("input_tokens") or 0),
                    "ot": int(usage.get("output_tokens") or 0),
                    "tt": int(usage.get("total_tokens") or 0),
                },
            )
            await session.commit()
    except MemoryDatabaseUnavailable:
        logger.exception("[chat_history] database unavailable")
    except Exception:
        logger.exception("[chat_history] insert failed")