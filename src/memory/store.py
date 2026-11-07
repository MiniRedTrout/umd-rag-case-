from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime

from loguru import logger
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from src.memory.engine import MemoryDatabaseUnavailable, MemoryEngine
from src.memory.entities import ConversationMemoryTurn
from src.schemas.query import MAX_CONVERSATION_ID_CHARS


@dataclass(frozen=True, slots=True)
class StoredTurn:
    payload: dict | None
    created_at: datetime | None = None


class ConversationTurnStore:
    """Reads and writes rows of ``conversation_memory_turns``."""

    def __init__(self, engine: MemoryEngine) -> None:
        self._engine = engine

    async def read_recent(
        self,
        conversation_id: str,
        *,
        limit: int,
    ) -> list[StoredTurn]:
        if not self.accepts(conversation_id) or limit <= 0:
            return []
        statement = (
            select(
                ConversationMemoryTurn.payload,
                ConversationMemoryTurn.created_at,
            )
            .where(ConversationMemoryTurn.conversation_id == conversation_id)
            .order_by(ConversationMemoryTurn.id.desc())
            .limit(limit)
        )
        try:
            async with self._engine.sessionmaker() as session:
                rows = (await session.execute(statement)).all()
        except SQLAlchemyError as error:
            logger.error(f"[memory] read failed: {type(error).__name__}")
            raise MemoryDatabaseUnavailable(
                f"Could not read conversation memory: {type(error).__name__}"
            ) from None
        return [StoredTurn(row[0], row[1]) for row in rows]

    async def append(
        self, conversation_id: str, *, payload: dict
    ) -> None:
        if not self.accepts(conversation_id):
            return
        entity = ConversationMemoryTurn(
            conversation_id=conversation_id,
            payload=payload,
        )
        try:
            async with self._engine.sessionmaker() as session:
                session.add(entity)
                await session.commit()
        except SQLAlchemyError as error:
            logger.error(f"[memory] write failed: {type(error).__name__}")
            raise MemoryDatabaseUnavailable(
                f"Could not save conversation memory: {type(error).__name__}"
            ) from None

    @staticmethod
    def accepts(conversation_id: str) -> bool:
        return (
            bool(conversation_id) and len(conversation_id) <= MAX_CONVERSATION_ID_CHARS
        )