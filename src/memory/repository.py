from __future__ import annotations

from src.graph.memory import TurnMemory
from src.memory.engine import MemoryEngine
from src.memory.payload import from_payload, to_payload
from src.memory.store import ConversationTurnStore


class PostgresSessionMemoryRepository:
    """Compacted turns of a dialog, kept in PostgreSQL."""

    def __init__(self, engine: MemoryEngine) -> None:
        self._store = ConversationTurnStore(engine)

    async def get_recent_turns(
        self, conversation_id: str, *, limit: int
    ) -> list[TurnMemory]:
        rows = await self._store.read_recent(conversation_id, limit=limit)
        turns: list[TurnMemory] = []
        for row in rows:
            turn = from_payload(row.payload)
            if turn is None:
                continue
            if row.created_at is not None:
                turn.created_at = row.created_at.timestamp()
            turns.append(turn)
        turns.reverse()
        return turns

    async def append_turn(self, conversation_id: str, turn: TurnMemory) -> None:
        await self._store.append(conversation_id, payload=to_payload(turn))