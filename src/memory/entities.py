from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, MetaData, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.memory.config import SCHEMA
from src.schemas.query import MAX_CONVERSATION_ID_CHARS


class Base(DeclarativeBase):
    """Declarative base for the memory schema."""

    metadata = MetaData(schema=SCHEMA)


class ConversationMemoryTurn(Base):
    """One completed turn of one dialog."""

    __tablename__ = "conversation_memory_turns"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(MAX_CONVERSATION_ID_CHARS), nullable=False
    )
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index(
            "ix_conversation_memory_turns_conversation_id_id",
            "conversation_id",
            "id",
        ),
    )