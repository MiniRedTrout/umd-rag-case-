from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from src.schemas.llm_parameters import LLMGenerationParams

MIN_CONVERSATION_ID_CHARS = 8
MAX_CONVERSATION_ID_CHARS = 128


class MessageRole(StrEnum):
    """Role of the message sender"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """Message in a conversation"""
    role: MessageRole = Field(description="Role of the message sender")
    content: str = Field(description="Content of the message")


class LLMParams(LLMGenerationParams):
    model_name: str = Field(description="Model name must be set")


class QueryChat(LLMParams):
    """Represent a chat query."""

    messages: list[Message] = Field(
        default=[], description="Array of previous messages in the conversation"
    )
    conversation_id: str | None = Field(
        default=None,
        min_length=MIN_CONVERSATION_ID_CHARS,
        max_length=MAX_CONVERSATION_ID_CHARS,
        description=(
            "Stable per-dialog id. When present, previous turns of this dialog "
            "are replayed into the prompt. Absent turns memory off."
        ),
    )

    def get_messages(self, inputs: dict[str, Any]) -> list[dict]:
        return [message.model_dump() for message in self.messages]