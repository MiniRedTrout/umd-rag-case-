from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class SSEEventType(StrEnum):
    """Define public SSE wire-protocol event names."""

    RESPONSE_START = "response_start"
    THINKING_START = "thinking_start"
    THINKING_DELTA = "thinking_delta"
    THINKING_END = "thinking_end"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    MESSAGE_START = "message_start"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_END = "message_end"
    ERROR = "error"
    DONE = "done"


FinishReason: TypeAlias = Literal[
    "stop", "length", "tool_calls", "content_filter", "error"
]

ToolStatus: TypeAlias = Literal["success", "empty", "error"]


class StatusEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["status"] = "status"
    text: str


class UsagePayload(BaseModel):
    """Represent token usage for tool and message completion events."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0, strict=True)
    output_tokens: int = Field(ge=0, strict=True)
    cached_input_tokens: int = Field(ge=0, strict=True)
    reasoning_tokens: int = Field(ge=0, strict=True)
    total_tokens: int = Field(default=0, ge=0, strict=True)
    peak_input_tokens: int = Field(default=0, ge=0, strict=True)
    peak_output_tokens: int = Field(default=0, ge=0, strict=True)
    peak_reasoning_tokens: int = Field(default=0, ge=0, strict=True)
    peak_total_tokens: int = Field(default=0, ge=0, strict=True)


class Reference(BaseModel):
    """Represent a cited source."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str
    url: str = ""
    source_type: str = ""


class ResponseStartEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["response_start"] = "response_start"


class ThinkingStartEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["thinking_start"] = "thinking_start"
    thinking_id: str = Field(min_length=1)
    visibility: Literal["internal", "user"]


class ThinkingDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["thinking_delta"] = "thinking_delta"
    thinking_id: str = Field(min_length=1)
    text: str


class ThinkingEndEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["thinking_end"] = "thinking_end"
    thinking_id: str = Field(min_length=1)


ToolKind: TypeAlias = Literal["search", "read", "skill", "finish", "clarify", "other"]


class ToolStartEvent(BaseModel):
    """Represent the start of a tool call."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_start"] = "tool_start"
    tool_call_id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    input: dict[str, Any]
    kind: ToolKind | None = None
    label: str | None = Field(default=None, min_length=1)


class ToolEndEvent(BaseModel):
    """Represent completion of a tool call."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_end"] = "tool_end"
    tool_call_id: str = Field(min_length=1)
    status: ToolStatus
    result: str
    references: list[Reference] = Field(default_factory=list)
    usage: UsagePayload


class MessageStartEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["message_start"] = "message_start"


class MessageDeltaEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["message_delta"] = "message_delta"
    text: str


class MessageEndEvent(BaseModel):
    """Represent a final normalized message and its token total."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["message_end"] = "message_end"
    text: str
    finish_reason: FinishReason
    references: list[Reference]
    usage: UsagePayload


ErrorType: TypeAlias = Literal[
    "service_unavailable_error",
    "overloaded_error",
    "rate_limit_error",
    "timeout_error",
    "authentication_error",
    "permission_error",
    "not_found_error",
    "request_too_large",
    "invalid_request_error",
    "content_filter_error",
    "api_error",
]


class ErrorDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ErrorType
    message: str
    provider_status: int | None = Field(default=None, ge=0)
    retry_after: int | None = Field(default=None, ge=0)


class ErrorEvent(BaseModel):
    """Represent an error inside the stream."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["error"] = "error"
    recoverable: bool
    error: ErrorDetails
    tool_call_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _reject_null_tool_call_id(cls, data: Any) -> Any:
        if (
            isinstance(data, dict)
            and "tool_call_id" in data
            and data.get("tool_call_id") is None
        ):
            raise ValueError("tool_call_id must be absent, not null")
        return data


class DoneEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["done"] = "done"


SSEEventPayload: TypeAlias = (
    ResponseStartEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolStartEvent
    | ToolEndEvent
    | MessageStartEvent
    | MessageDeltaEvent
    | MessageEndEvent
    | ErrorEvent
    | DoneEvent
    | StatusEvent
)

EventWriter: TypeAlias = Callable[[SSEEventPayload], None]

_SSE_EVENT_ADAPTER: TypeAdapter[SSEEventPayload] = TypeAdapter(SSEEventPayload)


def validate_sse_event(payload: dict[str, Any]) -> SSEEventPayload:
    """Validate a payload against the wire contract.

    Args:
        payload: dict[str, Any] — Event payload to validate.

    Returns:
        SSEEventPayload — Validated event payload.
    """
    return _SSE_EVENT_ADAPTER.validate_python(payload)


def response_start_event() -> SSEEventPayload:
    return ResponseStartEvent()


def thinking_start_event(
    *, thinking_id: str, visibility: Literal["internal", "user"]
) -> SSEEventPayload:
    return ThinkingStartEvent(thinking_id=thinking_id, visibility=visibility)


def thinking_delta_event(*, thinking_id: str, text: str) -> SSEEventPayload:
    return ThinkingDeltaEvent(thinking_id=thinking_id, text=text)


def thinking_end_event(*, thinking_id: str) -> SSEEventPayload:
    return ThinkingEndEvent(thinking_id=thinking_id)


def tool_start_event(
    *,
    tool_call_id: str,
    tool: str,
    input: dict[str, Any],
    kind: ToolKind | None = None,
    label: str | None = None,
) -> SSEEventPayload:
    return ToolStartEvent(
        tool_call_id=tool_call_id,
        tool=tool,
        input=input,
        kind=kind,
        label=label or None,
    )


def tool_end_event(
    *,
    tool_call_id: str,
    status: ToolStatus,
    result: str,
    usage: dict[str, int],
    references: list[dict[str, Any]] | None = None,
) -> SSEEventPayload:
    return ToolEndEvent(
        tool_call_id=tool_call_id,
        status=status,
        result=result,
        references=[Reference.model_validate(r) for r in references or []],
        usage=UsagePayload.model_validate(usage),
    )


def message_start_event() -> SSEEventPayload:
    return MessageStartEvent()


def message_delta_event(*, text: str) -> SSEEventPayload:
    return MessageDeltaEvent(text=text)


def message_end_event(
    *,
    text: str,
    finish_reason: FinishReason,
    usage: dict[str, int],
    references: list[dict[str, Any]] | None = None,
) -> SSEEventPayload:
    return MessageEndEvent(
        text=text,
        finish_reason=finish_reason,
        references=[Reference.model_validate(r) for r in references or []],
        usage=UsagePayload.model_validate(usage),
    )


def error_event(
    *,
    error_type: ErrorType,
    message: str,
    recoverable: bool,
    provider_status: int | None = None,
    retry_after: int | None = None,
    tool_call_id: str | None = None,
) -> SSEEventPayload:
    payload: dict[str, Any] = {
        "recoverable": recoverable,
        "error": ErrorDetails(
            type=error_type,
            message=message,
            provider_status=provider_status,
            retry_after=retry_after,
        ),
    }
    if tool_call_id is not None:
        payload["tool_call_id"] = tool_call_id
    return ErrorEvent.model_validate(payload)


def done_event() -> SSEEventPayload:
    return DoneEvent()


def status_event(*, text: str) -> SSEEventPayload:
    return StatusEvent(text=text)
