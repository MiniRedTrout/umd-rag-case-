"""Public surface of the SSE event schema.

Importing from ``src.schemas.events`` is the intended way to reach the
event types and factories. Everything defined in ``sse`` and ``errors``
is re-exported here, so callers never need the internal module path.
"""

from __future__ import annotations

from src.schemas.events.errors import (
    classify_exception,
    is_recoverable,
    public_exception_message,
)
from src.schemas.events.sse import (
    DoneEvent,
    ErrorDetails,
    ErrorEvent,
    ErrorType,
    EventWriter,
    FinishReason,
    MessageDeltaEvent,
    MessageEndEvent,
    MessageStartEvent,
    Reference,
    ResponseStartEvent,
    SSEEventPayload,
    SSEEventType,
    StatusEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolEndEvent,
    ToolKind,
    ToolStartEvent,
    ToolStatus,
    UsagePayload,
    done_event,
    error_event,
    message_delta_event,
    message_end_event,
    message_start_event,
    response_start_event,
    status_event,
    thinking_delta_event,
    thinking_end_event,
    thinking_start_event,
    tool_end_event,
    tool_start_event,
    validate_sse_event,
)

__all__ = [
    # --- types ---
    "ErrorType",
    "EventWriter",
    "FinishReason",
    "SSEEventPayload",
    "SSEEventType",
    "ToolKind",
    "ToolStatus",
    # --- models ---
    "DoneEvent",
    "ErrorDetails",
    "ErrorEvent",
    "MessageDeltaEvent",
    "MessageEndEvent",
    "MessageStartEvent",
    "Reference",
    "ResponseStartEvent",
    "StatusEvent",
    "ThinkingDeltaEvent",
    "ThinkingEndEvent",
    "ThinkingStartEvent",
    "ToolEndEvent",
    "ToolStartEvent",
    "UsagePayload",
    # --- factories ---
    "done_event",
    "error_event",
    "message_delta_event",
    "message_end_event",
    "message_start_event",
    "response_start_event",
    "status_event",
    "thinking_delta_event",
    "thinking_end_event",
    "thinking_start_event",
    "tool_end_event",
    "tool_start_event",
    # --- validation / errors ---
    "classify_exception",
    "is_recoverable",
    "public_exception_message",
    "validate_sse_event",
]