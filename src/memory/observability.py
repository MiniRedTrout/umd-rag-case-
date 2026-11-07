from __future__ import annotations

from hashlib import sha256

from prometheus_client import Counter

MEMORY_FAILURES = Counter(
    "agent_memory_failures_total",
    "Conversation-memory operations that failed against PostgreSQL.",
    labelnames=("operation",),
)

MEMORY_DEGRADATIONS = Counter(
    "agent_memory_degradations_total",
    "Conversation-memory context narrowed between turns (version skips, "
    "provenance loss, caps, refusals).",
    labelnames=("event",),
)

_HANDLE_CHARS = 12


def conversation_handle(conversation_id: str) -> str:
    """Return a stable, non-reversible way to name one dialog in a log line.

    Args:
        conversation_id: str — The dialog identifier, which is a secret.

    Returns:
        str — A short digest, or ``"-"`` when there is no dialog.
    """
    if not conversation_id:
        return "-"
    return sha256(conversation_id.encode()).hexdigest()[:_HANDLE_CHARS]
