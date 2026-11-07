from __future__ import annotations

import re
import uuid
from contextvars import ContextVar, Token

#: The header callers may set, and the header every LLM request carries.
REQUEST_ID_HEADER = "X-Request-ID"

#: Advertises that this build forwards ``X-Request-ID`` to the LLM provider,
#: so a benchmark preflight can skip builds that would correlate nothing.
CORRELATION_CAPABILITY_HEADER = "X-Request-Correlation"
CORRELATION_CAPABILITY_VALUE = "1"

#: Safe header material only: plain ASCII identifier characters. Anything
#: else — spaces, separators, control bytes, header-injection newlines — is
#: refused, and a random id is minted instead of echoing untrusted input.
_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")

REQUEST_ID_CONTEXT_VAR: ContextVar[str | None] = ContextVar("request_id", default=None)


def resolve_request_id(raw: str | None) -> str:
    """Validate a caller-supplied id, or mint one.

    Returns the caller's id when it is safe ASCII (``[A-Za-z0-9._:-]``, 1..128
    characters); otherwise a fresh random UUID.
    """
    if raw is not None and _SAFE_REQUEST_ID.fullmatch(raw):
        return raw
    return str(uuid.uuid4())


def get_request_id() -> str | None:
    """Return the current request's correlation id, or None outside one.

    Returns:
        str | None — The id bound to the current context.
    """
    return REQUEST_ID_CONTEXT_VAR.get()


def set_request_id(request_id: str) -> Token[str | None]:
    """Publish the id for the current context.

    Returns a token; pass it to :func:`reset_request_id` to restore the
    previous value.
    """
    return REQUEST_ID_CONTEXT_VAR.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    """Restore the value the token was taken against.

    Args:
        token: Token[str | None] — The token returned by :func:`set_request_id`.
    """
    REQUEST_ID_CONTEXT_VAR.reset(token)
