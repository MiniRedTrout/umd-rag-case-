"""Classify transport errors raised by chat and embedding backends.

Shared by the retry decorators and the embedding size fallback: both need to
tell a transient fault apart from a verdict on the request itself.
"""

#: The payload exceeds what the backend accepts. Resending it unchanged can
#: never succeed, so the request must shrink or fail.
PAYLOAD_TOO_LARGE = 413
#: Censorship is a verdict on the content, not a transient fault.
UNAVAILABLE_FOR_LEGAL_REASONS = 451

#: Backends that answer 400 instead of 413 say so in the message body.
_TOO_LARGE_MARKERS = ("request_too_large", "maximum context length")


def status_code(exc: BaseException) -> int | None:
    """Read the HTTP status carried by an exception.

    Args:
        exc: BaseException — Exception raised by a backend call.

    Returns:
        int | None — Status code, when the exception carries one.
    """
    direct = getattr(exc, "status_code", None)
    if isinstance(direct, int):
        return direct
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def is_content_filtered(exc: BaseException) -> bool:
    """Determine whether the gateway censor rejected a request.

    Args:
        exc: BaseException — Exception raised by a backend call.

    Returns:
        bool — Whether the content itself was blocked; retrying cannot help.
    """
    return status_code(exc) == UNAVAILABLE_FOR_LEGAL_REASONS


def is_request_too_large(exc: BaseException) -> bool:
    """Determine whether a backend rejected a request for its size.

    Args:
        exc: BaseException — Exception raised by a backend call.

    Returns:
        bool — Whether the request must shrink before it can succeed.
    """
    if status_code(exc) == PAYLOAD_TOO_LARGE:
        return True

    body = getattr(exc, "body", None)
    if isinstance(body, dict) and body.get("type") == "request_too_large":
        return True

    message = str(exc)
    return any(marker in message for marker in _TOO_LARGE_MARKERS)
