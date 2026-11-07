from __future__ import annotations

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.utils.request_context import (
    CORRELATION_CAPABILITY_HEADER,
    CORRELATION_CAPABILITY_VALUE,
    REQUEST_ID_CONTEXT_VAR,
    REQUEST_ID_HEADER,
    resolve_request_id,
)

_REQUEST_ID_BYTES = REQUEST_ID_HEADER.lower().encode("latin-1")


def _request_id_from_scope(scope: Scope) -> str:
    """Resolve the correlation id from the raw ASGI request headers.

    Returns the caller's id when safe, otherwise a generated one.
    """
    raw: str | None = None
    for key, value in scope.get("headers") or []:
        if key == _REQUEST_ID_BYTES:
            raw = value.decode("latin-1")
            break
    return resolve_request_id(raw)


class RequestIDMiddleware:
    """Carry ``X-Request-ID`` from the HTTP request into the process context.

    The validated (or freshly generated) id is bound to a ``ContextVar``
    before the application runs and reset in ``finally`` after it returns.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id_from_scope(scope)
        response_started = False

        async def send_with_correlation(message: Message) -> None:
            """Pass every message through, stamping the response headers.

            Only ``http.response.start`` is touched, in place; body frames
            are forwarded untouched, so nothing is buffered.
            """
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = request_id
                headers[CORRELATION_CAPABILITY_HEADER] = CORRELATION_CAPABILITY_VALUE
            await send(message)

        token = REQUEST_ID_CONTEXT_VAR.set(request_id)
        try:
            # `receive` is passed through untouched: it belongs to the
            # response that owns disconnect watching, never to this layer.
            await self.app(scope, receive, send_with_correlation)
        except Exception:
            if not response_started:
                # The application died before saying anything: say something
                # that still carries the correlation, then let the error
                # propagate so the server logs it as it always has.
                await send_with_correlation(
                    {
                        "type": "http.response.start",
                        "status": 500,
                        "headers": [(b"content-type", b"text/plain; charset=utf-8")],
                    }
                )
                await send_with_correlation(
                    {
                        "type": "http.response.body",
                        "body": b"Internal Server Error",
                        "more_body": False,
                    }
                )
            raise
        finally:
            REQUEST_ID_CONTEXT_VAR.reset(token)
