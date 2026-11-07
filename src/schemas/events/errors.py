from __future__ import annotations

from src.schemas.events.sse import ErrorType

_STATUS_TO_ERROR_TYPE: dict[int, ErrorType] = {
    401: "authentication_error",
    403: "permission_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    451: "content_filter_error",
    502: "service_unavailable_error",
    503: "service_unavailable_error",
    504: "timeout_error",
    529: "overloaded_error",
}

_PUBLIC_UPSTREAM_MESSAGES: dict[ErrorType, str] = {
    "service_unavailable_error": "Сервис временно недоступен. Попробуйте позже.",
    "overloaded_error": "Сервис перегружен. Попробуйте позже.",
    "timeout_error": "Сервис не ответил вовремя. Попробуйте позже.",
    "rate_limit_error": "Слишком много запросов. Попробуйте позже.",
    "authentication_error": "Сервис недоступен. Попробуйте позже.",
    "permission_error": "Недостаточно прав для выполнения запроса.",
    "not_found_error": "Запрошенный ресурс не найден.",
    "request_too_large": "Запрос слишком большой. Сократите его и повторите.",
    "invalid_request_error": "Некорректный запрос.",
    "content_filter_error": "Запрос отклонён политикой безопасности.",
    "api_error": "Сервис вернул ошибку. Попробуйте позже.",
}

_RECOVERABLE: frozenset[ErrorType] = frozenset(
    {
        "service_unavailable_error",
        "overloaded_error",
        "rate_limit_error",
        "timeout_error",
        "api_error",
    }
)


def _status_code_of(exc: BaseException) -> int | None:
    direct = getattr(exc, "status_code", None)
    if isinstance(direct, int):
        return direct

    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    return None


def classify_exception(exc: BaseException) -> tuple[ErrorType, int | None]:
    """Classify an exception and its upstream status.

    Args:
        exc: BaseException — Exception to classify.

    Returns:
        tuple[ErrorType, int | None] — Error type and upstream status, if available.
    """
    status_code = _status_code_of(exc)
    if status_code is not None:
        return _STATUS_TO_ERROR_TYPE.get(status_code, "api_error"), status_code
    if isinstance(exc, TimeoutError):
        return "timeout_error", None
    return "api_error", None


def public_exception_message(exc: BaseException) -> str:
    """Return a client-safe message for an exception.

    Args:
        exc: BaseException — Exception to present safely.

    Returns:
        str — Public error message.
    """
    error_type, _ = classify_exception(exc)
    return _PUBLIC_UPSTREAM_MESSAGES[error_type]


def is_recoverable(error_type: ErrorType) -> bool:
    """Determine whether a stream may continue after an error.

    Args:
        error_type: ErrorType — Classified error type.

    Returns:
        bool — Whether the error is recoverable.
    """
    return error_type in _RECOVERABLE
