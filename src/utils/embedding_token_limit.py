import os
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


DEFAULT_TOKEN_MARGIN = 16
DEFAULT_FALLBACK_ATTEMPTS = 3
TEXT_PREVIEW_LENGTH = 300


def get_embedding_max_tokens() -> int | None:
    raw_value = os.getenv("EMBEDDING_MAX_LEN")
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except ValueError:
        logger.warning(f"Invalid EMBEDDING_MAX_LEN={raw_value!r}; skipping embedding text fallback")
        return None


def get_embedding_fallback_attempts() -> int:
    raw_value = os.getenv("EMBEDDING_FALLBACK_ATTEMPTS")
    if raw_value is None:
        return DEFAULT_FALLBACK_ATTEMPTS
    try:
        return max(int(raw_value), 1)
    except ValueError:
        logger.warning(
            f"Invalid EMBEDDING_FALLBACK_ATTEMPTS={raw_value!r}; using {DEFAULT_FALLBACK_ATTEMPTS}"
        )
        return DEFAULT_FALLBACK_ATTEMPTS


def character_reduce_length(
    text: str,
    max_tokens: int | None = None,
    margin: int = DEFAULT_TOKEN_MARGIN,
    attempt: int = 0,
) -> str:
    if max_tokens is None:
        max_tokens = get_embedding_max_tokens()
    if max_tokens is None or text.strip() == "":
        return text

    budget = max(max_tokens // (2 ** attempt) - margin, 1)
    return text[:budget]


def local_reduce_length(text: str, max_tokens: int, margin: int = DEFAULT_TOKEN_MARGIN) -> str:
    if text.strip() == "":
        return text

    try:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=max(max_tokens - margin, 1),
            chunk_overlap=0,
        )
        split_result = text_splitter.split_text(text)
        return split_result[0] if split_result else text
    except Exception as exc:
        logger.warning(f"Local tiktoken text reduction failed; using character fallback: {exc}")
        return character_reduce_length(text, max_tokens, margin)


def is_request_too_large(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 413:
        return True

    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 413:
        return True

    error = getattr(exc, "body", None)
    if isinstance(error, dict) and error.get("type") == "request_too_large":
        return True

    error_message = str(exc)
    return "request_too_large" in error_message or "maximum context length" in error_message


def text_preview(text: str, max_length: int = TEXT_PREVIEW_LENGTH) -> str:
    preview = " ".join(text.split())
    if len(preview) <= max_length:
        return preview
    return f"{preview[:max_length]}..."


class EmbeddingTokenLimiter:
    def __init__(
        self,
        *,
        max_tokens: int | None = None,
        margin: int = DEFAULT_TOKEN_MARGIN,
        **_: Any,
    ) -> None:
        self.max_tokens = max_tokens if max_tokens is not None else get_embedding_max_tokens()
        self.margin = margin
        self.fallback_attempts = get_embedding_fallback_attempts()

    def character_limit_texts(self, texts: list[str], attempt: int = 0) -> list[str]:
        return [
            character_reduce_length(text, self.max_tokens, self.margin, attempt)
            for text in texts
        ]
