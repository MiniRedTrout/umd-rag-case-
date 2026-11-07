"""Recover from aggregate embedding request limits without hiding data loss."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from loguru import logger

from src.models.errors import PAYLOAD_TOO_LARGE, is_request_too_large, status_code
from src.utils.splitter import estimate_tokens

Vectors = list[list[float]]
SyncEmbedder = Callable[[list[str]], Vectors]
AsyncEmbedder = Callable[[list[str]], Awaitable[Vectors]]


class EmbeddingInputTooLarge(RuntimeError):
    """A single input remained too large after backend-side truncation."""

    status_code = PAYLOAD_TOO_LARGE


@dataclass
class EmbeddingStats:
    """Monotonic operational counters for one embedding model client."""

    requests: int = 0
    size_rejections: int = 0
    split_retries: int = 0
    single_rejections: int = 0

    def copy(self) -> EmbeddingStats:
        return EmbeddingStats(
            requests=self.requests,
            size_rejections=self.size_rejections,
            split_retries=self.split_retries,
            single_rejections=self.single_rejections,
        )

    def since(self, earlier: EmbeddingStats) -> EmbeddingStats:
        return EmbeddingStats(
            requests=self.requests - earlier.requests,
            size_rejections=self.size_rejections - earlier.size_rejections,
            split_retries=self.split_retries - earlier.split_retries,
            single_rejections=self.single_rejections - earlier.single_rejections,
        )


def _keys_for(texts: list[str], chunk_keys: list[str] | None) -> list[str]:
    if chunk_keys is None:
        return ["<unknown>"] * len(texts)
    if len(chunk_keys) != len(texts):
        raise ValueError("chunk_keys must have the same length as embedding texts")
    return chunk_keys


def _single_input_error(
    *, model: str, chunk_key: str, text: str, cause: Exception
) -> EmbeddingInputTooLarge:
    status = status_code(cause)
    return EmbeddingInputTooLarge(
        "Embedding backend rejected one input after server-side truncation: "
        f"model={model!r}, chunk_key={chunk_key!r}, item_count=1, "
        f"estimated_tokens={estimate_tokens(text)}, status={status}, "
        f"backend={cause}"
    )


def embed_with_size_fallback(
    texts: list[str],
    embed: SyncEmbedder,
    *,
    model: str,
    chunk_keys: list[str] | None = None,
    stats: EmbeddingStats | None = None,
) -> Vectors:
    """Embed texts, bisecting only aggregate request-size failures."""
    if not texts:
        return []

    keys = _keys_for(texts, chunk_keys)
    counters = stats if stats is not None else EmbeddingStats()
    counters.requests += 1
    try:
        return embed(texts)
    except Exception as exc:
        if not is_request_too_large(exc):
            raise
        counters.size_rejections += 1
        cause = exc

    if len(texts) == 1:
        counters.single_rejections += 1
        raise _single_input_error(
            model=model, chunk_key=keys[0], text=texts[0], cause=cause
        ) from cause

    middle = len(texts) // 2
    counters.split_retries += 1
    logger.debug(
        f"Embedding batch rejected for model={model!r}, item_count={len(texts)}, "
        f"estimated_tokens={sum(estimate_tokens(text) for text in texts)}; "
        f"splitting into {middle} and {len(texts) - middle}"
    )
    return embed_with_size_fallback(
        texts[:middle],
        embed,
        model=model,
        chunk_keys=keys[:middle],
        stats=counters,
    ) + embed_with_size_fallback(
        texts[middle:],
        embed,
        model=model,
        chunk_keys=keys[middle:],
        stats=counters,
    )


async def aembed_with_size_fallback(
    texts: list[str],
    embed: AsyncEmbedder,
    *,
    model: str,
    chunk_keys: list[str] | None = None,
    stats: EmbeddingStats | None = None,
) -> Vectors:
    """Asynchronously embed texts with aggregate-only batch bisection."""
    if not texts:
        return []

    keys = _keys_for(texts, chunk_keys)
    counters = stats if stats is not None else EmbeddingStats()
    counters.requests += 1
    try:
        return await embed(texts)
    except Exception as exc:
        if not is_request_too_large(exc):
            raise
        counters.size_rejections += 1
        cause = exc

    if len(texts) == 1:
        counters.single_rejections += 1
        raise _single_input_error(
            model=model, chunk_key=keys[0], text=texts[0], cause=cause
        ) from cause

    middle = len(texts) // 2
    counters.split_retries += 1
    logger.debug(
        f"Embedding batch rejected for model={model!r}, item_count={len(texts)}, "
        f"estimated_tokens={sum(estimate_tokens(text) for text in texts)}; "
        f"splitting into {middle} and {len(texts) - middle}"
    )
    head = await aembed_with_size_fallback(
        texts[:middle],
        embed,
        model=model,
        chunk_keys=keys[:middle],
        stats=counters,
    )
    tail = await aembed_with_size_fallback(
        texts[middle:],
        embed,
        model=model,
        chunk_keys=keys[middle:],
        stats=counters,
    )
    return head + tail
