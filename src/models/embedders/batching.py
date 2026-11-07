"""Stable, order-preserving batches for corpus embedding."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingItem:
    """One indexed text and the identity used in diagnostics."""

    chunk_key: str
    text: str


@dataclass(frozen=True)
class EmbeddingBatch:
    """An order-preserving embedding request unit."""

    items: tuple[EmbeddingItem, ...]

    @property
    def texts(self) -> list[str]:
        return [item.text for item in self.items]

    @property
    def chunk_keys(self) -> list[str]:
        return [item.chunk_key for item in self.items]

    @property
    def item_count(self) -> int:
        return len(self.items)


def build_embedding_batches(
    items: Iterable[EmbeddingItem], *, max_items: int
) -> list[EmbeddingBatch]:
    """Split items into fixed-size batches, preserving their input order.

    Size is counted in items, and only in items. An estimated-token budget used
    to cap batches as well, but it bound first and invisibly: with 384-token
    chunks it filled a batch at 21 items regardless of the item limit, so the
    item limit read like a setting and behaved like a comment. It also cost a
    tokenizer pass over the whole corpus before a single vector was requested.

    What actually bounds a request is the backend, and that limit is already
    handled where it is observed: a batch the server refuses for its size is
    bisected and retried by ``embed_with_size_fallback``. Chunk length is capped
    upstream by ``chunking.chunk_size`` and again by server-side truncation, so
    the largest request this can build is bounded and predictable.

    Args:
        items: Iterable[EmbeddingItem] — Items to batch, in index order.
        max_items: int — Items per batch.

    Returns:
        list[EmbeddingBatch] — Batches in input order.

    Raises:
        ValueError: If ``max_items`` is not positive.
    """
    if max_items < 1:
        raise ValueError(f"max_items must be positive, got {max_items}")

    batches: list[EmbeddingBatch] = []
    current: list[EmbeddingItem] = []

    for item in items:
        current.append(item)
        if len(current) == max_items:
            batches.append(EmbeddingBatch(tuple(current)))
            current = []

    if current:
        batches.append(EmbeddingBatch(tuple(current)))
    return batches
