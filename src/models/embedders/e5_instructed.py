from __future__ import annotations
from typing import Any

from src.utils.e5_text_wrapper import add_passage, add_query, E5_QUERY_PREFIX


class E5Instructed:
    """Wrap an embeddings object with E5 input prefixes.

    Args:
        embeddings: Any — Embeddings object to wrap.
    """

    def __init__(self, embeddings: Any) -> None:
        self._embeddings = embeddings

    def __getattr__(self, name: str) -> Any:
        return getattr(self._embeddings, name)

    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return self._embeddings.embed_documents(add_passage(texts), **kwargs)

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        return self._embeddings.embed_query(f"{E5_QUERY_PREFIX}{text}", **kwargs)

    async def aembed_documents(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float]]:
        return await self._embeddings.aembed_documents(add_passage(texts), **kwargs)

    async def aembed_queries(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float]]:
        return await self._embeddings.aembed_documents(add_query(texts), **kwargs)

    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        return await self._embeddings.aembed_query(f"{E5_QUERY_PREFIX}{text}", **kwargs)
