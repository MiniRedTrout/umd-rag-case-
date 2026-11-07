from __future__ import annotations

import asyncio
from collections.abc import Sequence
from enum import StrEnum

from src.core.config.agent_config import get_agent_config
from src.rag.chunks import ChunkRecord, DocumentContext, merge_records
from src.rag.rankers.reciprocal_rank import reciprocal_rank_fusion
from src.utils.group_indices import group_chunk_indices
from src.utils.logger import logger


class SearchStrategy(StrEnum):
    """Define a corpus-ranking strategy."""

    VECTOR_E5 = "e5"
    KEYWORD = "bm25"
    VECTOR_BGE = "bge"


DEFAULT_STRATEGIES: tuple[SearchStrategy, ...] = (
    SearchStrategy.VECTOR_E5,
    SearchStrategy.KEYWORD,
    SearchStrategy.VECTOR_BGE,
)


class ChunkRepository:
    """Provide search, fetch, and readable-context access to the corpus."""

    def __init__(self, wrapper) -> None:
        self._wrapper = wrapper

    async def search(
        self,
        queries: Sequence[str],
        *,
        limit: int,
        strategies: Sequence[SearchStrategy] = DEFAULT_STRATEGIES,
        per_strategy: int | None = None,
    ) -> list[ChunkRecord]:
        """Search and fuse chunks for query phrasings."""

        wanted = [query for query in queries if (query or "").strip()]
        if not wanted or not strategies:
            return []

        retrieval = get_agent_config().retrieval
        k = per_strategy or retrieval.chunks_to_retrieve
        semaphore = asyncio.Semaphore(retrieval.concurrency)

        async def run(strategy: SearchStrategy, query: str) -> list[ChunkRecord]:
            async with semaphore:
                return await self._one(strategy, query, k=k)

        ranked = await asyncio.gather(
            *(run(strategy, query) for strategy in strategies for query in wanted)
        )
        return reciprocal_rank_fusion(list(ranked))[:limit]

    async def _one(
        self,
        strategy: SearchStrategy,
        query: str,
        *,
        k: int,
    ) -> list[ChunkRecord]:
        """Run one strategy against one query."""
        if strategy is SearchStrategy.KEYWORD:
            return await self._wrapper.query_keyword(query, k=k)

        return await self._wrapper.query_vector(
            query,
            k=k,
            use_e5=strategy is SearchStrategy.VECTOR_E5,
        )

    async def fetch(
        self,
        path: str,
        indices: Sequence[int],
    ) -> list[ChunkRecord]:
        """Fetch named chunks of one document in document order."""
        if not indices:
            return []
        return await self._wrapper.fetch_chunks(
            path=path,
            ids=list(indices),
            k=len(indices),
        )

    async def document_context(
        self,
        chunks: Sequence[tuple[str, int]],
    ) -> list[DocumentContext]:
        """Expand chosen chunks into readable passages."""
        if not chunks:
            return []

        runs = group_chunk_indices(list(chunks))
        semaphore = asyncio.Semaphore(get_agent_config().retrieval.concurrency)

        async def read(
            path: str, ids: list[int]
        ) -> tuple[str, list[int], list[ChunkRecord]]:
            async with semaphore:
                return (
                    path,
                    ids,
                    await self._wrapper.fetch_chunks(
                        path=path,
                        ids=ids,
                        k=len(ids),
                    ),
                )

        fetched = await asyncio.gather(*(read(path, ids) for path, ids in runs))

        contexts: list[DocumentContext] = []
        for path, ids, records in fetched:
            context = merge_records(records)
            if context is None or not context.text.strip():
                logger.warning(f"No chunks fetched for '{path}' {ids}")
                continue
            contexts.append(context)
        return contexts