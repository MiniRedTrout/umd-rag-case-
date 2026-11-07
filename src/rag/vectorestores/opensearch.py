from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import AuthenticationException

from src.core.config.agent_config import get_agent_config
from src.models.embedders.size_fallback import EmbeddingStats
from src.rag.chunks import SOURCE_FIELDS, ChunkRecord, record_from_source, record_to_source
from src.utils.logger import logger
from src.utils.splitter import reduce_length, truncate_to_tokens

#: Tokens the document title may add on top of ``chunking.chunk_size``.
TITLE_TOKEN_BUDGET = 48


@dataclass(frozen=True)
class ModelIndexingStats:
    model: str
    requests: int
    size_rejections: int
    split_retries: int
    single_rejections: int


@dataclass(frozen=True)
class IndexingReport:
    chunks: int
    batches: int
    elapsed_seconds: float
    chars_min: int
    chars_max: int
    chars_average: float
    bulk_auth_retries: int
    embed_seconds: float
    bulk_seconds: float
    models: tuple[ModelIndexingStats, ...]


def _embeddable(chunk: ChunkRecord) -> str:
    """Return the text a chunk is embedded as, headed by its document title."""
    title = truncate_to_tokens((chunk.title or "").strip(), TITLE_TOKEN_BUDGET)
    return f"{title}\n\n{chunk.text}" if title else chunk.text


def _faq_key(entry: FAQEntry) -> str:
    payload = "\0".join((entry.question, entry.answer))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _document_key(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _stats_snapshot(embeddings) -> EmbeddingStats:
    stats = getattr(embeddings, "embedding_stats", None)
    return stats.copy() if stats is not None else EmbeddingStats()


def _model_stats(embeddings, earlier: EmbeddingStats) -> ModelIndexingStats:
    current = _stats_snapshot(embeddings).since(earlier)
    return ModelIndexingStats(
        model=str(getattr(embeddings, "model_label", type(embeddings).__name__)),
        requests=current.requests,
        size_rejections=current.size_rejections,
        split_retries=current.split_retries,
        single_rejections=current.single_rejections,
    )


async def _bulk_with_auth_retry(
    client: AsyncOpenSearch,
    operations: list[dict],
    *,
    batch_number: int,
    max_retries: int,
    initial_backoff_s: float,
) -> tuple[dict, int]:
    retries = 0
    while True:
        try:
            return await client.bulk(body=operations), retries
        except AuthenticationException:
            if retries >= max_retries:
                raise
            delay = initial_backoff_s * (2**retries)
            retries += 1
            logger.warning(
                f"OpenSearch authentication failed for bulk batch {batch_number}; "
                f"retrying in {delay:.1f}s ({retries}/{max_retries})"
            )
            if delay:
                await asyncio.sleep(delay)


def _pull_batch(stream: Iterator[ChunkRecord], batch_size: int) -> list[ChunkRecord]:
    return list(islice(stream, batch_size))


def _char_stats(chunks: Sequence[ChunkRecord]) -> tuple[int, int, int]:
    if not chunks:
        return 0, 0, 0
    lengths = [len(chunk.text) for chunk in chunks]
    return min(lengths), max(lengths), sum(lengths)


def build_filters(
    *,
    ids: list[int] | None = None,
    doc_source: str | None = None,
) -> list[dict]:
    filters: list[dict] = []
    if ids:
        filters.append({"terms": {"chunk_index": ids}})
    if doc_source:
        filters.append({"terms": {"path": [doc_source]}})
    return filters


class AsyncOpenSearchWrapper:
    """Provide an asynchronous OpenSearch-backed vector store."""

    def __init__(
        self,
        embeddings_e5: OpenAIEmbeddings,
        embeddings_bge: OpenAIEmbeddings,
        embedding_field_name_e5: str,
        embedding_field_name_bge: str,
        index_name: str,
        client: AsyncOpenSearch,
    ) -> None:
        self.client = client
        self.embeddings_e5 = embeddings_e5
        self.embeddings_bge = embeddings_bge
        self.index_name = index_name
        self.embedding_field_name_e5 = embedding_field_name_e5
        self.embedding_field_name_bge = embedding_field_name_bge

    async def add_chunks(
        self,
        chunks: Iterable[ChunkRecord],
        index_name: str | None = None,
    ) -> IndexingReport:
        target_index = index_name or self.index_name
        agent_config = get_agent_config()
        batch_size = agent_config.embedding.max_batch_size
        indexing_config = agent_config.indexing
        stream = iter(chunks)

        e5_before = _stats_snapshot(self.embeddings_e5)
        bge_before = _stats_snapshot(self.embeddings_bge)
        started_at = time.monotonic()

        chunks_done = 0
        batches = 0
        bulk_auth_retries = 0
        chars_min: int | None = None
        chars_max = 0
        chars_total = 0
        embed_seconds = 0.0
        bulk_seconds = 0.0

        while True:
            batch_chunks = await asyncio.to_thread(_pull_batch, stream, batch_size)
            if not batch_chunks:
                break

            batches += 1
            batch_number = batches
            item_count = len(batch_chunks)
            batch_keys = [chunk.chunk_key for chunk in batch_chunks]
            batch_texts = [_embeddable(chunk) for chunk in batch_chunks]
            chars_lo, chars_hi, chars_sum = _char_stats(batch_chunks)

            embed_started = time.monotonic()
            try:
                batch_vectors_e5, batch_vectors_bge = await asyncio.gather(
                    self.embeddings_e5.aembed_documents(
                        batch_texts, chunk_keys=batch_keys
                    ),
                    self.embeddings_bge.aembed_documents(
                        batch_texts, chunk_keys=batch_keys
                    ),
                )
            except Exception as exc:
                logger.exception(
                    f"Embedding failed for batch {batch_number}: "
                    f"item_count={item_count}, chunk_keys={batch_keys}"
                )
                raise RuntimeError(
                    f"Embedding failure in batch {batch_number}"
                ) from exc
            embed_elapsed = time.monotonic() - embed_started

            bulk_operations = []
            for chunk, vector_e5, vector_bge in zip(
                batch_chunks, batch_vectors_e5, batch_vectors_bge, strict=True
            ):
                doc_body = {
                    self.embedding_field_name_e5: vector_e5,
                    self.embedding_field_name_bge: vector_bge,
                    **record_to_source(chunk),
                }
                bulk_operations.append(
                    {"index": {"_index": target_index, "_id": chunk.chunk_key}}
                )
                bulk_operations.append(doc_body)

            bulk_started = time.monotonic()
            response, retries = await _bulk_with_auth_retry(
                self.client,
                bulk_operations,
                batch_number=batch_number,
                max_retries=indexing_config.bulk_auth_retries,
                initial_backoff_s=indexing_config.bulk_auth_backoff_s,
            )
            if response.get("errors"):
                failures = [
                    item["index"]
                    for item in response.get("items", [])
                    if item.get("index", {}).get("error")
                ]
                first = failures[0] if failures else {}
                detail = first.get("error", {})
                raise RuntimeError(
                    f"Bulk indexing rejected {len(failures)} documents in "
                    f"batch {batch_number}: {detail.get('type')} — "
                    f"{detail.get('reason')}"
                )
            bulk_elapsed = time.monotonic() - bulk_started

            chunks_done += item_count
            bulk_auth_retries += retries
            embed_seconds += embed_elapsed
            bulk_seconds += bulk_elapsed
            chars_min = chars_lo if chars_min is None else min(chars_min, chars_lo)
            chars_max = max(chars_max, chars_hi)
            chars_total += chars_sum

        if chunks_done == 0:
            logger.info("No chunks to process.")
            return IndexingReport(
                chunks=0, batches=0, elapsed_seconds=0.0,
                chars_min=0, chars_max=0, chars_average=0.0,
                bulk_auth_retries=0, embed_seconds=0.0, bulk_seconds=0.0,
                models=(),
            )

        elapsed = time.monotonic() - started_at
        return IndexingReport(
            chunks=chunks_done,
            batches=batches,
            elapsed_seconds=elapsed,
            chars_min=chars_min or 0,
            chars_max=chars_max,
            chars_average=chars_total / chunks_done,
            bulk_auth_retries=bulk_auth_retries,
            embed_seconds=embed_seconds,
            bulk_seconds=bulk_seconds,
            models=(
                _model_stats(self.embeddings_e5, e5_before),
                _model_stats(self.embeddings_bge, bge_before),
            ),
        )

    async def query_keyword(
        self,
        question: str,
        k: int | None = None,
        ids: list[int] | None = None,
        doc_source: str | None = None,
    ) -> list[ChunkRecord]:
        retrieval = get_agent_config().retrieval
        if k is None:
            k = retrieval.top_k
        question = (question or "")[: retrieval.max_retrieval_query_chars]
        filters = build_filters(ids=ids, doc_source=doc_source)

        search_body = {
            "size": k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": question,
                                "fields": ["text^3"],
                                "fuzziness": "AUTO",
                            }
                        }
                    ],
                    "should": [
                        {"match": {"title": {"query": question, "boost": 2}}}
                    ],
                    "filter": filters,
                }
            },
            "_source": SOURCE_FIELDS,
        }

        search_result = await self.client.search(
            index=self.index_name, body=search_body
        )
        return self._process_search_result(search_result["hits"]["hits"])

    async def query_vector(
        self,
        question: str,
        ids: list[int] | None = None,
        doc_source: str | None = None,
        k: int | None = None,
        use_e5: bool = True,
    ) -> list[ChunkRecord]:
        if not (question or "").strip():
            logger.debug(
                "query_vector called with an empty question; returning no hits"
            )
            return []

        if use_e5:
            q_vector = await self.embeddings_e5.aembed_query(
                reduce_length(question)
            )
            embedding_field = self.embedding_field_name_e5
        else:
            q_vector = await self.embeddings_bge.aembed_query(
                reduce_length(question)
            )
            embedding_field = self.embedding_field_name_bge

        if k is None:
            k = get_agent_config().retrieval.top_k

        filters = build_filters(ids=ids, doc_source=doc_source)

        knn_clause = {
            "vector": q_vector,
            "k": k,
        }

        if filters:
            knn_clause["filter"] = {"bool": {"must": filters}}

        search_body = {
            "size": k,
            "query": {
                "knn": {
                    embedding_field: knn_clause
                }
            },
            "_source": SOURCE_FIELDS,
        }

        search_result = await self.client.search(
            index=self.index_name, body=search_body
        )
        return self._process_search_result(search_result["hits"]["hits"])

    async def fetch_chunks(
        self,
        path: str,
        ids: list[int],
        k: int | None = None,
    ) -> list[ChunkRecord]:
        if not ids:
            return []

        filters: list[dict] = [
            {"term": {"path": path}},
            {"terms": {"chunk_index": ids}},
        ]

        search_body = {
            "size": k or len(ids),
            "query": {"bool": {"filter": filters}},
            "sort": [{"chunk_index": "asc"}],
            "_source": SOURCE_FIELDS,
        }

        result = await self.client.search(index=self.index_name, body=search_body)
        return self._process_search_result(result["hits"]["hits"])

    @staticmethod
    def _process_search_result(
        hits_raw: Sequence[Mapping[str, Any]],
    ) -> list[ChunkRecord]:
        return [record_from_source(hit["_source"]) for hit in hits_raw]

    async def close(self) -> None:
        await self.client.close()