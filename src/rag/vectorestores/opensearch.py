from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any

from langchain_openai import OpenAIEmbeddings
from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import AuthenticationException

from src.core.config.agent_config import get_agent_config
from src.models.embedders.size_fallback import EmbeddingStats
from src.rag.chunks import (
    SOURCE_FIELDS,
    ChunkRecord,
    record_from_source,
    record_to_source,
)
from src.utils.logger import logger
from src.utils.splitter import truncate_to_tokens

TITLE_TOKEN_BUDGET = 48

_SOURCE_FIELDS = SOURCE_FIELDS

_PROGRESS_STEP_PERCENT = 10.0


@dataclass(frozen=True)
class ModelIndexingStats:
    """Embedding request counters for one model during one import."""

    model: str
    requests: int
    size_rejections: int
    split_retries: int
    single_rejections: int


@dataclass(frozen=True)
class IndexingReport:
    """Operational summary of one completed indexing pass."""

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


@dataclass(frozen=True)
class _PreparedBatch:
    """One embed-and-bulk unit, with document closes counted by the producer."""

    batch_number: int
    chunks: tuple[ChunkRecord, ...]
    documents_closed: int
    chars_min: int
    chars_max: int
    chars_total: int


def _embeddable(chunk: ChunkRecord) -> str:
    """Return the text a chunk is embedded as, headed by its document title.

    The title is truncated so a long filename cannot eat the embedding
    model's context budget: only the first ``TITLE_TOKEN_BUDGET`` tokens of
    the title are kept.

    Args:
        chunk: ChunkRecord — Chunk about to be embedded.

    Returns:
        str — Title and text, or the text alone when there is no title.
    """
    title = truncate_to_tokens((chunk.title or "").strip(), TITLE_TOKEN_BUDGET)
    return f"{title}\n\n{chunk.text}" if title else chunk.text


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
    """Retry an idempotent bulk body when the authentication backend returns 401."""
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


def _log_progress(
    index_name: str,
    *,
    documents_done: int,
    documents_total: int | None,
    chunks_done: int,
    started_at: float,
    reported_percent: float,
) -> float:
    """Log indexing progress at percentage milestones over documents.

    Args:
        index_name: Index being populated.
        documents_done: Documents whose chunks have all been indexed.
        documents_total: Total documents when known; None falls back to chunks.
        chunks_done: Chunks indexed so far.
        started_at: Monotonic start time.
        reported_percent: Last reported progress percentage.

    Returns:
        float — Newly reported progress percentage.
    """
    if documents_total:
        done = documents_done
        total = documents_total
        unit = "документов"
        detail = f", {chunks_done} чанка"
    else:
        done = chunks_done
        total = max(chunks_done, 1)
        unit = "chunks"
        detail = ""

    percent = done / total * 100 if total else 100.0
    if percent - reported_percent < _PROGRESS_STEP_PERCENT and done < total:
        return reported_percent

    elapsed = time.monotonic() - started_at
    remaining = (
        f", ~{(total - done) * elapsed / done:.0f}s left"
        if done and done < total
        else ""
    )
    logger.info(
        f"Indexing '{index_name}': {percent:.0f}% ({done}/{total} {unit}"
        f"{detail}, {elapsed:.0f}s elapsed{remaining})"
    )
    return percent


def _pull_batch(stream: Iterator[ChunkRecord], batch_size: int) -> list[ChunkRecord]:
    """Take the next fixed-size batch from a chunk stream.

    Args:
        stream: Iterator[ChunkRecord] — Remaining chunks, in document order.
        batch_size: int — Maximum chunks to take.

    Returns:
        list[ChunkRecord] — Next batch, empty when the stream is exhausted.
    """
    return list(islice(stream, batch_size))


def _close_documents(
    chunks: Sequence[ChunkRecord],
    *,
    last_doc_id: str | None,
    pending: bool,
    exhausted: bool,
) -> tuple[int, str | None, bool]:
    """Count documents whose last chunk falls in this batch.

    Chunks arrive document-ordered, so a change of ``doc_id`` closes the
    previous document. The final document of the corpus closes when the stream
    is exhausted.

    Args:
        chunks: Sequence[ChunkRecord] — Chunks in this batch, in stream order.
        last_doc_id: str | None — Document still open from the previous batch.
        pending: bool — Whether ``last_doc_id`` has any chunks already seen.
        exhausted: bool — Whether this batch is the last of the stream.

    Returns:
        tuple[int, str | None, bool] — Documents closed in this batch, and the
        updated cursor.
    """
    closed = 0
    for chunk in chunks:
        if last_doc_id is None:
            last_doc_id = chunk.doc_id
            pending = True
        elif chunk.doc_id != last_doc_id:
            closed += 1
            last_doc_id = chunk.doc_id
            pending = True
    if exhausted and pending:
        closed += 1
        pending = False
    return closed, last_doc_id, pending


def _reject_reopened_documents(
    chunks: Sequence[ChunkRecord],
    *,
    last_doc_id: str | None,
    closed_ids: set[str],
) -> None:
    """Refuse a document id that already closed earlier in the stream.

    Chunks are indexed as ``{doc_id}:{chunk_index}``. A later document that
    reuses an id overwrites the earlier bulk ``_id``s, and OpenSearch reports
    the write as success. Catch it here, while the path is still in hand.

    Args:
        chunks: Sequence[ChunkRecord] — Chunks in this batch, in stream order.
        last_doc_id: str | None — Document still open from the previous batch.
        closed_ids: set[str] — Document ids whose last chunk has been seen.

    Raises:
        RuntimeError: If a previously closed document id reappears.
    """
    current_id = last_doc_id
    for chunk in chunks:
        if current_id is not None and chunk.doc_id != current_id:
            closed_ids.add(current_id)
        if chunk.doc_id in closed_ids:
            raise RuntimeError(
                f"Duplicate document id {chunk.doc_id!r} reappeared at "
                f"{chunk.path!r}; repeated bulk _ids overwrite instead of failing"
            )
        current_id = chunk.doc_id


def _char_stats(chunks: Sequence[ChunkRecord]) -> tuple[int, int, int]:
    """Return min, max, and total character counts for chunk bodies."""
    if not chunks:
        return 0, 0, 0
    lengths = [len(chunk.text) for chunk in chunks]
    return min(lengths), max(lengths), sum(lengths)


def build_filters(
    *,
    ids: list[int] | None = None,
    doc_source: str | None = None,
) -> list[dict]:
    """Build shared keyword and vector search filters.

    Args:
        ids: Optional chunk-index filter.
        doc_source: Optional document path filter.

    Returns:
        list[dict] — OpenSearch filter clauses.
    """
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
        """Initialize client, embeddings, and index settings.

        Args:
            embeddings_e5: E5 embedding provider.
            embeddings_bge: BGE embedding provider.
            embedding_field_name_e5: E5 vector field name.
            embedding_field_name_bge: BGE vector field name.
            index_name: Default OpenSearch index.
            client: OpenSearch client.
        """
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
        *,
        total_documents: int | None = None,
    ) -> IndexingReport:
        """Embed and index chunk text in OpenSearch from a stream.

        Runs a producer and ``concurrency`` workers under one ``TaskGroup``:
        the producer reads the chunk stream, the workers embed and bulk
        independently. A bounded queue applies backpressure so a fast producer
        cannot exhaust memory.

        Args:
            chunks: Records to index. Consumed in fixed-size batches so a
                large import never materialises the whole corpus in memory.
            index_name: Optional target index.
            total_documents: Optional document count for progress logging.
                When set, progress is reported over documents (known up front);
                chunk count is only known after the stream is exhausted.

        Raises:
            RuntimeError: If embedding or bulk indexing fails.
        """
        target_index = index_name or self.index_name
        agent_config = get_agent_config()
        batch_size = agent_config.embedding.max_batch_size
        indexing_config = agent_config.indexing
        concurrency = indexing_config.concurrency
        stream = iter(chunks)

        e5_before = _stats_snapshot(self.embeddings_e5)
        bge_before = _stats_snapshot(self.embeddings_bge)
        started_at = time.monotonic()
        progress_lock = asyncio.Lock()
        reported_percent = 0.0
        chunks_done = 0
        documents_done = 0
        batches = 0
        bulk_auth_retries = 0
        chars_min: int | None = None
        chars_max = 0
        chars_total = 0
        embed_seconds = 0.0
        bulk_seconds = 0.0
        queue: asyncio.Queue[_PreparedBatch | None] = asyncio.Queue(maxsize=concurrency)

        async def produce() -> None:
            nonlocal batches
            last_doc_id: str | None = None
            pending = False
            closed_ids: set[str] = set()
            current = await asyncio.to_thread(_pull_batch, stream, batch_size)
            while current:
                nxt = await asyncio.to_thread(_pull_batch, stream, batch_size)
                batches += 1
                _reject_reopened_documents(
                    current, last_doc_id=last_doc_id, closed_ids=closed_ids
                )
                closed, last_doc_id, pending = _close_documents(
                    current,
                    last_doc_id=last_doc_id,
                    pending=pending,
                    exhausted=not nxt,
                )
                chars_lo, chars_hi, chars_sum = _char_stats(current)
                await queue.put(
                    _PreparedBatch(
                        batch_number=batches,
                        chunks=tuple(current),
                        documents_closed=closed,
                        chars_min=chars_lo,
                        chars_max=chars_hi,
                        chars_total=chars_sum,
                    )
                )
                current = nxt
            for _ in range(concurrency):
                await queue.put(None)

        async def index_batch(batch: _PreparedBatch) -> None:
            nonlocal reported_percent, chunks_done, documents_done
            nonlocal bulk_auth_retries, chars_min, chars_max, chars_total
            nonlocal embed_seconds, bulk_seconds

            item_count = len(batch.chunks)
            batch_keys = [chunk.chunk_key for chunk in batch.chunks]
            batch_texts = [_embeddable(chunk) for chunk in batch.chunks]
            logger.debug(
                f"Processing batch {batch.batch_number} "
                f"(item_count={item_count}, chunk_keys={batch_keys})"
            )

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
                if len(batch_vectors_e5) != item_count:
                    raise RuntimeError(
                        f"E5 returned {len(batch_vectors_e5)} vectors for "
                        f"{item_count} chunks"
                    )
                if len(batch_vectors_bge) != item_count:
                    raise RuntimeError(
                        f"BGE returned {len(batch_vectors_bge)} vectors for "
                        f"{item_count} chunks"
                    )
            except Exception as exc:
                logger.exception(
                    f"Embedding failed for batch {batch.batch_number}: "
                    f"item_count={item_count}, chunk_keys={batch_keys}"
                )
                raise RuntimeError(
                    f"Embedding failure in batch {batch.batch_number}"
                ) from exc
            embed_elapsed = time.monotonic() - embed_started

            bulk_operations: list[dict] = []
            for chunk, vector_e5, vector_bge in zip(
                batch.chunks, batch_vectors_e5, batch_vectors_bge, strict=True
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
            try:
                response, retries = await _bulk_with_auth_retry(
                    self.client,
                    bulk_operations,
                    batch_number=batch.batch_number,
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
                    logger.error(
                        f"OpenSearch rejected {len(failures)} of {item_count} "
                        f"documents in batch {batch.batch_number}; "
                        f"first failure id={first.get('_id')!r}: {detail}"
                    )
                    raise RuntimeError(
                        f"Bulk indexing rejected {len(failures)} documents in "
                        f"batch {batch.batch_number}: {detail.get('type')} — "
                        f"{detail.get('reason')}"
                    )
            except Exception as exc:
                logger.exception(f"Bulk indexing failed for batch {batch.batch_number}")
                raise RuntimeError(
                    f"Bulk indexing failure in batch {batch.batch_number}"
                ) from exc
            bulk_elapsed = time.monotonic() - bulk_started

            async with progress_lock:
                chunks_done += item_count
                documents_done += batch.documents_closed
                bulk_auth_retries += retries
                embed_seconds += embed_elapsed
                bulk_seconds += bulk_elapsed
                chars_min = (
                    batch.chars_min
                    if chars_min is None
                    else min(chars_min, batch.chars_min)
                )
                chars_max = max(chars_max, batch.chars_max)
                chars_total += batch.chars_total
                reported_percent = _log_progress(
                    target_index,
                    documents_done=documents_done,
                    documents_total=total_documents,
                    chunks_done=chunks_done,
                    started_at=started_at,
                    reported_percent=reported_percent,
                )

            del response
            del bulk_operations
            del batch_vectors_e5, batch_vectors_bge
            del batch_texts, batch_keys

        async def worker() -> None:
            while True:
                batch = await queue.get()
                if batch is None:
                    return
                await index_batch(batch)

        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(produce())
                for _ in range(concurrency):
                    group.create_task(worker())
        except ExceptionGroup as errors:
            raise errors.exceptions[0] from errors

        if chunks_done == 0:
            logger.info("No chunks to process.")
            return IndexingReport(
                chunks=0,
                batches=0,
                elapsed_seconds=0.0,
                chars_min=0,
                chars_max=0,
                chars_average=0.0,
                bulk_auth_retries=0,
                embed_seconds=0.0,
                bulk_seconds=0.0,
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
        """Retrieve keyword-matching chunks for a question.

        Args:
            question: Query text to search.
            k: Number of top results to return. If None, uses retrieval.top_k.
            ids: Optional chunk indices to filter by.
            doc_source: Optional document path to filter by.

        Returns:
            list[ChunkRecord] — Retrieved chunks.
        """
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
            "_source": _SOURCE_FIELDS,
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
        """Retrieve similar chunks for a question.

        Args:
            question: Query text to embed and search.
            ids: Optional chunk indices to filter by.
            doc_source: Optional document path to filter by.
            k: Number of top results to return. If None, uses retrieval.top_k.
            use_e5: Whether to use the E5 embedding provider.

        Returns:
            list[ChunkRecord] — Retrieved chunks.
        """
        if not (question or "").strip():
            logger.debug(
                "query_vector called with an empty question; returning no hits"
            )
            return []

        if use_e5:
            q_vector = await self.embeddings_e5.aembed_query(question)
            embedding_field = self.embedding_field_name_e5
        else:
            q_vector = await self.embeddings_bge.aembed_query(question)
            embedding_field = self.embedding_field_name_bge

        if k is None:
            k = get_agent_config().retrieval.top_k

        filters = build_filters(ids=ids, doc_source=doc_source)

        knn_clause: dict[str, Any] = {"vector": q_vector, "k": k}
        if filters:
            knn_clause["filter"] = {"bool": {"must": filters}}

        search_body = {
            "size": k,
            "query": {"knn": {embedding_field: knn_clause}},
            "_source": _SOURCE_FIELDS,
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
        """Fetch named chunks from one document in document order.

        Args:
            path: Document path.
            ids: Chunk indices to fetch.
            k: Optional maximum number of chunks.

        Returns:
            list[ChunkRecord] — Matching chunks.
        """
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
            "_source": _SOURCE_FIELDS,
        }

        result = await self.client.search(index=self.index_name, body=search_body)
        return self._process_search_result(result["hits"]["hits"])

    @staticmethod
    def _process_search_result(
        hits_raw: Sequence[Mapping[str, Any]],
    ) -> list[ChunkRecord]:
        """Convert raw OpenSearch hits to chunk records.

        Args:
            hits_raw: Raw OpenSearch hits.

        Returns:
            list[ChunkRecord] — Converted chunk records.
        """
        return [record_from_source(hit["_source"]) for hit in hits_raw]

    async def close(self) -> None:
        """Close the OpenSearch client connection."""
        await self.client.close()