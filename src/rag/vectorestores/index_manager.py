from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from opensearchpy import AsyncOpenSearch

from src.utils.logger import logger

_VISIBILITY_TIMEOUT_S = 300.0
_VISIBILITY_POLL_INTERVAL_S = 2.0
_VISIBILITY_LOG_INTERVAL_S = 30.0
_VISIBILITY_STALL_S = 60.0


@dataclass(frozen=True)
class IndexValidation:
    """Validated document and vector counts for a corpus index."""

    documents: int
    vectors: dict[str, int]


async def read_index_generation(client: AsyncOpenSearch, index_name: str) -> str:
    """Return an opaque token identifying the physical corpus behind a name.

    Built from the UUIDs OpenSearch assigns each index, so it changes exactly
    when the data does at the whole-index level: an import deletes and
    recreates the index, and a swap points an alias at a different one. Both
    give the remembered coordinates a different corpus to mean something
    against. ``index_name`` may be either an alias or a concrete index — the
    call resolves an alias to its members, and sorting them keeps the token
    stable when there is more than one.

    A rewrite that edits documents in place is not caught: same index, same
    UUID, different text. That case needs its own signal.

    Args:
        client: AsyncOpenSearch — Cluster client.
        index_name: str — Alias or index to identify.

    Returns:
        str — Generation token, or "" when it could not be read.
    """
    try:
        response = await client.indices.get(index=index_name)
    except Exception:  # noqa: BLE001 — an unknown generation is not fatal
        logger.exception(f"Could not read the generation of '{index_name}'")
        return ""
    uuids = sorted(
        str((body or {}).get("settings", {}).get("index", {}).get("uuid") or "")
        for body in (response or {}).values()
    )
    return ",".join(uuid for uuid in uuids if uuid)


async def count_documents(client: AsyncOpenSearch, index_name: str) -> int:
    """Count currently searchable chunks in an index.

    Args:
        client: AsyncOpenSearch — OpenSearch client.
        index_name: str — Index to count.

    Returns:
        int — Indexed chunk count.
    """
    response = await client.count(index=index_name)
    return int(response.get("count", 0))


async def validate_index(
    client: AsyncOpenSearch,
    index_name: str,
    *,
    expected_documents: int,
    embedding_fields: list[str],
    visibility_timeout_s: float = _VISIBILITY_TIMEOUT_S,
    visibility_poll_interval_s: float = _VISIBILITY_POLL_INTERVAL_S,
    visibility_stall_s: float = _VISIBILITY_STALL_S,
) -> IndexValidation:
    """Wait for automatic refresh, then require all chunks and vectors.

    OpenSearch refreshes indices automatically. Polling count queries avoids the
    privileged ``indices:admin/refresh`` operation while retaining a bounded,
    fail-closed validation step before declaring the rebuild complete.

    A shortfall that stops growing is not a slow refresh: repeated bulk
    ``_id``s overwrite, the count will never catch up, and waiting out the
    timeout cannot close it.
    """
    started_at = time.monotonic()
    deadline = started_at + visibility_timeout_s
    next_info_log = started_at
    last_snapshot: tuple[int, tuple[tuple[str, int], ...]] | None = None
    stall_since = started_at
    stalled = False

    while True:
        documents = await count_documents(client, index_name)
        vectors: dict[str, int] = {}
        for field in embedding_fields:
            response = await client.count(
                index=index_name,
                body={"query": {"exists": {"field": field}}},
            )
            vectors[field] = int(response.get("count", 0))

        if documents == expected_documents and all(
            count == expected_documents for count in vectors.values()
        ):
            elapsed = time.monotonic() - started_at
            if elapsed >= visibility_poll_interval_s:
                logger.info(
                    f"Index '{index_name}' became fully searchable after {elapsed:.1f}s"
                )
            return IndexValidation(documents=documents, vectors=vectors)

        if documents > expected_documents or any(
            count > expected_documents for count in vectors.values()
        ):
            break
        now = time.monotonic()
        snapshot = (documents, tuple(vectors.items()))
        if snapshot != last_snapshot:
            last_snapshot = snapshot
            stall_since = now
        elif now - stall_since >= visibility_stall_s:
            stalled = True
            break
        if now >= deadline:
            break

        message = (
            f"Waiting for index '{index_name}' to become searchable: "
            f"documents={documents}/{expected_documents}, vectors={vectors}"
        )
        if now >= next_info_log:
            logger.info(message)
            next_info_log = now + _VISIBILITY_LOG_INTERVAL_S
        else:
            logger.debug(message)
        await asyncio.sleep(visibility_poll_interval_s)

    vector_counts = ", ".join(
        f"{field}={count}/{expected_documents}" for field, count in vectors.items()
    )
    elapsed = time.monotonic() - started_at
    if stalled and documents < expected_documents:
        missing = expected_documents - documents
        raise RuntimeError(
            f"Index '{index_name}' holds {documents} of {expected_documents} chunks "
            f"({missing} missing) and stopped changing after {elapsed:.0f}s. "
            f"A shortfall that does not grow means duplicate chunk ids "
            f"in the stream: repeated bulk _ids overwrite instead of failing. "
            f"vectors=[{vector_counts}]"
        )
    raise RuntimeError(
        f"Index '{index_name}' did not become fully searchable within "
        f"{visibility_timeout_s:g}s: documents={documents}/{expected_documents}, "
        f"vectors=[{vector_counts}]"
    )
