from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def chunk_key_of(doc_id: str, chunk_index: int) -> str:
    """Build a chunk identity."""
    return f"{doc_id}:{chunk_index}"


class ChunkRecord(BaseModel):
    """Represent an indexed chunk for ranking, reading, and citation."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    chunk_key: str
    doc_id: str
    chunk_index: int = 0
    chunk_total: int = 0

    path: str = ""
    title: str = ""
    url: str = ""
    source_format: str = ""
    text: str = ""

    @property
    def key(self) -> tuple[str, int]:
        return (self.doc_id, self.chunk_index)


class DocumentContext(BaseModel):
    """Represent a contiguous run of chunks merged for reading."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    path: str = ""
    title: str = ""
    url: str = ""
    source_format: str = ""
    chunk_total: int = 0
    #: Inclusive ``(first, last)`` chunk index of the run.
    chunk_range: tuple[int, int] = (0, 0)
    #: ``(chunk_index, start)`` of every chunk in ``text``, in document order.
    chunk_offsets: list[tuple[int, int]] = Field(default_factory=list)
    text: str = ""


_KEY_FIELD = "id"

SOURCE_FIELDS = [
    _KEY_FIELD,
    "doc_id",
    "path",
    "source_format",
    "title",
    "url",
    "chunk_index",
    "chunk_total",
    "text",
]


def record_to_source(record: ChunkRecord) -> dict[str, Any]:
    """Convert a chunk record to an OpenSearch document body."""
    body = record.model_dump(exclude={"chunk_key"}, by_alias=True)
    body[_KEY_FIELD] = record.chunk_key
    return body


def record_from_source(source: dict[str, Any]) -> ChunkRecord:
    """Build a chunk record from an OpenSearch source."""
    doc_id = str(source.get("doc_id") or "")
    chunk_index = int(source.get("chunk_index") or 0)
    return ChunkRecord(
        chunk_key=str(source.get(_KEY_FIELD) or chunk_key_of(doc_id, chunk_index)),
        doc_id=doc_id,
        chunk_index=chunk_index,
        chunk_total=int(source.get("chunk_total") or 0),
        path=str(source.get("path") or source.get("source") or ""),
        title=str(source.get("title") or source.get("filename") or ""),
        url=str(source.get("url") or ""),
        source_format=str(source.get("source_format") or ""),
        text=str(source.get("text") or ""),
    )


#: Shortest repeat taken for chunk overlap; shorter ones are likely chance.
_MIN_OVERLAP_CHARS = 20

#: How far back into the previous chunk an overlap is looked for.
_MAX_OVERLAP_CHARS = 2000


def _strip_overlap(previous: str, current: str) -> str:
    """Drop the head of ``current`` that repeats the tail of ``previous``."""
    tail = previous[-_MAX_OVERLAP_CHARS:]
    for size in range(min(len(tail), len(current)), _MIN_OVERLAP_CHARS - 1, -1):
        if tail.endswith(current[:size]):
            return current[size:].lstrip()
    return current


def merge_records(records: list[ChunkRecord]) -> DocumentContext | None:
    """Merge contiguous chunk records into readable document context."""
    if not records:
        return None
    first = records[0]
    offsets: list[tuple[int, int]] = []
    parts: list[str] = []
    start = 0
    for record in records:
        text = _strip_overlap(parts[-1], record.text) if parts else record.text
        offsets.append((record.chunk_index, start))
        parts.append(text)
        start += len(text) + 1
    return DocumentContext(
        doc_id=first.doc_id,
        path=first.path,
        title=first.title,
        url=first.url,
        source_format=first.source_format,
        chunk_total=first.chunk_total,
        chunk_range=(first.chunk_index, records[-1].chunk_index),
        chunk_offsets=offsets,
        text="\n".join(parts),
    )