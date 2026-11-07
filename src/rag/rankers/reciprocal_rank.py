from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from src.rag.chunks import ChunkRecord


def unique_by_key(
    records: Iterable[ChunkRecord],
    *,
    seen: set[str] | None = None,
) -> list[ChunkRecord]:
    tracked = seen if seen is not None else set()
    kept: list[ChunkRecord] = []
    for record in records:
        key = record.chunk_key
        if not key or key in tracked:
            continue
        tracked.add(key)
        kept.append(record)
    return kept


def reciprocal_rank_fusion(
    results: Sequence[Sequence[ChunkRecord]], k: int = 60
) -> list[ChunkRecord]:
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    by_key: dict[str, ChunkRecord] = {}
    order = 0

    for ranking in results:
        for rank, record in enumerate(ranking, start=1):
            key = record.chunk_key
            if not key:
                continue
            scores[key] += 1.0 / (rank + k)
            if key not in by_key:
                by_key[key] = record
                first_seen[key] = order
                order += 1

    return sorted(
        by_key.values(),
        key=lambda record: (-scores[record.chunk_key], first_seen[record.chunk_key]),
    )