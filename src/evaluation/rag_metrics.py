from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.rag.chunks import ChunkRecord
from src.utils.slugify import slugify


def _source_key(value: str) -> str:
    """Build a stable comparison key for an indexed path or dataset source."""
    return slugify(Path(value).name)


def source_matches(record: ChunkRecord, expected_source: str) -> bool:
    return _source_key(record.path or record.doc_id) == _source_key(expected_source)


@dataclass(frozen=True)
class QueryMetric:
    question_id: str
    expected_source: str
    retrieved_sources: tuple[str, ...]
    retrieved_contexts: tuple[str, ...]
    reciprocal_rank: float
    hit_at: dict[int, bool]
    precision_at: dict[int, float]


def evaluate_query(
    question_id: str,
    expected_source: str,
    records: Sequence[ChunkRecord],
    ks: Iterable[int],
) -> QueryMetric:
    requested_ks = tuple(sorted(set(ks)))
    retrieved_sources = tuple(record.path or record.doc_id for record in records)
    first_rank = next(
        (rank for rank, record in enumerate(records, start=1) if source_matches(record, expected_source)),
        None,
    )
    return QueryMetric(
        question_id=question_id,
        expected_source=expected_source,
        retrieved_sources=retrieved_sources,
        retrieved_contexts=tuple(record.text for record in records),
        reciprocal_rank=1.0 / first_rank if first_rank else 0.0,
        hit_at={
            k: any(source_matches(record, expected_source) for record in records[:k])
            for k in requested_ks
        },
        precision_at={
            k: sum(source_matches(record, expected_source) for record in records[:k]) / k
            for k in requested_ks
        },
    )


def aggregate_metrics(results: Sequence[QueryMetric]) -> dict[str, float | int]:
    if not results:
        return {"queries": 0, "mrr": 0.0}

    ks = sorted(results[0].hit_at)
    metrics: dict[str, float | int] = {
        "queries": len(results),
        "mrr": sum(result.reciprocal_rank for result in results) / len(results),
    }
    for k in ks:
        metrics[f"recall_at_{k}"] = sum(result.hit_at[k] for result in results) / len(results)
        metrics[f"precision_at_{k}"] = sum(result.precision_at[k] for result in results) / len(results)
    return metrics

