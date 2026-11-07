

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Metrics:
    indexing_runs: int = 0
    indexing_failures: int = 0
    indexing_seconds_total: float = 0.0
    indexing_seconds_last: float = 0.0
    indexed_chunks_total: int = 0
    indexed_documents_total: int = 0
    seconds_per_chunk_last: float = 0.0
    seconds_per_chunk_total: float = 0.0
    seconds_per_chunk_count: int = 0

    chat_requests: int = 0
    chat_answered: int = 0
    chat_not_found: int = 0
    chat_errors: int = 0
    chat_content_filtered: int = 0

    generation_seconds_total: float = 0.0
    generation_seconds_last: float = 0.0
    generation_calls: int = 0
    tokens_per_second_last: float = 0.0

    input_tokens_total: int = 0
    output_tokens_total: int = 0

    retrieval_calls: int = 0
    retrieval_hits_total: int = 0

    _lock: threading.Lock = field(default_factory=threading.Lock)

_metrics = _Metrics()


def _avg(total: float, count: int) -> float:
    return total / count if count else 0.0


def record_indexing(
    *, seconds: float, chunks: int, documents: int, failed: bool
) -> None:
    with _metrics._lock:
        _metrics.indexing_runs += 1
        _metrics.indexing_seconds_total += seconds
        _metrics.indexing_seconds_last = seconds
        if failed:
            _metrics.indexing_failures += 1
        else:
            _metrics.indexed_chunks_total += chunks
            _metrics.indexed_documents_total += documents
    if not failed and chunks > 0:
        _metrics.seconds_per_chunk_last = seconds / chunks
        _metrics.seconds_per_chunk_total += seconds
        _metrics.seconds_per_chunk_count += 1


def record_retrieval(hits: int) -> None:
    with _metrics._lock:
        _metrics.retrieval_calls += 1
        _metrics.retrieval_hits_total += hits



def record_chat(outcome: str) -> None:
    """outcome: answered | not_found | error | content_filter"""
    with _metrics._lock:
        _metrics.chat_requests += 1
        if outcome == "answered":
            _metrics.chat_answered += 1
        elif outcome == "not_found":
            _metrics.chat_not_found += 1
        elif outcome == "content_filter":
            _metrics.chat_content_filtered += 1
        else:
            _metrics.chat_errors += 1


def record_generation(*, seconds: float, usage: dict[str, int]) -> None:
    with _metrics._lock:
        _metrics.generation_calls += 1
        _metrics.generation_seconds_total += seconds
        _metrics.generation_seconds_last = seconds
        _metrics.input_tokens_total += int(usage.get("input_tokens") or 0)
        _metrics.output_tokens_total += int(usage.get("output_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    if seconds > 0 and output > 0:
        _metrics.tokens_per_second_last = output / seconds


def snapshot() -> dict[str, float | int]:
    with _metrics._lock:
        return {
            "indexing_runs": _metrics.indexing_runs,
            "indexing_failures": _metrics.indexing_failures,
            "indexing_seconds_last": round(_metrics.indexing_seconds_last, 2),
            "indexing_seconds_avg": round(
                _avg(_metrics.indexing_seconds_total, _metrics.indexing_runs), 2
            ),
            "indexed_chunks_total": _metrics.indexed_chunks_total,
            "indexed_documents_total": _metrics.indexed_documents_total,
            "chat_requests": _metrics.chat_requests,
            "chat_answered": _metrics.chat_answered,
            "chat_not_found": _metrics.chat_not_found,
            "chat_errors": _metrics.chat_errors,
            "chat_content_filtered": _metrics.chat_content_filtered,
            "generation_calls": _metrics.generation_calls,
            "generation_seconds_last": round(_metrics.generation_seconds_last, 2),
            "generation_seconds_avg": round(
                _avg(_metrics.generation_seconds_total, _metrics.generation_calls),
                2,
            ),
            "input_tokens_total": _metrics.input_tokens_total,
            "output_tokens_total": _metrics.output_tokens_total,
            "retrieval_calls": _metrics.retrieval_calls,
            "retrieval_hits_avg": round(
                _avg(_metrics.retrieval_hits_total, _metrics.retrieval_calls), 2
            ),
            "seconds_per_chunk_last": round(_metrics.seconds_per_chunk_last, 3),
            "seconds_per_chunk_avg": round(
                _avg(_metrics.seconds_per_chunk_total, _metrics.seconds_per_chunk_count),
                3,
            ),
            "tokens_per_second_last": round(_metrics.tokens_per_second_last, 1),
            "answered_ratio": round(
                _metrics.chat_answered / _metrics.chat_requests
                if _metrics.chat_requests
                else 0,
                3,
            ),
            "not_found_ratio": round(
                _metrics.chat_not_found / _metrics.chat_requests
                if _metrics.chat_requests
                else 0,
                3,
            ),
        }