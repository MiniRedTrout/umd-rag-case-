from __future__ import annotations

from collections.abc import Iterable

from src.core.config.agent_config import get_agent_config


def clean_queries(queries: Iterable[object]) -> list[str]:
    """Normalize, deduplicate, and limit queries."""
    retrieval = get_agent_config().retrieval
    seen: set[str] = set()
    cleaned: list[str] = []
    for query in queries:
        text = " ".join(str(query or "").split())[: retrieval.max_retrieval_query_chars]
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            cleaned.append(text)
    return cleaned[: retrieval.max_retrieval_queries]