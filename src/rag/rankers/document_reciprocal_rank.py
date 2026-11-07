from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from langchain_core.documents import Document


def document_reciprocal_rank_fusion(
    results: Sequence[Sequence[Document]], k: int = 60
) -> list[Document]:
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    by_text: dict[str, Document] = {}
    order = 0

    for ranking in results:
        for rank, document in enumerate(ranking, start=1):
            text = document.page_content
            if not text:
                continue
            scores[text] += 1.0 / (rank + k)
            if text not in by_text:
                by_text[text] = document
                first_seen[text] = order
                order += 1

    return sorted(
        by_text.values(),
        key=lambda document: (
            -scores[document.page_content],
            first_seen[document.page_content],
        ),
    )