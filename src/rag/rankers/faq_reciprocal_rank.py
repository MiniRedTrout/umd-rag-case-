from collections import defaultdict
from collections.abc import Sequence

from src.rag.faq import FAQRecord


def faq_reciprocal_rank_fusion(
    results: Sequence[Sequence[FAQRecord]], k: int = 60
) -> list[FAQRecord]:
    """Fuse ranked FAQ record lists by reciprocal rank.

    Args:
        results: Sequence[Sequence[FAQRecord]] — Ranked candidate lists.
        k: int — Rank damping constant.

    Returns:
        list[FAQRecord] — Distinct FAQ records in fused rank order.
    """
    score: dict[str, float] = defaultdict(float)
    for ranking in results:
        for rank, record in enumerate(ranking, start=1):
            if not record.id:
                continue
            score[record.id] += 1 / (rank + k)

    seen: set[str] = set()
    unique: list[FAQRecord] = []
    for ranking in results:
        for record in ranking:
            if not record.id or record.id in seen:
                continue
            seen.add(record.id)
            unique.append(record)

    unique.sort(key=lambda record: score[record.id], reverse=True)
    return unique
