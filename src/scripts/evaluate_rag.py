"""Evaluate indexed-document retrieval against ``real.json``.

Example:
    python -m src.scripts.evaluate_rag --dataset real.json --output eval.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from src.evaluation.rag_metrics import aggregate_metrics, evaluate_query
from src.rag.search import DEFAULT_STRATEGIES, SearchStrategy


def _load_items(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    items: list[dict[str, Any]] = []
    for file_entry in payload.get("files", []):
        source = str(file_entry.get("source") or "")
        for item in file_entry.get("items", []):
            item = dict(item)
            item["_expected_source"] = source
            items.append(item)
    return items


async def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    # Imported lazily so metric helpers remain usable without OpenSearch or LLM settings.
    from src.core.config.llm_config import llm_settings

    repository = llm_settings.REPOSITORY
    strategies = tuple(SearchStrategy(value) for value in args.strategies)
    items = _load_items(Path(args.dataset))
    if args.limit:
        items = items[: args.limit]

    results = []
    try:
        for item in items:
            records = await repository.search(
                [str(item.get("question") or "")],
                limit=max(args.k),
                strategies=strategies,
                per_strategy=args.per_strategy,
            )
            results.append(
                evaluate_query(
                    question_id=str(item.get("id") or ""),
                    expected_source=str(item["_expected_source"]),
                    records=records,
                    ks=args.k,
                )
            )
    finally:
        await llm_settings.client.close()

    return {
        "dataset": str(args.dataset),
        "strategies": list(args.strategies),
        "k": args.k,
        "metrics": aggregate_metrics(results),
        "queries": [
            {
                "id": result.question_id,
                "question": next(
                    item["question"]
                    for item in items
                    if str(item.get("id") or "") == result.question_id
                ),
                "reference_answer": next(
                    item.get("reference_answer", "")
                    for item in items
                    if str(item.get("id") or "") == result.question_id
                ),
                "expected_source": result.expected_source,
                "retrieved_sources": list(result.retrieved_sources),
                "retrieved_contexts": list(result.retrieved_contexts),
                "reciprocal_rank": result.reciprocal_rank,
                "hit_at": result.hit_at,
                "precision_at": result.precision_at,
            }
            for result in results
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("real.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation-results.json"))
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--per-strategy", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=[strategy.value for strategy in SearchStrategy],
        default=[strategy.value for strategy in DEFAULT_STRATEGIES],
    )
    args = parser.parse_args()
    if any(k < 1 for k in args.k):
        parser.error("--k values must be positive")

    report = asyncio.run(_evaluate(args))
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

