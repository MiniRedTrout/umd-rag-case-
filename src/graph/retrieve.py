from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from src.core.config.agent_config import get_agent_config
from src.core.config.llm_config import llm_settings
from src.graph.state import ChatGraphState
from src.schemas.events import status_event          # NEW
from src.utils.logger import logger
from src.utils.metrics import record_retrieval


DEFAULT_CONTEXT_BUDGET_CHARS = 8000


def _writer():
    """Return the LangGraph stream writer or a no-op (same as in synthesize)."""
    try:
        from langgraph.config import get_stream_writer
        return get_stream_writer()
    except RuntimeError:
        return lambda _payload: None


def _trim_context_by_budget(
    contexts: list[dict[str, Any]], budget_chars: int
) -> list[dict[str, Any]]:
    """Keep the best blocks in order until the budget is spent."""
    kept: list[dict[str, Any]] = []
    used = 0
    for entry in contexts:
        text = str(entry.get("text") or "")
        if not text:
            continue
        remaining = budget_chars - used
        if remaining <= 200:
            break
        if len(text) <= remaining:
            kept.append(entry)
            used += len(text)
        else:
            cut = dict(entry)
            cut["text"] = text[:remaining].rstrip() + "…"
            kept.append(cut)
            used += len(cut["text"])
            break
    return kept


def make_retrieve_node():
    async def retrieve_node(
        state: ChatGraphState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        writer = _writer()                            # NEW
        question = (state.get("question") or "").strip()
        if not question:
            return {"context": [], "references": []}

        retrieval = get_agent_config().retrieval
        synthesis = get_agent_config().synthesis

        writer(status_event(text="Ищу релевантные фрагменты в базе знаний…"))  # NEW

        try:
            records = await llm_settings.REPOSITORY.search(
                [question], limit=retrieval.chunks_to_retrieve
            )
        except Exception:
            logger.exception("[retrieve] search failed")
            record_retrieval(0)
            writer(status_event(text="Поиск не удался, отвечаю без контекста."))  # NEW
            return {"context": [], "references": []}

        record_retrieval(len(records))
        if not records:
            writer(status_event(text="Ничего не нашёл по этому вопросу."))  # NEW
            return {"context": [], "references": []}

        writer(status_event(text=f"Нашёл {len(records)} фрагментов, отбираю лучшие…"))  # NEW
        first_rank: dict[str, int] = {}
        by_doc: dict[str, list] = {}
        for rank, record in enumerate(records):
            by_doc.setdefault(record.doc_id, []).append(record)
            first_rank.setdefault(record.doc_id, rank)

        ordered_docs = sorted(
            by_doc.items(),
            key=lambda item: first_rank[item[0]],
        )[: synthesis.max_context_documents]

        selected = []
        for _, doc_records in ordered_docs:
            selected.extend(doc_records)

        try:
            contexts = await llm_settings.REPOSITORY.document_context(
                [(r.path, r.chunk_index) for r in selected]
            )
        except Exception:
            logger.exception("[retrieve] context expansion failed")
            writer(status_event(text="Не удалось собрать контекст."))  # NEW
            return {"context": [], "references": []}

        context: list[dict[str, Any]] = []
        for index, document in enumerate(contexts, 1):
            if not document.doc_id:
                continue
            context.append(
                {
                    "cite_key": f"c{index}",
                    "doc_id": document.doc_id,
                    "title": document.title or document.doc_id,
                    "url": document.url,
                    "path": document.path,
                    "chunk_range": list(document.chunk_range),
                    "text": document.text,
                    "rank": index,
                }
            )

        budget = getattr(synthesis, "max_context_chars", DEFAULT_CONTEXT_BUDGET_CHARS)
        context = _trim_context_by_budget(context, budget)

        used_chars = sum(len(c["text"]) for c in context)
        logger.info(
            f"[retrieve] {len(records)} hits → {len(context)} in context "
            f"(budget {budget} chars, used {used_chars} chars)"
        )
        writer(status_event(
            text=f"В контексте {len(context)} фрагментов ({used_chars} симв.)."
        ))  
        return {"context": context, "references": []}

    return retrieve_node