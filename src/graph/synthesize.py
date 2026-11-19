from __future__ import annotations

import re
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langfuse import get_client  # <-- LANGFUSE

from src.core.config.agent_config import get_agent_config
from src.graph.memory import remember, TurnMemory
from src.graph.prompt import get_agent_system_prompt  # <-- ВАШ ПРОМПТ
from src.graph.state import ChatGraphState
from src.memory.history import persist_chat_history as _persist_chat_history_impl
from src.models.errors import is_content_filtered
from src.models.llms.llm_factory import LLMFactory
from src.models.llms.request_policy import (
    InputProvenance,
    LLMRequestPolicy,
    OutputAudience,
)
from src.models.usage_callback import UsageCollector
from src.schemas.events import (
    FinishReason,
    classify_exception,
    error_event,
    message_delta_event,
    message_end_event,
    message_start_event,
    public_exception_message,
)
from src.schemas.query import LLMParams
from src.schemas.usage import emitted_usage, max_usage, merge_usage, zero_usage
from src.utils.logger import logger
from src.utils.metrics import record_chat, record_generation, snapshot as metrics_snapshot

NOT_FOUND_TEXT = "К сожалению, информация не найдена."

USER_PROMPT = """Вопрос: {question}

Фрагменты документов:
{context}"""

_CITATION_RE = re.compile(r"\[(\d+)\]")


async def _persist_chat_history(
    *,
    state: ChatGraphState,
    question: str,
    answer: str,
    model_name: str,
    finish_reason: str,
    usage: dict[str, int],
) -> None:
    conversation_id = str(state.get("conversation_id") or "").strip()
    if not conversation_id:
        return
    try:
        await _persist_chat_history_impl(
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            model_name=model_name,
            finish_reason=finish_reason,
            usage=usage,
        )
    except Exception:
        logger.exception("[chat_history] failed to persist (non-fatal)")


def make_synthesize_node():
    async def synthesize_node(
        state: ChatGraphState, config: RunnableConfig = None
    ) -> dict[str, Any]:
        writer = _writer()
        total_usage = dict(state.get("total_usage") or zero_usage())
        peak_usage = dict(state.get("peak_usage") or zero_usage())
        context = state.get("context") or []

        logger.info(
            f"[synthesize] start: context_blocks={len(context)}, "
            f"model={state.get('model_name')}, "
            f"conversation_id={state.get('conversation_id')!r}"
        )

        if not context:
            record_chat("not_found")
            await _emit_fixed(
                writer, NOT_FOUND_TEXT, state, total_usage, peak_usage
            )
            return {
                "answer_text": NOT_FOUND_TEXT,
                "finish_reason": "stop",
                "references": [],
                "total_usage": total_usage,
                "peak_usage": peak_usage,
            }

        payload = {
            "question": state.get("question", ""),
            "context": _format_context(context),
        }

        llm_params = LLMParams(
            model_name=state.get("model_name", ""),
            **(state.get("llm_params") or {}),
        )
        synthesis = get_agent_config().synthesis
        llm_params.max_tokens = max(
            int(llm_params.max_tokens or 0), synthesis.answer_max_tokens
        )
        
        llm_params.temperature = 0.1 
        
        collector = UsageCollector()
        model = LLMFactory(
            model_name=llm_params.model_name,
            llm_params=llm_params,
            request_policy=LLMRequestPolicy(
                input_provenance=InputProvenance.END_USER,
                output_audience=OutputAudience.END_USER,
            ),
            model_kwargs={"stream_options": {"include_usage": True}},
        ).model

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", get_agent_system_prompt()),
                ("human", USER_PROMPT),
            ]
        )
        chain = (prompt | model).with_config({"callbacks": [collector]})

        logger.info(
            f"[synthesize] calling LLM: "
            f"prompt_chars={len(payload['context']) + len(payload['question'])}, "
            f"max_tokens={llm_params.max_tokens}"
        )

        answer = ""
        started = False
        first_token_seen = False
        chunk_count = 0
        finish_reason: FinishReason = "stop"
        generation_started: float | None = None

        langfuse = get_client()
        
        with langfuse.start_as_current_observation(
            as_type="generation", 
            name="synthesize_llm",
            model=llm_params.model_name,
            input={"question": payload["question"], "context_length": len(payload["context"])}
        ) as generation:
            try:
                async for chunk in chain.astream(payload, config=config):
                    chunk_count += 1
                    text = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if not text:
                        continue
                    if not first_token_seen:
                        first_token_seen = True
                        generation_started = time.monotonic()
                        logger.info("[synthesize] first token received from LLM")
                    if not started:
                        started = True
                        writer(message_start_event())
                    answer += text
                    writer(message_delta_event(text=text))

            except Exception as exc:
                error_type, provider_status = classify_exception(exc)
                if not is_content_filtered(exc):
                    logger.exception("[synthesize] answer generation failed")
                finish_reason = (
                    "content_filter"
                    if error_type == "content_filter_error"
                    else "error"
                )
                if not started:
                    started = True
                    writer(message_start_event())
                writer(
                    error_event(
                        error_type=error_type,
                        message=public_exception_message(exc),
                        recoverable=False,
                        provider_status=provider_status,
                    )
                )

            generation.update(
                output=answer,
                usage_details={
                    "input": collector.total.get("input_tokens", 0),
                    "output": collector.total.get("output_tokens", 0),
                }
            )

        total_usage = merge_usage(total_usage, collector.total)
        peak_usage = max_usage(peak_usage, collector.peak)

        if generation_started is not None:
            record_generation(
                seconds=time.monotonic() - generation_started,
                usage=total_usage,
            )

        if finish_reason == "content_filter":
            record_chat("content_filter")
        elif finish_reason == "error":
            record_chat("error")
        elif not answer.strip() and finish_reason == "stop":
            record_chat("not_found")
        else:
            record_chat("answered")

        references = _build_references(context, answer)

        logger.info(
            f"[synthesize] done: chunks={chunk_count}, "
            f"answer_chars={len(answer)}, finish_reason={finish_reason}, "
            f"references={len(references)}, "
            f"usage={emitted_usage(total_usage, peak_usage)}; "
            f"metrics={metrics_snapshot()}"
        )

        if not answer.strip() and finish_reason == "stop":
            logger.error("[synthesize] the model returned no answer")
            finish_reason = "error"
            if not started:
                writer(message_start_event())
            writer(
                error_event(
                    error_type="api_error",
                    message="Модель не вернула ответ.",
                    recoverable=False,
                )
            )

        answered = finish_reason in ("stop", "length")

        await _persist_chat_history(
            state=state,
            question=state.get("question", ""),
            answer=answer if answered else "",
            model_name=state.get("model_name", ""),
            finish_reason=finish_reason,
            usage=emitted_usage(total_usage, peak_usage),
        )

        if answered:
            await remember(
                str(state.get("conversation_id") or "").strip(),
                TurnMemory(question=state.get("question", ""), answer=answer),
            )

        writer(
            message_end_event(
                text=answer,
                finish_reason=finish_reason,
                references=references,
                usage=emitted_usage(total_usage, peak_usage),
            )
        )

        return {
            "answer_text": answer,
            "finish_reason": finish_reason,
            "references": references,
            "total_usage": total_usage,
            "peak_usage": peak_usage,
        }

    return synthesize_node


def _writer():
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except RuntimeError:
        return lambda _payload: None


async def _emit_fixed(writer, text, state, total_usage, peak_usage) -> None:
    conv_id = str(state.get("conversation_id") or "").strip()
    if conv_id:
        await remember(
            conv_id, TurnMemory(question=state.get("question", ""), answer=text)
        )
    writer(message_start_event())
    writer(message_delta_event(text=text))
    writer(
        message_end_event(
            text=text,
            finish_reason="stop",
            usage=emitted_usage(total_usage, peak_usage),
        )
    )


def _format_context(context: list[dict[str, Any]]) -> str:
    """Render context blocks, each prefixed with ``[N]`` for citation."""
    blocks: list[str] = []
    for position, entry in enumerate(context, 1):
        title = entry.get("title") or entry.get("doc_id") or ""
        text = str(entry.get("text") or "")
        blocks.append(f"[{position}] ### {title}\n{text}")
    return "\n\n———\n\n".join(blocks)


def _build_references(
    context: list[dict[str, Any]], answer: str
) -> list[dict[str, Any]]:
    """Return the context entries the answer actually cited, in ``[N]`` order."""
    if not context:
        return []
    cited: set[int] = set()
    for match in _CITATION_RE.finditer(answer or ""):
        try:
            index = int(match.group(1))
        except ValueError:
            continue
        if 1 <= index <= len(context):
            cited.add(index)
    if not cited:
        return []
    references: list[dict[str, Any]] = []
    for index in sorted(cited):
        entry = context[index - 1]
        references.append(
            {
                "cite_key": f"c{index}",
                "index": index,
                "doc_id": entry.get("doc_id", ""),
                "title": entry.get("title") or entry.get("doc_id", ""),
                "url": entry.get("url", ""),
                "path": entry.get("path", ""),
                "chunk_range": entry.get("chunk_range", []),
            }
        )
    return references