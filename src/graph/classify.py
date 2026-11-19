from __future__ import annotations

import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field

from src.graph.prompt import get_classify_prompt  # <-- ИМПОРТ ПРОМПТА
from src.graph.state import ChatGraphState
from src.models.llms.llm_factory import LLMFactory
from src.models.llms.request_policy import (
    InputProvenance,
    LLMRequestPolicy,
    OutputAudience,
)
from src.schemas.query import LLMParams
from src.utils.logger import logger


_GREETING_WORDS: tuple[str, ...] = (
    "привет",
    "приветик",
    "здравствуй",
    "здравствуйте",
    "добрый день",
    "добрый вечер",
    "доброе утро",
    "доброго дня",
    "спасибо",
    "благодарю",
    "пока",
    "до свидания",
    "как дела",
    "как ты",
    "hi",
    "hello",
    "thanks",
    "thank you",
    "bye",
)

_GREETING_ALT = "|".join(re.escape(word) for word in _GREETING_WORDS)

_PURE_SMALLTALK_RE = re.compile(
    rf"^\s*(?:{_GREETING_ALT})\s*[!?.,…\-]*\s*$",
    re.IGNORECASE,
)

_GREETING_PREFIX_RE = re.compile(
    rf"^\s*(?:{_GREETING_ALT})\s*[!?.,…\-]+\s*\S",
    re.IGNORECASE,
)

_SHORT_GREETING_MAX_WORDS = 5


def _looks_like_smalltalk(question: str) -> bool:
    if not question:
        return False
    if _PURE_SMALLTALK_RE.match(question):
        return True
    if _GREETING_PREFIX_RE.match(question):
        return False
    if len(question.split()) > _SHORT_GREETING_MAX_WORDS:
        return False
    return False


class ClassifiedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str = Field(description="smalltalk | question")
    answer: str = Field(
        default="",
        description=(
            "Если kind = smalltalk — короткий ответ пользователю в 1–2 предложения. "
            "Если kind = question — пустая строка."
        ),
    )


def make_classify_node():
    async def classify_node(state: ChatGraphState, config=None) -> dict[str, Any]:
        question = (state.get("question") or "").strip()
        if not question:
            return {"request_kind": "question"}

        if _looks_like_smalltalk(question):
            logger.info(f"[classify] regex smalltalk: {question!r}")
            return {"request_kind": "smalltalk", "smalltalk_question": question}

        llm_params = LLMParams(
            model_name=state.get("model_name", ""),
            **(state.get("llm_params") or {}),
        )
        llm_params.max_tokens = 200
        llm_params.temperature = 0.0
        
        model = LLMFactory(
            model_name=llm_params.model_name,
            llm_params=llm_params,
            request_policy=LLMRequestPolicy(
                input_provenance=InputProvenance.END_USER,
                output_audience=OutputAudience.INTERNAL,
            ),
        ).model

        prompt = ChatPromptTemplate.from_messages([
            ("system", get_classify_prompt()),
            ("human", "Сообщение: {question}"),
        ])

        try:
            chain = prompt | model.with_structured_output(ClassifiedRequest)
            result = await chain.ainvoke({"question": question}, config=config)
            kind = result.kind if result.kind in ("smalltalk", "question") else "question"
            answer = result.answer.strip() if kind == "smalltalk" else ""
        except Exception:
            logger.exception("[classify] failed, defaulting to question")
            kind = "question"
            answer = ""

        logger.info(f"[classify] kind={kind}, question={question!r}")
        out: dict[str, Any] = {"request_kind": kind}
        if kind == "smalltalk":
            out["smalltalk_question"] = question
            out["smalltalk_answer"] = answer
        return out

    return classify_node