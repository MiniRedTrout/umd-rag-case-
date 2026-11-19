import uuid

from fastapi import APIRouter, HTTPException
from langfuse import get_client, propagate_attributes

from src.rag.agent.orchestrator import run_orchestrator_agent
from src.schemas import QueryChat
from src.schemas.agent_trace import AgentTraceResponse
from src.services.chat_service import prepare_chat_messages
from src.utils.logger import logger

router = APIRouter()


@router.post(
    "/trace",
    status_code=200,
    response_model=AgentTraceResponse,
    summary="Run agent and return structured system steps",
)
async def trace_agent_answer(query_extended: QueryChat) -> AgentTraceResponse:
    logger.info("Received structured trace query via agent")

    if len(query_extended.messages) == 0:
        logger.error("No messages in trace request")
        raise HTTPException(status_code=400, detail="No messages in request")

    try:
        langchain_messages = await prepare_chat_messages(query_extended)
    except ValueError as e:
        logger.warning(f"Rejected client message: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e

    llm_params = query_extended.model_dump(include={"temperature", "top_p", "max_tokens", "repetition_penalty"})
    trace_id = uuid.uuid4().hex[:12]

    langfuse = get_client()

    with propagate_attributes(
        trace_name="agent_trace_endpoint",
        session_id=query_extended.conversation_id or trace_id,
    ):
        with langfuse.start_as_current_observation(
            as_type="span",
            name="run_orchestrator_agent",
            input={"messages": [m.content for m in langchain_messages[-2:]]},  # Логируем последние сообщения
        ) as span:
            
            result = await run_orchestrator_agent(
                messages=langchain_messages,
                model_name=query_extended.model_name,
                trace_id=trace_id,
                **llm_params,
            )
            
            span.update(output=result.answer)

    return result