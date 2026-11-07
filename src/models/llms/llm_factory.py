from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

from src.core import settings
from src.models.api_credentials import (
    build_chat_extra_body,
    get_api_key,
    get_default_headers,
)
from src.models.decorators import (
    async_generator_retry_on_error,
    async_retry_on_error,
    generator_retry_on_error,
    retry_on_error,
)
from src.models.llms.request_policy import LLMRequestPolicy
from src.schemas import LLMGenerationParams
from src.security.ssl_manager import ssl_manager
from src.utils.logger import logger
from src.utils.request_context import REQUEST_ID_HEADER, get_request_id

SDK_MAX_RETRIES = 0
CHAT_REQUEST_TIMEOUT_S = 120


class GPNChatOpenAI(ChatOpenAI):
    def __init__(self, *args, **kwargs):
        kwargs["max_retries"] = SDK_MAX_RETRIES
        super().__init__(*args, **kwargs)
        self.max_retries = SDK_MAX_RETRIES

    def _update_clients(self) -> None:
        client_params: dict = {
            "api_key": (self.openai_api_key if self.openai_api_key else None),
            "organization": self.openai_organization,
            "base_url": self.openai_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        sync_specific = {"http_client": self.http_client}
        self.root_client = openai.OpenAI(**client_params, **sync_specific)  # type: ignore[arg-type]
        self.client = self.root_client.chat.completions
        async_specific = {"http_client": self.http_async_client}
        self.root_async_client = openai.AsyncOpenAI(
            **client_params, **async_specific  # type: ignore[arg-type]
        )
        self.async_client = self.root_async_client.chat.completions

    def _update_api_key(self) -> None:
        logger.warning("LLM API key refresh is not configured")

    @generator_retry_on_error()
    def _stream(
        self, *args: Any, stream_usage: bool | None = None, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        stream_usage = self._should_stream_usage(stream_usage, **kwargs)
        if stream_usage:
            kwargs["stream_options"] = {"include_usage": stream_usage}
        return super()._stream(*args, **kwargs)

    @retry_on_error()
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return super()._generate(
            messages=messages, stop=stop, run_manager=run_manager, **kwargs
        )

    @async_generator_retry_on_error()
    async def _astream(
        self, *args: Any, stream_usage: bool | None = None, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        stream_usage = self._should_stream_usage(stream_usage, **kwargs)
        if stream_usage:
            kwargs["stream_options"] = {"include_usage": stream_usage}
        async for chunk in super()._astream(*args, **kwargs):
            yield chunk

    @async_retry_on_error()
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await super()._agenerate(
            messages=messages, stop=stop, run_manager=run_manager, **kwargs
        )


class LLMFactory:
    """Create configured large language model instances."""

    def __init__(
        self,
        model_name: str,
        llm_params: LLMGenerationParams,
        *,
        request_policy: LLMRequestPolicy,
        enable_reasoning: bool = False,
        **kwargs,
    ) -> None:
        params = llm_params.model_dump(include={"temperature", "top_p", "max_tokens"})
        default_headers = get_default_headers(
            settings.VLLM_LLM_URL, request_policy=request_policy
        )
        if (request_id := get_request_id()) is not None:
            default_headers = {**(default_headers or {}), REQUEST_ID_HEADER: request_id}

        self.model: BaseChatModel | LLM = GPNChatOpenAI(
            model=model_name,
            verbose=True,
            api_key=get_api_key(settings.VLLM_LLM_URL),
            base_url=settings.VLLM_LLM_URL,
            default_headers=default_headers,
            http_client=httpx.Client(
                verify=ssl_manager.llm if ssl_manager.llm is not None else False
            ),
            http_async_client=httpx.AsyncClient(
                verify=ssl_manager.llm if ssl_manager.llm is not None else False,
                proxy="http://127.0.0.1:10809",  
            ),
            timeout=CHAT_REQUEST_TIMEOUT_S,
            extra_body=build_chat_extra_body(
                settings.VLLM_LLM_URL,
                model_name,
                enable_reasoning=enable_reasoning,
                request_policy=request_policy,
            ),
            **params,
            **kwargs,
        )