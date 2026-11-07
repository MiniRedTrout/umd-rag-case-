from typing import Any

import httpx
import openai
from langchain_openai import OpenAIEmbeddings
from loguru import logger
from pydantic import PrivateAttr

from src.models.decorators import async_retry_on_error, retry_on_error
from src.models.embedders.size_fallback import (
    EmbeddingStats,
    aembed_with_size_fallback,
    embed_with_size_fallback,
)
from src.security.ssl_manager import ssl_manager
from src.utils.env_wrapper import get_env_var, get_optional_env_var


_EMBEDDING_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=200.0,
    write=30.0,
    pool=10.0,
)


class GPNOpenAIEmbeddings(OpenAIEmbeddings):
    _embedding_stats: EmbeddingStats = PrivateAttr(default_factory=EmbeddingStats)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._api_key_retries = 3

    @property
    def embedding_stats(self) -> EmbeddingStats:
        return self._embedding_stats

    @property
    def model_label(self) -> str:
        return str(self.model)

    @property
    def api_key_retries(self) -> int:
        return self._api_key_retries

    @api_key_retries.setter
    def api_key_retries(self, value: int) -> None:
        self._api_key_retries = value

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
        self.client = openai.OpenAI(**client_params, **sync_specific).embeddings  # type: ignore[arg-type]
        async_specific = {"http_client": self.http_async_client}
        self.async_client = openai.AsyncOpenAI(
            **client_params, **async_specific  # type: ignore[arg-type]
        ).embeddings

    def _update_api_key(self) -> None:
        logger.warning("Embedding API key refresh is not configured")

    def _embed_batch(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return super().embed_documents(texts, **kwargs)

    async def _aembed_batch(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return await super().aembed_documents(texts, **kwargs)

    @retry_on_error()
    def embed_documents(
        self, texts: list[str], *, chunk_keys: list[str] | None = None, **kwargs: Any
    ) -> list[list[float]]:
        return embed_with_size_fallback(
            texts,
            lambda batch: self._embed_batch(batch, **kwargs),
            model=self.model_label,
            chunk_keys=chunk_keys,
            stats=self._embedding_stats,
        )

    @retry_on_error()
    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        embeddings = embed_with_size_fallback(
            [text],
            lambda batch: self._embed_batch(batch, **kwargs),
            model=self.model_label,
            chunk_keys=["<query>"],
            stats=self._embedding_stats,
        )
        return embeddings[0] if embeddings else []

    @async_retry_on_error()
    async def aembed_documents(
        self, texts: list[str], *, chunk_keys: list[str] | None = None, **kwargs: Any
    ) -> list[list[float]]:
        return await aembed_with_size_fallback(
            texts,
            lambda batch: self._aembed_batch(batch, **kwargs),
            model=self.model_label,
            chunk_keys=chunk_keys,
            stats=self._embedding_stats,
        )

    @async_retry_on_error()
    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        embeddings = await aembed_with_size_fallback(
            [text],
            lambda batch: self._aembed_batch(batch, **kwargs),
            model=self.model_label,
            chunk_keys=["<query>"],
            stats=self._embedding_stats,
        )
        return embeddings[0] if embeddings else []

    @async_retry_on_error()
    async def aembed_queries(
        self, texts: list[str], *, chunk_keys: list[str] | None = None, **kwargs: Any
    ) -> list[list[float]]:
        return await self.aembed_documents(texts, chunk_keys=chunk_keys, **kwargs)


def load_embeddings(model_name: str, base_url: str | None = None):
    endpoint = base_url or get_env_var("VLLM_EMB_URL")
    return GPNOpenAIEmbeddings(
        base_url=endpoint,
        model=model_name,
        openai_api_key=get_optional_env_var("VLLM_API_KEY") or "not-needed",
        http_client=httpx.Client(
            verify=ssl_manager.embedding if ssl_manager.embedding is not None else False,
            timeout=_EMBEDDING_TIMEOUT,
        ),
        http_async_client=httpx.AsyncClient(
            verify=ssl_manager.embedding if ssl_manager.embedding is not None else False,
            timeout=_EMBEDDING_TIMEOUT,
        ),
        check_embedding_ctx_length=False,
    )