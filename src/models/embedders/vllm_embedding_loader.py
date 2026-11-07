from __future__ import annotations

from typing import Any, TYPE_CHECKING

import httpx
from loguru import logger
from langchain_core.embeddings import Embeddings

from src.models.decorators import async_retry_on_error, retry_on_error
from src.security.ssl_manager import ssl_manager
from src.utils.embedding_token_limit import EmbeddingTokenLimiter, is_request_too_large, text_preview
from src.utils.env_wrapper import get_env_var, get_optional_env_var
from src.models.embedders.embedding_loader import _EMBEDDING_TIMEOUT

if TYPE_CHECKING:
    from httpx import Response


def _openai_compatible_embeddings_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/embeddings"
    return f"{base}/v1/embeddings"


class GPNVLLMEmbeddings(Embeddings):
    """Embedding class for vLLM embedding endpoints (OpenAI-compatible)."""

    def __init__(
        self,
        api_base: str,
        model: str,
        api_key: str,
        *,
        http_client: httpx.Client,
        http_async_client: httpx.AsyncClient,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.http_client = http_client
        self.http_async_client = http_async_client
        self._api_key_retries = 3

    @property
    def api_key_retries(self) -> int:
        return self._api_key_retries

    @api_key_retries.setter
    def api_key_retries(self, value: int) -> None:
        self._api_key_retries = value

    def _update_api_key(self) -> None:
        logger.warning("vLLM embedding API key refresh is not configured")

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _token_limiter(self) -> EmbeddingTokenLimiter:
        return EmbeddingTokenLimiter(
            api_base=self.api_base,
            model=self.model,
            http_client=self.http_client,
            http_async_client=self.http_async_client,
            headers_factory=self._build_headers,
        )

    def _handle_response(self, response: Response) -> list[list[float]]:
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data.get("data", [])]  # type: ignore

    def _sync_request(self, payload: dict[str, Any]) -> list[list[float]]:
        url = _openai_compatible_embeddings_url(self.api_base)
        headers = self._build_headers()
        resp = self.http_client.post(url, json=payload, headers=headers)
        return self._handle_response(resp)

    async def _async_request(self, payload: dict[str, Any]) -> list[list[float]]:
        url = _openai_compatible_embeddings_url(self.api_base)
        headers = self._build_headers()
        resp = await self.http_async_client.post(url, json=payload, headers=headers)
        return self._handle_response(resp)

    @retry_on_error()
    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        limiter = self._token_limiter()
        payload = {"model": self.model, "input": texts}
        try:
            return self._sync_request(payload)
        except Exception as exc:
            if not is_request_too_large(exc):
                raise
            logger.warning("Embedding batch exceeded model context; retrying as individual texts")
            return self._embed_documents_individually(texts, limiter)

    @retry_on_error()
    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        result = self.embed_documents([text], **kwargs)
        return result[0] if result else []

    @async_retry_on_error()
    async def aembed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        limiter = self._token_limiter()
        payload = {"model": self.model, "input": texts}
        try:
            return await self._async_request(payload)
        except Exception as exc:
            if not is_request_too_large(exc):
                raise
            logger.warning("Embedding batch exceeded model context; retrying as individual texts")
            return await self._aembed_documents_individually(texts, limiter)

    @async_retry_on_error()
    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        docs = await self.aembed_documents([text], **kwargs)
        return docs[0] if docs else []

    def _embed_documents_individually(
        self, texts: list[str], limiter: EmbeddingTokenLimiter
    ) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for idx, text in enumerate(texts):
            try:
                embeddings.extend(self._sync_request({"model": self.model, "input": [text]}))
                continue
            except Exception as exc:
                if not is_request_too_large(exc):
                    raise
                last_exc = exc
                for attempt in range(limiter.fallback_attempts):
                    try:
                        embeddings.extend(
                            self._sync_request(
                                {
                                    "model": self.model,
                                    "input": limiter.character_limit_texts([text], attempt),
                                }
                            )
                        )
                        break
                    except Exception as fallback_exc:
                        if not is_request_too_large(fallback_exc):
                            raise
                        last_exc = fallback_exc
                else:
                    raise last_exc
        return embeddings

    async def _aembed_documents_individually(
        self, texts: list[str], limiter: EmbeddingTokenLimiter
    ) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for idx, text in enumerate(texts):
            try:
                embeddings.extend(await self._async_request({"model": self.model, "input": [text]}))
                continue
            except Exception as exc:
                if not is_request_too_large(exc):
                    raise
                last_exc = exc
                for attempt in range(limiter.fallback_attempts):
                    try:
                        embeddings.extend(
                            await self._async_request(
                                {
                                    "model": self.model,
                                    "input": limiter.character_limit_texts([text], attempt),
                                }
                            )
                        )
                        break
                    except Exception as fallback_exc:
                        if not is_request_too_large(fallback_exc):
                            raise
                        last_exc = fallback_exc
                else:
                    raise last_exc
        return embeddings


def load_vllm_embeddings(model_name: str, base_url: str | None = None):
    endpoint = base_url or get_env_var("VLLM_EMB_URL")
    api_key = get_optional_env_var("VLLM_API_KEY") or "not-needed"
    return GPNVLLMEmbeddings(
        model=model_name,
        api_base=endpoint,
        api_key=api_key,
        http_client=httpx.Client(
            verify=ssl_manager.embedding if ssl_manager.embedding is not None else False,
            timeout=_EMBEDDING_TIMEOUT,
        ),
        http_async_client=httpx.AsyncClient(
            verify=ssl_manager.embedding if ssl_manager.embedding is not None else False,
            timeout=_EMBEDDING_TIMEOUT,
        ),
    )