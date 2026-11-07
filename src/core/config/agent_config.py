from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.utils.env_wrapper import get_optional_env_var
from src.utils.logger import logger

_STRICT = ConfigDict(extra="forbid")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve(config_path: str) -> Path:
    path = Path(config_path)
    return path.resolve() if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


class RetrievalConfig(BaseModel):
    """Define retrieval limits."""

    model_config = _STRICT

    max_retrieval_query_chars: int = Field(default=220, ge=20, le=2000)
    max_retrieval_queries: int = Field(default=6, ge=1, le=20)
    chunks_to_retrieve: int = Field(default=15, ge=1, le=200)
    top_k: int = Field(default=3, ge=1, le=100)
    chunk_expansion: int = Field(default=1, ge=0, le=10)
    concurrency: int = Field(default=8, ge=1, le=64)
    group_one_document_context: bool = True
    sort_ascending_one_document_context: bool = True
    canonicalize_vector_queries: bool = True


class SynthesisConfig(BaseModel):
    """Define answer-context and token limits."""

    model_config = _STRICT

    max_context_documents: int = Field(default=3, ge=1, le=50)
    max_context_chars: int = Field(default=8000, ge=500, le=50000)
    answer_max_tokens: int = Field(default=2048, ge=512, le=32768)
    citations: bool = False


class HistoryConfig(BaseModel):
    """Define bounds for the client-supplied conversation history."""

    model_config = _STRICT

    max_messages: int = Field(default=20, ge=0, le=200)
    max_message_chars: int = Field(default=4000, ge=50, le=100000)


class MemoryConfig(BaseModel):
    """Define cross-turn conversation memory limits."""

    model_config = _STRICT

    enabled: bool = Field(default=True)
    max_turns: int = Field(default=3, ge=1, le=20)
    answer_chars: int = Field(default=1800, ge=100, le=20000)
    question_chars: int = Field(default=500, ge=20, le=4000)


class ChunkingConfig(BaseModel):
    """Define document chunking settings."""

    model_config = _STRICT

    chunk_size: int = Field(default=384, ge=64, le=8192)
    chunk_overlap: int = Field(default=64, ge=0, le=4096)

    @model_validator(mode="after")
    def _overlap_below_size(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
        return self


class EmbeddingConfig(BaseModel):
    """Define embedding-side limits."""

    model_config = _STRICT

    max_input_tokens: int = Field(default=512, ge=64, le=8192)
    max_batch_size: int = 32


class IndexingConfig(BaseModel):
    """Define storage-write recovery limits for a corpus import."""

    model_config = _STRICT

    bulk_auth_retries: int = Field(default=3, ge=0, le=10)
    bulk_auth_backoff_s: float = Field(default=2.0, ge=0.0, le=60.0)
    visibility_timeout_s: float = Field(default=300.0, ge=1.0, le=1800.0)
    visibility_poll_interval_s: float = Field(default=2.0, ge=0.1, le=30.0)
    visibility_stall_s: float = Field(default=60.0, ge=1.0, le=1800.0)


class ModelProfile(BaseModel):
    """Define per-model-family request hints."""

    model_config = _STRICT

    chat_extra_body: dict[str, Any] = Field(default_factory=dict)
    reasoning_extra_body: dict[str, Any] = Field(default_factory=dict)


class LLMBackendsConfig(BaseModel):
    """Define backend defaults selected by chat URL."""

    model_config = _STRICT

    openrouter_default_extra_body: dict[str, Any] = Field(
        default_factory=lambda: {"reasoning": {"effort": "none"}}
    )
    vllm_default_extra_body: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Define the complete YAML-loaded operational configuration."""

    model_config = _STRICT

    history: HistoryConfig = Field(default_factory=HistoryConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    llm_backends: LLMBackendsConfig = Field(default_factory=LLMBackendsConfig)

    model_profiles: dict[str, ModelProfile] = Field(
        default_factory=lambda: {
            "qwen": ModelProfile(
                chat_extra_body={},
                reasoning_extra_body={
                    "chat_template_kwargs": {"enable_thinking": True}
                },
            ),
            "deepseek": ModelProfile(
                reasoning_extra_body={"reasoning": {"effort": "medium"}},
            ),
        }
    )

    _DEFAULT_PATH: ClassVar[str] = "params.agent.yaml"

    @classmethod
    def from_yaml(cls, path: str | None = None) -> AgentConfig:
        raw_path = (
            path or get_optional_env_var("AGENT_CONFIG_PATH") or cls._DEFAULT_PATH
        )
        config_path = _resolve(raw_path)
        try:
            with config_path.open(encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except FileNotFoundError:
            logger.info(f"No agent config at {config_path}; using defaults")
            data = {}
        else:
            logger.info(f"Loaded agent config from {config_path}")
        return cls.model_validate(data.get("agent", data))


_config: AgentConfig | None = None


def get_agent_config(path: str | None = None) -> AgentConfig:
    global _config
    if _config is None:
        _config = AgentConfig.from_yaml(path)
    return _config


def reload_agent_config(path: str | None = None) -> AgentConfig:
    global _config
    _config = None
    return get_agent_config(path)


__all__ = [
    "AgentConfig",
    "ModelProfile",
    "get_agent_config",
    "reload_agent_config",
]