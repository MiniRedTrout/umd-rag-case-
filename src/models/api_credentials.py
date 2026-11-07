from typing import Any

from src.core.config.agent_config import ModelProfile, get_agent_config
from src.models.llms.request_policy import LLMRequestPolicy
from src.utils.env_wrapper import get_env_var, get_optional_env_var


def is_openrouter(base_url: str | None) -> bool:
    """Identify whether an endpoint targets OpenRouter."""
    return "openrouter.ai" in (base_url or "").casefold()


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def resolve_model_profile(model_name: str) -> ModelProfile:
    name = (model_name or "").casefold()
    for key, profile in get_agent_config().model_profiles.items():
        if key.casefold() in name:
            return profile
    return ModelProfile()


def is_groq(base_url: str | None) -> bool:
    return "api.groq.com" in (base_url or "").casefold()


def uses_gateway(base_url: str | None) -> bool:
    """Determine whether an endpoint sits behind the GPN gateway.

    Only the real GPN gateway reads ``use_censor`` and ``X-Content-Origin``.
    Every public OpenAI-compatible endpoint (Groq, OpenRouter, Ollama,
    local vLLM) rejects them as unknown fields.
    """
    if is_openrouter(base_url) or is_groq(base_url):
        return False
    # Ollama / local vLLM / anything on localhost is also not the GPN gateway.
    lowered = (base_url or "").casefold()
    if "localhost" in lowered or "127.0.0.1" in lowered:
        return False
    return True


def build_chat_extra_body(
    base_url: str | None,
    model_name: str = "",
    *,
    enable_reasoning: bool = False,
    request_policy: LLMRequestPolicy | None = None,
) -> dict[str, Any]:
    config = get_agent_config()
    extra_body = dict(
        config.llm_backends.openrouter_default_extra_body
        if is_openrouter(base_url)
        else config.llm_backends.vllm_default_extra_body
    )

    profile = resolve_model_profile(model_name)
    extra_body = _deep_merge(extra_body, profile.chat_extra_body)
    if enable_reasoning and profile.reasoning_extra_body:
        extra_body = _deep_merge(extra_body, profile.reasoning_extra_body)

    if request_policy is not None and uses_gateway(base_url):
        extra_body = _deep_merge(extra_body, request_policy.extra_body)
    return extra_body


_NO_KEY = "not-needed"


def get_api_key(base_url: str | None = None) -> str:
    """Select the credential for an endpoint."""
    if base_url is None:
        base_url = get_optional_env_var("VLLM_LLM_URL")
    if is_openrouter(base_url):
        return get_env_var("OPENROUTER_API_KEY")
    if static_key := get_optional_env_var("VLLM_API_KEY"):
        return static_key
    return _NO_KEY


def get_default_headers(
    base_url: str | None = None,
    *,
    request_policy: LLMRequestPolicy | None = None,
) -> dict[str, str] | None:
    """Build backend-specific request headers for an endpoint."""
    if base_url is None:
        base_url = get_optional_env_var("VLLM_LLM_URL")

    if not is_openrouter(base_url):
        if request_policy is None:
            return None
        return dict(request_policy.headers)

    headers = {}

    if referer := get_optional_env_var("OPENROUTER_HTTP_REFERER"):
        headers["HTTP-Referer"] = referer

    if title := get_optional_env_var("OPENROUTER_TITLE"):
        headers["X-OpenRouter-Title"] = title

    return headers or None