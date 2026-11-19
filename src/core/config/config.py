from pathlib import Path

from dotenv import load_dotenv

from src.utils.env_wrapper import get_bool_env_var, get_env_var, get_optional_env_var

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Keep experiment switches in a separate, non-secret configuration file.
load_dotenv(_PROJECT_ROOT / "applications.conf")
load_dotenv(_PROJECT_ROOT / ".env")


class Settings:
    """Application configuration."""

    SHOW_LINKS: bool = get_env_var("SHOW_LINKS", True)
    VLLM_E5_NAME = get_env_var("VLLM_E5_NAME")
    VLLM_BGE_NAME = get_env_var("VLLM_BGE_NAME")

    EMB_E5_MODEL = get_env_var("EMB_E5_MODEL", VLLM_E5_NAME)
    EMB_BGE_MODEL = get_env_var("EMB_BGE_MODEL", VLLM_BGE_NAME)
    CHUNK_WINDOW_SIZE = 1
    SORT_ASCENDING_ONE_DOCUMENT_CONTEXT = True
    GROUP_ONE_DOCUMENT_CONTEXT = True

    SECTIONAL_PARSING_ENABLED = get_bool_env_var("SECTIONAL_PARSING_ENABLED", True)
    TEXT_SPLITTER = get_env_var("TEXT_SPLITTER", "recursive").lower()
    if TEXT_SPLITTER not in {"recursive", "markdown"}:
        raise ValueError("TEXT_SPLITTER must be either 'recursive' or 'markdown'")

    OPENSEARCH_USE_SSL = get_env_var("OPENSEARCH_USE_SSL", "false").lower() == "true"
    OPENSEARCH_VERIFY_CERTS = get_env_var("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true"
    OPENSEARCH_SSL_ASSERT_HOSTNAME = get_env_var("OPENSEARCH_SSL_ASSERT_HOSTNAME", "false").lower() == "true"
    OPENSEARCH_SSL_SHOW_WARN = get_env_var("OPENSEARCH_SSL_SHOW_WARN", "false").lower() == "true"
    OPENSEARCH_HOSTS = (get_env_var("OPENSEARCH_HOSTS"),)
    OPENSEARCH_USER = get_optional_env_var("OPENSEARCH_USER")
    OPENSEARCH_PASSWORD = get_optional_env_var("OPENSEARCH_PASSWORD")
    OPENSEARCH_USE_AUTH = (get_optional_env_var("OPENSEARCH_USE_AUTH") or "").lower() == "true"
    OPENSEARCH_COLLECTION = get_env_var("OPENSEARCH_COLLECTION")
    FAQ_OPENSEARCH_COLLECTION = get_env_var("FAQ_OPENSEARCH_COLLECTION", f"{OPENSEARCH_COLLECTION}_faq")
    DOCS_OPENSEARCH_COLLECTION = get_env_var("DOCS_OPENSEARCH_COLLECTION", f"{OPENSEARCH_COLLECTION}_documents")

    VLLM_EMB_URL = get_env_var("VLLM_EMB_URL")
    VLLM_EMB_URL_E5 = get_env_var("VLLM_EMB_URL_E5", VLLM_EMB_URL)
    VLLM_EMB_URL_BGE = get_env_var("VLLM_EMB_URL_BGE", VLLM_EMB_URL)

    VLLM_E5_NAME = get_env_var("VLLM_E5_NAME")
    VLLM_BGE_NAME = get_env_var("VLLM_BGE_NAME")

    EMB_E5_SIZE = int(get_env_var("EMB_E5_SIZE"))
    EMB_BGE_SIZE = int(get_env_var("EMB_BGE_SIZE"))
    MAX_BATCH_SIZE = int(get_env_var("MAX_BATCH_SIZE", "32"))

    VLLM_LLM_URL = get_env_var("VLLM_LLM_URL")
    LLM_ENABLE_THINKING = get_env_var("LLM_ENABLE_THINKING", "false").lower() == "true"

    PYPROJECT_PATH = get_env_var("PYPROJECT_PATH", default="pyproject.toml")


settings = Settings()
