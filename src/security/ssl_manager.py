from functools import cached_property
from ssl import SSLContext
from typing import Dict, Optional

from src.security.ssl_protection import SSLProtection
from src.utils.env_wrapper import get_optional_env_var
from src.utils.logger import logger


class SSLManager:
    """Manages SSL contexts for different services with lazy loading and error handling."""

    _SERVICE_CA_MAP = {
        "opensearch": "OPENSEARCH_SSL_CA_FILE",
        "embedding": "EMBEDDING_SSL_CA_FILE",
        "llm": "LLM_SSL_CA_FILE",
        "postgres": "POSTGRES_SSL_CA_FILE",
    }

    def __init__(self):
        self._ssl_contexts: Dict[str, Optional[SSLContext]] = {}
        self._errors: Dict[str, Exception] = {}

    def get_ssl_context(self, service: str) -> Optional[SSLContext]:
        if service not in self._SERVICE_CA_MAP:
            raise ValueError(f"Unknown service: {service}. Available: {list(self._SERVICE_CA_MAP.keys())}")
        if service in self._errors:
            return None
        if service not in self._ssl_contexts:
            try:
                ca_file = get_optional_env_var(self._SERVICE_CA_MAP[service])
                if ca_file is not None:
                    self._ssl_contexts[service] = SSLProtection(ca_file=ca_file).ssl_context
                else:
                    self._ssl_contexts[service] = None
            except Exception as e:
                self._errors[service] = e
                self._ssl_contexts[service] = None
                logger.error(f"Error occurred while getting an ssl context for service: {service}")
        return self._ssl_contexts[service]

    def has_ssl(self, service: str) -> bool:
        return self.get_ssl_context(service) is not None

    def get_error(self, service: str) -> Optional[Exception]:
        return self._errors.get(service)

    @cached_property
    def postgres(self) -> Optional[SSLContext]:
        return self.get_ssl_context("postgres")

    @cached_property
    def opensearch(self) -> Optional[SSLContext]:
        return self.get_ssl_context("opensearch")

    @cached_property
    def embedding(self) -> Optional[SSLContext]:
        return self.get_ssl_context("embedding")

    @cached_property
    def llm(self) -> Optional[SSLContext]:
        return self.get_ssl_context("llm")


ssl_manager = SSLManager()