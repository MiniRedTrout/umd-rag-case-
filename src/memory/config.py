from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.utils.env_wrapper import get_optional_env_var

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / "applications.conf")
load_dotenv(_PROJECT_ROOT / ".env")

__all__ = [
    "MemoryConfigError",
    "MemoryDatabaseConfig",
    "SCHEMA",
    "load_memory_database_config",
    "redact_dsn",
]

SCHEMA = "memory_shared"


class MemoryConfigError(RuntimeError):
    """Raised when the PostgreSQL database is not configured."""


@dataclass(frozen=True)
class MemoryDatabaseConfig:
    dsn: str
    pool_size: int = 5
    max_overflow: int = 5
    pool_timeout: float = 10.0
    pool_recycle: int = 600

    @property
    def safe_dsn(self) -> str:
        return redact_dsn(self.dsn)


def redact_dsn(dsn: str) -> str:
    """Remove the password from a DSN for logging."""
    if "@" not in dsn:
        return dsn
    scheme, _, rest = dsn.partition("://")
    userinfo, _, host = rest.partition("@")
    if ":" in userinfo:
        user, _, _ = userinfo.partition(":")
        return f"{scheme}://{user}:***@{host}"
    return dsn


def load_memory_database_config() -> MemoryDatabaseConfig:
    """Build memory's pool configuration from POSTGRES_* environment variables."""
    user = get_optional_env_var("POSTGRES_USER")
    password = get_optional_env_var("POSTGRES_PASSWORD")
    host = get_optional_env_var("POSTGRES_HOST")
    port = get_optional_env_var("POSTGRES_PORT") or "5432"
    db = get_optional_env_var("POSTGRES_DB")
    if not (user and host and db):
        raise MemoryConfigError(
            "POSTGRES_USER, POSTGRES_HOST and POSTGRES_DB are required for "
            "conversation memory."
        )
    auth = f"{user}:{password}" if password else user
    dsn = f"postgresql+asyncpg://{auth}@{host}:{port}/{db}"
    return MemoryDatabaseConfig(dsn=dsn, **_pool_settings())


def _pool_settings() -> dict[str, float | int]:
    settings: dict[str, float | int] = {}
    value = _pool_value("MEMORY_POSTGRES_POOL_SIZE")
    if value is not None:
        settings["pool_size"] = value
    overflow = _pool_value("MEMORY_POSTGRES_MAX_OVERFLOW")
    if overflow is not None:
        settings["max_overflow"] = overflow
    timeout = _pool_value("MEMORY_POSTGRES_POOL_TIMEOUT", floating=True)
    if timeout is not None:
        settings["pool_timeout"] = timeout
    recycle = _pool_value("MEMORY_POSTGRES_POOL_RECYCLE")
    if recycle is not None:
        settings["pool_recycle"] = recycle
    return settings


def _pool_value(name: str, *, floating: bool = False) -> int | float | None:
    raw = get_optional_env_var(name)
    if raw is None:
        return None
    try:
        value = float(raw) if floating else int(raw)
    except ValueError as error:
        kind = "a finite number" if floating else "an integer"
        raise MemoryConfigError(f"{name} must be {kind}") from error
    if floating and not math.isfinite(value):
        raise MemoryConfigError(f"{name} must be a finite number")
    return value
