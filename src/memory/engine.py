from __future__ import annotations

from loguru import logger
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.memory.config import SCHEMA, MemoryDatabaseConfig, load_memory_database_config
from src.memory.schema import TABLE


class MemoryDatabaseUnavailable(RuntimeError):
    """Raised when the memory database cannot be reached."""


class MemorySchemaMissing(MemoryDatabaseUnavailable):
    """Raised when the database answers but has never been migrated."""


class MemoryEngine:
    """Owns the engine and the sessionmaker for one process."""

    def __init__(self, config: MemoryDatabaseConfig) -> None:
        self._config = config
        self._engine = create_async_engine(
            config.dsn,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_timeout=config.pool_timeout,
            pool_recycle=config.pool_recycle,
            pool_pre_ping=True,
            echo=False,
        )
        self._sessionmaker = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def connect(self) -> None:
        try:
            async with self._engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception as error:
            raise MemoryDatabaseUnavailable(
                f"The PostgreSQL database at {self._config.safe_dsn} did not "
                f"answer: {type(error).__name__}"
            ) from None

    @property
    def sessionmaker(self) -> async_sessionmaker[AsyncSession]:
        return self._sessionmaker

    @property
    def safe_dsn(self) -> str:
        return self._config.safe_dsn

    async def verify(self) -> None:
        try:
            async with self._engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
                migrated = await connection.scalar(
                    text("SELECT to_regclass(:qualified_name)"),
                    {"qualified_name": f"{SCHEMA}.{TABLE}"},
                )
        except SQLAlchemyError as error:
            raise MemoryDatabaseUnavailable(
                f"The PostgreSQL database at {self._config.safe_dsn} did not "
                f"answer: {type(error).__name__}"
            ) from error
        if migrated is None:
            raise MemorySchemaMissing(
                f"The PostgreSQL database at {self._config.safe_dsn} is "
                f"reachable but has no {SCHEMA} schema. Run the migrations: "
                f"python -m src.memory.migrations.run_migrations"
            )

    async def dispose(self) -> None:
        await self._engine.dispose()


async def create_memory_engine(
    config: MemoryDatabaseConfig | None = None,
) -> MemoryEngine:
    engine = MemoryEngine(config or load_memory_database_config())
    try:
        await engine.connect()
        await engine.verify()
    except Exception:
        await engine.dispose()
        raise
    logger.info(f"[memory] connected to PostgreSQL at {engine.safe_dsn}")
    return engine