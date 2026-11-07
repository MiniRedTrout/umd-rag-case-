from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.core import settings
from src.core.config.llm_config import llm_settings
from src.memory.engine import MemoryEngine, create_memory_engine
from src.memory.repository import PostgresSessionMemoryRepository
from src.rag.vectorestores.index_schema import (
    DOCUMENTS_INDEX_PROPERTIES,
    FAQ_INDEX_PROPERTIES,
    initialize_index,
)
from src.rag.vectorestores.index_schema import sanitize_embedding_field_name

from src.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_fields = {
        sanitize_embedding_field_name(settings.VLLM_E5_NAME): settings.EMB_E5_SIZE,
        sanitize_embedding_field_name(settings.VLLM_BGE_NAME): settings.EMB_BGE_SIZE,
    }
    await initialize_index(
        index_name=settings.OPENSEARCH_COLLECTION,
        embedding_fields=embedding_fields,
        client=llm_settings.client,
        is_delete_existing=False,
    )
    await initialize_index(
        index_name=settings.FAQ_OPENSEARCH_COLLECTION,
        embedding_fields=embedding_fields,
        client=llm_settings.client,
        is_delete_existing=False,
        additional_properties=FAQ_INDEX_PROPERTIES,
    )
    await initialize_index(
        index_name=settings.DOCS_OPENSEARCH_COLLECTION,
        embedding_fields=embedding_fields,
        client=llm_settings.client,
        is_delete_existing=False,
        additional_properties=DOCUMENTS_INDEX_PROPERTIES,
    )

    memory_engine = await _open_memory()
    app.state.memory_engine = memory_engine

    try:
        index_name = settings.OPENSEARCH_COLLECTION
        try:
            exists = await llm_settings.client.indices.exists(index=index_name)
        except Exception:
            logger.exception(f"Could not inspect index '{index_name}'")
            exists = False

        if exists:
            logger.info(f"Serving index '{index_name}'")
        else:
            logger.warning(
                f"Index '{index_name}' does not exist. Upload a knowledge archive "
                f"to /internal/knowledge/import before querying."
            )

        yield
    finally:
        app.state.memory_engine = None
        await _close_memory(memory_engine)
        logger.info("Stop the app")


async def _open_memory() -> MemoryEngine | None:
    try:
        engine = await create_memory_engine()
    except Exception:
        logger.exception("[memory] conversation memory is unavailable; "
                         "requests will run without cross-turn memory")
        return None
    from src.memory.history import set_engine as set_history_engine

    from src.graph.memory import set_repository
    set_repository(PostgresSessionMemoryRepository(engine))
    set_history_engine(engine)

    set_repository(PostgresSessionMemoryRepository(engine))
    return engine


async def _close_memory(engine: MemoryEngine | None) -> None:
    from src.graph.memory import reset_repository
    from src.memory.history import reset_engine as reset_history_engine


    reset_repository()
    reset_history_engine()

    if engine is None:
        return
    try:
        await engine.dispose()
        logger.info("[memory] PostgreSQL connections closed")
    except Exception:
        logger.exception("[memory] failed to close the PostgreSQL pool")