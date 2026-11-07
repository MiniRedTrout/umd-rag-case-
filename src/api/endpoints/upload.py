from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    UploadFile,
    status,
)
import asyncio
import time

from src.core import settings
from src.core.config.agent_config import get_agent_config
from src.core.config.llm_config import llm_settings
from src.rag.loaders import load_data
from src.rag.vectorestores.index_manager import validate_index
from src.rag.vectorestores.index_schema import initialize_index
from src.schemas.upload import ImportResult
from src.utils.logger import logger

from src.utils.metrics import record_indexing, snapshot as metrics_snapshot


upload_router = APIRouter()
is_processing = False
processing_status: dict | None = None


def _embedding_fields() -> dict[str, int]:
    return {
        settings.VLLM_E5_NAME: settings.EMB_E5_SIZE,
        settings.VLLM_BGE_NAME: settings.EMB_BGE_SIZE,
    }


async def _validate(index_name: str, expected: int) -> None:
    indexing = get_agent_config().indexing
    await validate_index(
        llm_settings.client,
        index_name,
        expected_documents=expected,
        embedding_fields=list(_embedding_fields()),
        visibility_timeout_s=indexing.visibility_timeout_s,
        visibility_poll_interval_s=indexing.visibility_poll_interval_s,
        visibility_stall_s=indexing.visibility_stall_s,
    )


async def process_zip_background(zip_content: bytes) -> None:
    global is_processing, processing_status

    started = time.monotonic()
    failed = False
    chunks_count = 0
    documents_count = 0

    try:
        processing_status = {"status": "processing", "message": "Processing started..."}

        loaded = await asyncio.to_thread(load_data, zip_content)
        await initialize_index(
            index_name=settings.OPENSEARCH_COLLECTION,
            embedding_fields=_embedding_fields(),
            client=llm_settings.client,
            is_delete_existing=True,
        )

        message = "Import completed"
        if loaded.chunks:
            logger.info("Adding chunks to vector store...")
            await llm_settings.opensearch_wrapper.add_chunks(loaded.chunks)
            await _validate(settings.OPENSEARCH_COLLECTION, expected=len(loaded.chunks))
            chunks_count = len(loaded.chunks)
            documents_count = len({c.doc_id for c in loaded.chunks})
            message = f"Successfully loaded {chunks_count} chunks"

        processing_status = {"status": "completed", "message": message}

    except Exception as e:
        failed = True
        error_msg = f"Failed to process zip file: {e}"
        logger.exception(f"Background processing failed: {error_msg}")
        processing_status = {"status": "failed", "message": error_msg}

    finally:
        is_processing = False
        elapsed = time.monotonic() - started
        record_indexing(
            seconds=elapsed,
            chunks=chunks_count,
            documents=documents_count,
            failed=failed,
        )
        logger.info(
            f"[import] {'FAILED' if failed else 'done'} in {elapsed:.1f}s; "
            f"metrics={metrics_snapshot()}"
        )


@upload_router.post(
    "/import",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ImportResult,
)
async def upload_courses_zip(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> ImportResult:
    global is_processing, processing_status

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only zip files are allowed",
        )

    if is_processing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Another file is currently being processed. "
                "Please wait for it to complete."
            ),
        )
    is_processing = True

    try:
        zip_content = await file.read()
    except Exception as e:
        is_processing = False
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file: {e}",
        ) from e

    processing_status = {
        "status": "processing",
        "message": (
            f"File '{file.filename}' has been accepted and queued for processing"
        ),
    }

    background_tasks.add_task(process_zip_background, zip_content)
    logger.info(f"Started background processing for file: {file.filename}")

    return ImportResult(**processing_status)


@upload_router.get("/import-status")
async def import_status():
    return processing_status or {"status": "idle", "message": "Ничего не запущено"}
