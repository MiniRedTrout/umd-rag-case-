
from __future__ import annotations

import hashlib
import io
import re
import time
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.core.config.agent_config import get_agent_config
from src.core.config.applications import SECTIONAL_PARSING_ENABLED, TEXT_SPLITTER
from src.rag.chunks import ChunkRecord, chunk_key_of
from src.rag.loaders.docx_reader import DocxBlock, TABLE_CLOSE, TABLE_OPEN, read_docx
from src.utils.logger import logger


TABLE_BLOCK = re.compile(
    re.escape(TABLE_OPEN) + r".*?" + re.escape(TABLE_CLOSE), re.DOTALL
)


@dataclass
class LoadedArchive:
    chunks: list[ChunkRecord] = field(default_factory=list)


def _recursive_text_splitter() -> RecursiveCharacterTextSplitter:
    chunking = get_agent_config().chunking
    try:
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunking.chunk_size,
            chunk_overlap=chunking.chunk_overlap,
        )
    except Exception as exc:
        # The parser must remain usable offline; indexing can still be inspected
        # when tiktoken cannot download its encoding table.
        logger.warning(f"Falling back to character splitter: {exc}")
        return RecursiveCharacterTextSplitter(
            chunk_size=chunking.chunk_size,
            chunk_overlap=chunking.chunk_overlap,
        )


def _text_splitter() -> RecursiveCharacterTextSplitter | MarkdownHeaderTextSplitter:
    """В RAG-системе вы обычно сначала разбиваете документ
      на чанки, а затем отправляете эти чанки в модель
      эмбеддингов для векторизации. Если ваша модель 
      эмбеддингов — это text-embedding-3-small (или другая 
      модель OpenAI), то использование encoding_name="cl100k_base" 
      в сплиттере гарантирует, что границы чанков 
      рассчитываются на основе того же токенизатора, который
      использует модель эмбеддингов. Это позволяет избежать
      ситуаций, когда чанк, «помещающийся» в лимит по 
      подсчётам сплиттера, на самом деле превышает реальное 
      ограничение модели по токенам, или наоборот — 
      слишком консервативно режется.
    """
    if TEXT_SPLITTER == "markdown":
        return MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
            ],
            strip_headers=False,
        )

    return _recursive_text_splitter()


def _filter_repeated(texts: Iterable[str], seen: set[bytes]) -> list[str]:
    unique: list[str] = []
    for text in texts:
        if not text:
            continue
        digest = hashlib.blake2b(text.encode(), digest_size=16).digest()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(text)
    return unique


def _split_text(
    text: str,
    splitter: RecursiveCharacterTextSplitter | MarkdownHeaderTextSplitter,
) -> list[str]:
    parts = splitter.split_text(text)
    if TEXT_SPLITTER == "markdown":
        return [part.page_content for part in parts if part.page_content.strip()]
    return parts


def _split_with_tables(
    content: str,
    splitter: RecursiveCharacterTextSplitter | MarkdownHeaderTextSplitter,
) -> list[str]:
    pieces: list[str] = []
    last = 0
    for match in TABLE_BLOCK.finditer(content):
        before = content[last : match.start()]
        pieces.extend(_split_text(before, splitter) if before.strip() else [])
        pieces.append(match.group(0))
        last = match.end()
    tail = content[last:]
    pieces.extend(_split_text(tail, splitter) if tail.strip() else [])
    return pieces


def _split_docx_blocks(
    blocks: list[DocxBlock],
    splitter: RecursiveCharacterTextSplitter | MarkdownHeaderTextSplitter,
) -> list[str]:
    """Split DOCX blocks with optional section-aware context."""
    if not SECTIONAL_PARSING_ENABLED:
        content = "\n\n".join(block.text for block in blocks if block.text.strip())
        return _split_with_tables(content, splitter)

    texts: list[str] = []
    for block in blocks:
        if block.kind in {"table", "heading"}:
            texts.append(block.text)
        else:
            prefix = f"[{block.section_title}]\n" if block.section_title else ""
            for piece in _split_text(block.text, splitter):
                texts.append(prefix + piece)
    return texts


def _chunk_records(
    texts: list[str], *, doc_id: str, path: str, title: str, source_format: str
) -> list[ChunkRecord]:
    total = len(texts)
    return [
        ChunkRecord(
            chunk_key=chunk_key_of(doc_id, index),
            doc_id=doc_id,
            chunk_index=index,
            chunk_total=total,
            path=path,
            title=title,
            source_format=source_format,
            text=text,
        )
        for index, text in enumerate(texts)
    ]


def _load_chunks(zip_file: zipfile.ZipFile, all_files: list[str]) -> list[ChunkRecord]:
    text_files = [
        f
        for f in all_files
        if f.lower().endswith((".txt", ".md", ".docx"))
        and not f.startswith("__")
        and not f.startswith(".")
        and not Path(f).name.startswith("~$")
        and not f.endswith("/")
    ]
    logger.info(
        f"Found {len(text_files)} text files in zip "
        f"(from {len(all_files)} total items)"
    )

    splitter = _text_splitter()
    # DOCX is parsed into structural blocks, not converted to Markdown.
    docx_splitter = _recursive_text_splitter()
    records: list[ChunkRecord] = []

    started = time.monotonic()
    docx_files = 0
    txt_files = 0
    docx_seconds = 0.0
    txt_seconds = 0.0
    docx_chunks = 0
    txt_chunks = 0
    failed = 0

    for file_path in text_files:
        file_started = time.monotonic()
        filename = Path(file_path).name
        title = Path(filename).stem

        try:
            with zip_file.open(file_path) as file:
                raw = file.read()

            if file_path.lower().endswith(".docx"):
                docx_files += 1
                blocks = read_docx(raw, name=filename)
                texts = _split_docx_blocks(blocks, docx_splitter)
                chunks_for_file = len(texts)
                docx_chunks += chunks_for_file
                docx_seconds += time.monotonic() - file_started
            else:
                txt_files += 1
                content = raw.decode("utf-8")
                texts = _split_with_tables(content, splitter)
                chunks_for_file = len(texts)
                txt_chunks += chunks_for_file
                txt_seconds += time.monotonic() - file_started

            # Deduplication is intentionally scoped to this document. Repeated
            # boilerplate in different documents (for example, standard
            # headers) remains available to retrieval.
            texts = _filter_repeated(texts, set())
            fmt = "docx" if file_path.lower().endswith(".docx") else Path(file_path).suffix.lower().lstrip(".")

            records.extend(
                _chunk_records(texts, doc_id=file_path, path=file_path, title=title, source_format=fmt)
            )
            logger.debug(
                f"[load] '{filename}': {chunks_for_file} chunks "
                f"in {(time.monotonic() - file_started) * 1000:.0f} ms"
            )

        except Exception as e:
            failed += 1
            logger.warning(f"Failed to load file {file_path}: {str(e)}")
            continue

    total_elapsed = time.monotonic() - started
    logger.info(
        f"[load] done in {total_elapsed:.1f}s: "
        f"{len(records)} chunks from {len(text_files)} files "
        f"({failed} failed). "
        f"docx: {docx_files} files, {docx_chunks} chunks, {docx_seconds:.1f}s | "
        f"txt: {txt_files} files, {txt_chunks} chunks, {txt_seconds:.1f}s"
    )
    return records


def load_data(zip_content: bytes) -> LoadedArchive:
    logger.info("Starting load")
    result = LoadedArchive()

    with zipfile.ZipFile(io.BytesIO(zip_content), metadata_encoding="cp866") as zip_f:
        files = zip_f.namelist()
        logger.info(f"Archive contains {len(files)} files")
        result.chunks = _load_chunks(zip_f, files)
        logger.info(f"Created {len(result.chunks)} chunks")

    return result
