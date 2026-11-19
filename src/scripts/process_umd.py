"""Build a serialised intermediate representation from the UMD DOCX corpus.

The script intentionally stops after loading, parsing, splitting and
document-local deduplication. It does not call embedding services and does
not modify OpenSearch.

Example:
    python -m src.scripts.process_umd --input test_docx --output artifacts/umd_chunks.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SUPPORTED_SUFFIXES = {".docx", ".txt", ".md"}


def _build_archive(root: Path, max_files: int | None) -> tuple[bytes, list[Path]]:
    files = sorted(
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_SUFFIXES
            and not path.name.startswith("~$")
        )
    )
    if max_files is not None:
        files = files[:max_files]

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(root).as_posix())
    return buffer.getvalue(), files


def _make_result(
    *, root: Path, files: list[Path], loaded: Any, splitter: str, sectional: str
) -> dict[str, Any]:
    chunks = [chunk.model_dump() for chunk in loaded.chunks]
    by_document: dict[str, int] = {}
    for chunk in loaded.chunks:
        by_document[chunk.path] = by_document.get(chunk.path, 0) + 1

    return {
        "schema_version": "umd-intermediate-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "input_root": str(root),
        "source_files": len(files),
        "parsed_documents": len(by_document),
        "chunks": len(chunks),
        "configuration": {
            "text_splitter": splitter,
            "sectional_parsing_enabled": sectional,
            "deduplication_scope": "document",
        },
        "documents": [
            {"path": path, "chunks": count}
            for path, count in sorted(by_document.items())
        ],
        "chunks_data": chunks,
    }


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    chunks_by_path: dict[str, list[dict[str, Any]]] = {}
    for chunk in result["chunks_data"]:
        chunks_by_path.setdefault(chunk["path"], []).append(chunk)

    lines = [
        "# UMD: parsed document preview",
        "",
        f"- Source files: {result['source_files']}",
        f"- Parsed documents: {result['parsed_documents']}",
        f"- Chunks: {result['chunks']}",
        f"- Text splitter: `{result['configuration']['text_splitter']}`",
        f"- Sectional parsing: `{result['configuration']['sectional_parsing_enabled']}`",
        f"- Deduplication: `{result['configuration']['deduplication_scope']}`",
        "",
        "This file is a visual inspection artifact. Chunk headings are added by "
        "the exporter; the text below is produced by the parser.",
        "",
    ]
    for document in result["documents"]:
        document_path = document["path"]
        lines.extend(
            [
                f"## {Path(document_path).stem}",
                "",
                f"**Source:** `{document_path}`  ",
                f"**Chunks:** {document['chunks']}",
                "",
            ]
        )
        for chunk in chunks_by_path.get(document_path, []):
            lines.extend(
                [
                    f"### Chunk {chunk['chunk_index'] + 1}/{chunk['chunk_total']}",
                    "",
                    chunk["text"].rstrip(),
                    "",
                ]
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_document_markdown(output_root: Path, result: dict[str, Any]) -> None:
    chunks_by_path: dict[str, list[dict[str, Any]]] = {}
    for chunk in result["chunks_data"]:
        chunks_by_path.setdefault(chunk["path"], []).append(chunk)

    for document in result["documents"]:
        source_path = Path(document["path"])
        output_path = output_root / source_path.with_suffix(".md")
        chunks = chunks_by_path.get(document["path"], [])
        lines = [
            f"# {source_path.stem}",
            "",
            f"> Parsed from `{document['path']}`.",
            "> Chunk boundaries below are added by the exporter.",
            "",
        ]
        for chunk in chunks:
            lines.extend(
                [
                    f"<!-- chunk {chunk['chunk_index'] + 1}/{chunk['chunk_total']} -->",
                    chunk["text"].rstrip(),
                    "",
                ]
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("test_docx"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/umd_chunks.json")
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("artifacts/umd_parsed.md"),
        help="Write a human-readable Markdown preview of parsed chunks.",
    )
    parser.add_argument(
        "--documents-output",
        type=Path,
        default=Path("test_md"),
        help="Write one parsed Markdown file per source document.",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--splitter",
        choices=("recursive", "markdown"),
        default=None,
        help="Override TEXT_SPLITTER for this run.",
    )
    parser.add_argument(
        "--sectional-parsing",
        choices=("true", "false"),
        default=None,
        help="Override SECTIONAL_PARSING_ENABLED for this run.",
    )
    args = parser.parse_args()

    root = args.input.resolve()
    if not root.is_dir():
        parser.error(f"input directory does not exist: {root}")
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be positive")

    if args.splitter is not None:
        os.environ["TEXT_SPLITTER"] = args.splitter
    if args.sectional_parsing is not None:
        os.environ["SECTIONAL_PARSING_ENABLED"] = args.sectional_parsing

    # Import after CLI overrides so the same loader can be used for experiments.
    from src.rag.loaders import load_data
    from src.core.config.applications import SECTIONAL_PARSING_ENABLED, TEXT_SPLITTER

    archive, files = _build_archive(root, args.max_files)
    loaded = load_data(archive)
    result = _make_result(
        root=root,
        files=files,
        loaded=loaded,
        splitter=TEXT_SPLITTER,
        sectional=str(SECTIONAL_PARSING_ENABLED).lower(),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    _write_markdown(args.markdown_output, result)
    _write_document_markdown(args.documents_output, result)
    print(
        f"Processed {result['source_files']} files -> "
        f"{result['parsed_documents']} documents, {result['chunks']} chunks"
    )
    print(f"Intermediate result: {args.output}")
    print(f"Markdown preview: {args.markdown_output}")
    print(f"Per-document Markdown files: {args.documents_output}")


if __name__ == "__main__":
    main()

