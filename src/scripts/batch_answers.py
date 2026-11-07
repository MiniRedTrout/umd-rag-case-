"""Run workbook questions through the chat API and save every answer."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections.abc import AsyncIterator, Iterable
from copy import copy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from openpyxl import load_workbook

load_dotenv()

DEFAULT_CHAT_URL = os.getenv("BATCH_CHAT_URL")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")
HEADER_SCAN_ROWS = 15
QUESTION_ALIASES = ("Вопрос", "question", "query")
RETRYABLE_HTTP_STATUSES = frozenset({429, 502, 503, 504})
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY = 2.0
DEFAULT_RETRY_MAX_DELAY = 30.0
MAX_ERROR_BODY_CHARS = 500

RESULT_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "answer": ("Ответ", "Ответ ассистента", "answer"),
    "sources": ("Источники", "sources"),
    "finish_reason": ("Причина завершения", "finish reason"),
    "usage": ("Использование токенов", "token usage"),
    "http_attempts": ("HTTP попытки", "http attempts"),
    "http_history": ("HTTP история", "http history"),
    "error": ("Ошибка", "error"),
}


@dataclass(slots=True)
class ChatResult:
    answer: str
    references: list[dict[str, Any]]
    finish_reason: str
    usage: dict[str, Any]
    errors: list[str] = field(default_factory=list)
    http_history: list[dict[str, Any]] = field(default_factory=list)


class ChatRequestError(RuntimeError):
    def __init__(self, message: str, history: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.history = history


class SSEDecoder:
    def __init__(self) -> None:
        self._data_lines: list[str] = []
        self.events: list[dict[str, Any]] = []

    def feed_line(self, line: str) -> None:
        if line == "":
            self._flush()
            return
        if line.startswith(":"):
            return
        field_name, separator, value = line.partition(":")
        if not separator:
            value = ""
        elif value.startswith(" "):
            value = value[1:]
        if field_name == "data":
            self._data_lines.append(value)

    def finish(self) -> list[dict[str, Any]]:
        self._flush()
        return self.events

    def _flush(self) -> None:
        if not self._data_lines:
            return
        raw = "\n".join(self._data_lines)
        self._data_lines.clear()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in SSE data: {raw[:200]}") from exc
        if not isinstance(payload, dict):
            raise TypeError("SSE data must contain a JSON object")
        self.events.append(payload)


def _parse_events(events: Iterable[dict[str, Any]]) -> ChatResult:
    terminal: dict[str, Any] | None = None
    errors: list[str] = []
    saw_done = False
    for event in events:
        event_type = event.get("type")
        if event_type == "message_end" and event.get("finish_reason") != "tool_calls":
            terminal = event
        elif event_type == "error":
            details = event.get("error")
            if isinstance(details, dict):
                errors.append(str(details.get("message") or details.get("type") or "error"))
            else:
                errors.append("api_error")
        elif event_type == "done":
            saw_done = True
    if terminal is None:
        raise RuntimeError(f"SSE stream ended{' before done' if saw_done else ''} without a terminal message_end")
    return ChatResult(
        answer=str(terminal.get("text") or "").strip(),
        references=[item for item in terminal.get("references") or [] if isinstance(item, dict)],
        finish_reason=str(terminal.get("finish_reason") or ""),
        usage=terminal.get("usage") or {},
        errors=errors,
    )


def _find_header_row(worksheet) -> tuple[int, dict[str, int], str]:
    aliases = {alias.casefold(): alias for alias in QUESTION_ALIASES}
    for row_idx in range(1, min(HEADER_SCAN_ROWS, worksheet.max_row) + 1):
        headers: dict[str, int] = {}
        question_header = ""
        for col_idx, cell in enumerate(worksheet[row_idx], start=1):
            header = str(cell.value or "").strip()
            if not header:
                continue
            headers[header] = col_idx
            if header.casefold() in aliases:
                question_header = header
        if question_header:
            return row_idx, headers, question_header
    raise ValueError(f"Question column not found in first {HEADER_SCAN_ROWS} rows")


def _resolve_column(headers: dict[str, int], aliases: tuple[str, ...]) -> int | None:
    folded = {name.casefold(): col_idx for name, col_idx in headers.items()}
    for alias in aliases:
        if alias.casefold() in folded:
            return folded[alias.casefold()]
    return None


def _ensure_result_columns(worksheet, header_row: int, headers: dict[str, int]) -> dict[str, int]:
    columns: dict[str, int] = {}
    occupied = set(headers.values())
    next_candidate = 1
    for field_name, aliases in RESULT_FIELD_ALIASES.items():
        existing = _resolve_column(headers, aliases)
        if existing is not None:
            columns[field_name] = existing
            continue
        while next_candidate in occupied:
            next_candidate += 1
        columns[field_name] = next_candidate
        worksheet.cell(row=header_row, column=next_candidate, value=aliases[0])
        occupied.add(next_candidate)
    return columns


def _cell_text(worksheet, row_idx: int, col_idx: int) -> str:
    value = worksheet.cell(row=row_idx, column=col_idx).value
    return "" if value is None else str(value).strip()


def _format_references(references: list[dict[str, Any]]) -> str:
    rendered = []
    for ref in references:
        title = str(ref.get("title") or "").strip()
        url = str(ref.get("url") or "").strip()
        label = f"{title} — {url}" if url else title
        if label and label not in rendered:
            rendered.append(label)
    return "\n".join(rendered)


def _format_usage(usage: dict[str, Any]) -> str:
    preferred = ("input_tokens", "output_tokens", "total_tokens")
    return ", ".join(f"{name}={usage[name]}" for name in preferred if name in usage)


def _format_http_history(history: list[dict[str, Any]]) -> str:
    return " -> ".join(
        f"{item.get('attempt', '?')}: {item.get('error') or item.get('status') or 'unknown'}"
        for item in history
    )


async def _request_answer(
    client: httpx.AsyncClient,
    *,
    url: str,
    model_name: str,
    question: str,
    bearer_token: str | None,
    max_attempts: int,
    retry_base_delay: float,
    retry_max_delay: float,
) -> ChatResult:
    payload = {
        "model_name": model_name,
        "messages": [{"role": "user", "content": question}],
    }
    history: list[dict[str, Any]] = []
    attempt_limit = max_attempts
    attempt = 0
    headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token else None

    while attempt < attempt_limit:
        attempt += 1
        started = time.monotonic()
        history_item: dict[str, Any] | None = None
        try:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                history_item = {
                    "attempt": attempt,
                    "status": response.status_code,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "request_id": response.headers.get("x-request-id", ""),
                }
                history.append(history_item)
                if response.status_code == 200:
                    decoder = SSEDecoder()
                    async for line in response.aiter_lines():
                        decoder.feed_line(line)
                    result = _parse_events(decoder.finish())
                    result.http_history = history
                    return result
                body = (await response.aread()).decode(errors="replace")
                if response.status_code not in RETRYABLE_HTTP_STATUSES or attempt >= attempt_limit:
                    raise ChatRequestError(
                        f"HTTP {response.status_code} after {attempt} attempts: "
                        f"{body[:MAX_ERROR_BODY_CHARS]}",
                        history,
                    )
                retry_after = response.headers.get("retry-after")
        except (httpx.TransportError, httpx.StreamError) as exc:
            if history_item is None:
                history_item = {"attempt": attempt, "status": None}
                history.append(history_item)
            history_item.update(error=type(exc).__name__)
            if attempt >= attempt_limit:
                raise ChatRequestError(f"{type(exc).__name__}: {exc}", history) from exc
            retry_after = None
        await asyncio.sleep(retry_base_delay * (2 ** (attempt - 1)))

    raise AssertionError("unreachable")


async def _iter_results(
    client: httpx.AsyncClient,
    *,
    row_questions: list[tuple[int, str]],
    parallel_requests: int,
    request_options: dict[str, Any],
) -> AsyncIterator[tuple[int, ChatResult | None, Exception | None]]:
    semaphore = asyncio.Semaphore(parallel_requests)

    async def request_row(row_idx: int, question: str) -> tuple[int, ChatResult | None, Exception | None]:
        async with semaphore:
            try:
                return row_idx, await _request_answer(client, question=question, **request_options), None
            except Exception as exc:  # noqa: BLE001
                return row_idx, None, exc

    tasks = [asyncio.create_task(request_row(r, q)) for r, q in row_questions]
    try:
        for task in asyncio.as_completed(tasks):
            yield await task
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _run_batch(args: argparse.Namespace) -> None:
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_answers{input_path.suffix}")
    )
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")
    if output_path == input_path:
        raise SystemExit("Refusing to overwrite the input workbook")

    workbook = load_workbook(input_path)
    worksheet = workbook[workbook.sheetnames[0]]
    header_row, headers, question_header = _find_header_row(worksheet)
    columns = _ensure_result_columns(worksheet, header_row, headers)
    question_col = headers[question_header]

    rows = [
        r for r in range(header_row + 1, worksheet.max_row + 1)
        if _cell_text(worksheet, r, question_col)
    ]
    if not rows:
        raise SystemExit("No rows to process")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)
    row_questions = [(r, _cell_text(worksheet, r, question_col)) for r in rows]
    questions_by_row = dict(row_questions)

    timeout = httpx.Timeout(args.timeout, connect=min(args.timeout, 30.0))
    async with httpx.AsyncClient(
        timeout=timeout, verify=not args.insecure, follow_redirects=True
    ) as client:
        async for row_idx, result, request_error in _iter_results(
            client,
            row_questions=row_questions,
            parallel_requests=args.parallel_requests,
            request_options={
                "url": args.url,
                "model_name": args.model_name,
                "bearer_token": args.bearer_token,
                "max_attempts": args.max_attempts,
                "retry_base_delay": args.retry_base_delay,
                "retry_max_delay": args.retry_max_delay,
            },
        ):
            try:
                if request_error is not None:
                    raise request_error
                assert result is not None
                worksheet.cell(row_idx, columns["answer"], result.answer)
                worksheet.cell(row_idx, columns["sources"], _format_references(result.references))
                worksheet.cell(row_idx, columns["finish_reason"], result.finish_reason)
                worksheet.cell(row_idx, columns["usage"], _format_usage(result.usage))
                worksheet.cell(row_idx, columns["http_attempts"], len(result.http_history))
                worksheet.cell(row_idx, columns["http_history"], _format_http_history(result.http_history))
                worksheet.cell(row_idx, columns["error"], "\n".join(result.errors) if result.errors else None)
                print(f"row {row_idx}: OK — {questions_by_row[row_idx][:60]}")
            except Exception as exc:  # noqa: BLE001
                worksheet.cell(row_idx, columns["error"], f"{type(exc).__name__}: {exc}")
                print(f"row {row_idx}: FAILED — {exc}", file=sys.stderr)
            if row_idx % args.save_every == 0:
                workbook.save(output_path)

    workbook.save(output_path)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Excel questions through the SSE chat API.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--url", default=os.getenv("BATCH_CHAT_URL") or DEFAULT_CHAT_URL)
    parser.add_argument("--model-name", default=os.getenv("BATCH_MODEL_NAME") or DEFAULT_MODEL)
    parser.add_argument("--bearer-token", default=os.getenv("BATCH_AUTH_TOKEN") or None)
    parser.add_argument("--timeout", type=_positive_float, default=600.0)
    parser.add_argument("--parallel-requests", dest="parallel_requests", type=_positive_int, default=1)
    parser.add_argument("--max-attempts", type=_positive_int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--retry-base-delay", type=_positive_float, default=DEFAULT_RETRY_BASE_DELAY)
    parser.add_argument("--retry-max-delay", type=_positive_float, default=DEFAULT_RETRY_MAX_DELAY)
    parser.add_argument("--save-every", type=_positive_int, default=1)
    parser.add_argument("--insecure", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(_run_batch(args))


if __name__ == "__main__":
    main()