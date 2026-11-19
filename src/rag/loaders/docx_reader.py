from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass, field
from typing import Iterator

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from src.utils.logger import logger


TABLE_OPEN = "<<<TABLE>>>"
TABLE_CLOSE = "<<<END_TABLE>>>"

_MAX_STYLE_DEPTH = 8

_HEADING_MAX_CHARS = 200

_HEADING_STYLE_NAMES = re.compile(
    r"^(?:heading|заголовок)\s*(\d+)$", re.IGNORECASE
)

_HEADING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\d+(?:\.\d+)*\.?\s+\S"),                    # 1., 1.1., 1.1.1.
    re.compile(r"^Раздел\s+\d+", re.IGNORECASE),              # Раздел 3
    re.compile(r"^Глава\s+\d+", re.IGNORECASE),               # Глава 2
    re.compile(r"^Статья\s+\d+", re.IGNORECASE),              # Статья 5
    re.compile(r"^Приложение\s+[А-ЯA-Z0-9]", re.IGNORECASE),  # Приложение А
    re.compile(r"^ПРИЛОЖЕНИЕ\s+[А-ЯA-Z0-9]"),                # ПРИЛОЖЕНИЕ А
)


@dataclass
class DocxBlock:

    kind: str
    text: str
    level: int = 0
    section_title: str = ""


def iter_blocks(doc: Document) -> Iterator[Paragraph | Table]:

    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)
        elif child.tag == qn("w:sdt"):
            content = child.find(qn("w:sdtContent"))
            if content is None:
                continue
            for inner in content.iterchildren():
                if inner.tag == qn("w:p"):
                    yield Paragraph(inner, doc)
                elif inner.tag == qn("w:tbl"):
                    yield Table(inner, doc)


def _style_chain(style) -> Iterator[object]:
    seen = 0
    while style is not None and seen < _MAX_STYLE_DEPTH:
        yield style
        style = getattr(style, "base_style", None)
        seen += 1


def _outline_level_from_xml(paragraph: Paragraph) -> int | None:

    pPr = paragraph._element.find(qn("w:pPr"))
    if pPr is None:
        return None
    outline = pPr.find(qn("w:outlineLvl"))
    if outline is None:
        return None
    value = outline.get(qn("w:val"))
    try:
        level = int(value) + 1
    except (TypeError, ValueError):
        return None
    return level if 1 <= level <= 9 else None


def _outline_level_from_styles(paragraph: Paragraph) -> int | None:
    for style in _style_chain(paragraph.style):
        name = (style.name or "").strip()
        match = _HEADING_STYLE_NAMES.match(name)
        if match:
            return int(match.group(1))
        element = getattr(style, "element", None)
        if element is not None:
            outline = element.find(qn("w:pPr"))
            if outline is not None:
                level_el = outline.find(qn("w:outlineLvl"))
                if level_el is not None:
                    value = level_el.get(qn("w:val"))
                    try:
                        level = int(value) + 1
                    except (TypeError, ValueError):
                        continue
                    if 1 <= level <= 9:
                        return level
    return None


def _heading_level(paragraph: Paragraph) -> int | None:
    text = paragraph.text.strip()
    if not text or len(text) > _HEADING_MAX_CHARS:
        return None

    level = _outline_level_from_xml(paragraph)
    if level is not None:
        return level

    level = _outline_level_from_styles(paragraph)
    if level is not None:
        return level

    for pattern in _HEADING_PATTERNS:
        if pattern.match(text):
            if len(text.split()) <= 15:
                return 1 
            return None

    return None


def _table_to_markdown(table: Table) -> str:
    rows: list[list[str]] = []
    for row in table.rows:
        cells: list[str] = []
        column = 0
        for cell_xml in row._tr.tc_lst:
            properties = cell_xml.tcPr
            grid_span = properties.gridSpan
            try:
                span = int(grid_span.val) if grid_span is not None else 1
            except (TypeError, ValueError):
                span = 1

            vertical_merge = properties.vMerge
            is_continuation = (
                vertical_merge is not None
                and (vertical_merge.val is None or vertical_merge.val != "restart")
            )
            cell_text = _collapse_repeated_tokens(" ".join(cell_xml.itertext()).strip())
            cell_text = cell_text.replace("\n", " ").replace("|", "\\|")

            for offset in range(span):
                if is_continuation or offset > 0:
                    cells.append("")
                else:
                    cells.append(cell_text)
            column += span
        # Some Word exports repeat the value of a visually merged cell in
        # adjacent XML cells without writing a ``gridSpan`` marker. Keep the
        # first value and blank the repetitions so Markdown does not show a
        # fake list of identical headers.
        previous = None
        for index, value in enumerate(cells):
            if value and value == previous:
                cells[index] = ""
            elif value:
                previous = value
        rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, sep, *body])


def _collapse_repeated_tokens(text: str) -> str:
    """Collapse text duplicated by merged-cell exports.

    Some DOCX producers put the same logical cell text into one XML cell
    several times instead of using ``gridSpan``. Only exact repetitions of the
    complete token sequence are collapsed; ordinary repeated words remain.
    """
    tokens = text.split()
    for repetitions in range(2, min(20, len(tokens)) + 1):
        if len(tokens) % repetitions:
            continue
        width = len(tokens) // repetitions
        unit = tokens[:width]
        if unit * repetitions == tokens:
            return " ".join(unit)
    return text


def read_docx(content: bytes, *, name: str = "<bytes>") -> list[DocxBlock]:
    """Read one .docx into a flat list of structural blocks.

    Each block is a heading, a paragraph, or a table. Tables are emitted as
    a single markdown block wrapped in ``<<<TABLE>>>…<<<END_TABLE>>>``, so
    the chunker can keep them intact.

    Args:
        content: Raw .docx bytes.
        name: Filename for diagnostics.

    Returns:
        Blocks in document order.
    """
    started = time.monotonic()
    try:
        doc = Document(io.BytesIO(content))
    except Exception:
        logger.exception(f"[docx] failed to open '{name}'")
        raise

    blocks: list[DocxBlock] = []
    current_heading = ""
    paragraphs = 0
    tables = 0
    headings = 0

    for block in iter_blocks(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            paragraphs += 1
            level = _heading_level(block)
            if level is not None:
                headings += 1
                current_heading = text
                blocks.append(DocxBlock(kind="heading", text=text, level=level))
            else:
                blocks.append(
                    DocxBlock(
                        kind="paragraph",
                        text=text,
                        section_title=current_heading,
                    )
                )
        else:
            md = _table_to_markdown(block)
            if not md:
                continue
            tables += 1
            heading_prefix = f"**{current_heading}**\n" if current_heading else ""
            blocks.append(
                DocxBlock(
                    kind="table",
                    text=f"{TABLE_OPEN}\n{heading_prefix}{md}\n{TABLE_CLOSE}",
                    section_title=current_heading,
                )
            )

    elapsed = time.monotonic() - started
    return blocks
