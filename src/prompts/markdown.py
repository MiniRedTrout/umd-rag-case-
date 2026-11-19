"""Canonical markdown / frontmatter readers for prompt assets."""

from __future__ import annotations

from pathlib import Path


def read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Return (frontmatter dict, body) from a markdown file with optional YAML frontmatter."""
    stripped = text.strip()
    if not stripped.startswith("---"):
        return {}, stripped

    parts = stripped.split("---", 2)
    if len(parts) < 3:
        return {}, stripped

    meta: dict[str, str] = {}
    for line in parts[1].strip().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            meta[key] = value

    return meta, parts[2].strip()


def read_markdown_body(path: Path, *, fallback: str = "") -> str:
    """Read a markdown file and return body text without YAML frontmatter."""
    if not path.exists():
        return fallback
    _, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    return body.strip() or fallback
