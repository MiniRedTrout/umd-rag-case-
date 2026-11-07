

import sys
from pathlib import Path

from src.rag.loaders.docx_reader import TABLE_CLOSE, TABLE_OPEN, read_docx


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/vcr/test_docx")
    files = sorted(src.rglob("*.docx"))
    if not files:
        print(f"No .docx files in {src}")
        return

    for f in files:
        print("=" * 80)
        print(f"FILE: {f}")
        print("=" * 80)

        blocks = read_docx(f.read_bytes())
        print(f"\n--- {len(blocks)} blocks ---\n")

        tables = 0
        paragraphs = 0
        for i, block in enumerate(blocks, 1):
            if TABLE_OPEN in block:
                tables += 1
                inner = (
                    block.replace(TABLE_OPEN, "")
                    .replace(TABLE_CLOSE, "")
                    .strip()
                )
                lines = inner.splitlines()
                preview = "\n".join(lines[:15])
                more = f"\n... ({len(lines) - 15} more lines)" if len(lines) > 15 else ""
                print(f"[{i}] TABLE ({len(lines)} lines):")
                print(preview + more)
            else:
                paragraphs += 1
                text = block.strip()
                if len(text) > 200:
                    text = text[:200] + "…"
                print(f"[{i}] PARAGRAPH: {text}")
            print()

        print(f"\nSummary: {paragraphs} paragraphs, {tables} tables\n")


if __name__ == "__main__":
    main()