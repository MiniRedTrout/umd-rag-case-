import io
from pathlib import Path
from zipfile import ZipFile

from src.scripts.process_umd import _build_archive


def test_build_archive_accepts_markdown_and_skips_office_lock_files(tmp_path: Path) -> None:
    (tmp_path / "course.md").write_text("# Course\n\nText", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Notes", encoding="utf-8")
    (tmp_path / "~$course.docx").write_bytes(b"lock file")

    archive, files = _build_archive(tmp_path, None)

    assert {path.name for path in files} == {"course.md", "notes.txt"}
    with ZipFile(io.BytesIO(archive)) as zipped:
        assert sorted(zipped.namelist()) == ["course.md", "notes.txt"]

