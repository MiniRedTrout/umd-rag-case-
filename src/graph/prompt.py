
import hashlib
from pathlib import Path

from src.prompts.markdown import read_text_file
from src.utils.logger import logger

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


def _read_md(relative_path: str) -> str:
    full_path = PROMPT_DIR / relative_path
    if not full_path.exists():
        logger.warning(f"Prompt file not found: {full_path}")
        return ""
    return read_text_file(full_path)


def _search_instructions_block() -> str:
    body = _read_md("instructions/classify.md")
    return "<search_instructions>\n" + body + "\n</search_instructions>"


def _synthesize_instructions_block() -> str:
    body = _read_md("instructions/synthesize.md")
    return "<synthesize_instructions>\n" + body + "\n</synthesize_instructions>"


def _reformulate_instructions_block() -> str:
    body = _read_md("instructions/reformulate.md")
    return "<reformulate_instructions>\n" + body + "\n</reformulate_instructions>"


def get_classify_prompt() -> str:
    return _read_md("instructions/classify.md")


def build_agent_system_prompt() -> str:
    parts: list[str] = []
    parts.append("<role>\n" + _read_md("agent_system.md") + "\n</role>")
    parts.append(_search_instructions_block())
    parts.append(_synthesize_instructions_block())
    parts.append(_reformulate_instructions_block())

    return "\n\n".join(parts)


_AGENT_SYSTEM_PROMPT_CACHE: str | None = None


def get_agent_system_prompt() -> str:
    global _AGENT_SYSTEM_PROMPT_CACHE
    if _AGENT_SYSTEM_PROMPT_CACHE is None:
        _AGENT_SYSTEM_PROMPT_CACHE = build_agent_system_prompt()
    return _AGENT_SYSTEM_PROMPT_CACHE


def prompt_version_hash() -> str:
    """Return the version of the prompt actually used by the document RAG agent."""
    return hashlib.sha256(get_agent_system_prompt().encode("utf-8")).hexdigest()[:12]
