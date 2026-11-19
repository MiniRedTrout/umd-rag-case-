from pathlib import Path

from dotenv import load_dotenv

from src.utils.env_wrapper import get_bool_env_var, get_env_var

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / "applications.conf")

SECTIONAL_PARSING_ENABLED = get_bool_env_var("SECTIONAL_PARSING_ENABLED", True)
TEXT_SPLITTER = get_env_var("TEXT_SPLITTER", "recursive").lower()
if TEXT_SPLITTER not in {"recursive", "markdown"}:
    raise ValueError("TEXT_SPLITTER must be either 'recursive' or 'markdown'")

