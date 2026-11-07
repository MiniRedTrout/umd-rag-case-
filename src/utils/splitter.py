from src.utils.embedding_token_limit import get_embedding_max_tokens, local_reduce_length
from functools import lru_cache

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


def reduce_length(text: str) -> str:
    max_tokens = get_embedding_max_tokens()
    if max_tokens is None or text.strip() == "":
        return text
    return local_reduce_length(text, max_tokens)


@lru_cache(maxsize=1)
def _estimation_encoding():
    """Return the stable local encoder used only for batching estimates."""
    return tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    """Estimate input size for batching without claiming model-token accuracy."""
    if not text:
        return 0
    return len(_estimation_encoding().encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Cut a text down to a token budget.

    Args:
        text: str — Text to shorten.
        max_tokens: int — Tokens the result may occupy.

    Returns:
        str — Leading span of the text that fits the budget.
    """
    if max_tokens < 1:
        return ""

    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=max_tokens,
            chunk_overlap=0,
        )
        chunks = splitter.split_text(text)
    except Exception as exc:
        raise RuntimeError("Token-accurate truncation is unavailable") from exc

    return chunks[0] if chunks else ""
