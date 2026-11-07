import re
from collections.abc import Sequence

_SLUG_DROP = re.compile(r"[^\w\s-]", flags=re.UNICODE)
_SLUG_SEP = re.compile(r"[\s_]+", flags=re.UNICODE)
_SLUG_DASHES = re.compile(r"-{2,}")


def slugify(value: str) -> str:
    """Normalize a knowledge base directory name to a stable keyword form.

    Underscores and spaces become hyphens, so the folder ``itsolutions_user``
    indexes as ``itsolutions-user``. The assistant (assistant-chat-nlp) and the
    itsolution importer run the same function on both sides, so the index and
    every request filter must use it too, or a client sending the folder name
    as-is matches nothing.

    Args:
        value: Directory name to normalize.

    Returns:
        Stable keyword form.
    """
    text = _SLUG_DROP.sub(" ", value.strip().casefold())
    text = _SLUG_SEP.sub("-", text.strip())
    return _SLUG_DASHES.sub("-", text).strip("-")


def slugify_knowledge_bases(bases: Sequence[str] | None) -> list[str]:
    """Slugify requested knowledge bases the way the importer writes them.

    De-duplicates and keeps first-seen order. A blank name is kept as ``""``:
    it is how a request reads documents imported without a knowledge base level
    (and those the startup backfill set to ``""``).

    Args:
        bases: Raw knowledge base names from the request.

    Returns:
        Slugified bases; an empty list reads nothing.
    """
    seen: set[str] = set()
    granted: list[str] = []
    for base in bases or ():
        slug = slugify(base)
        if slug not in seen:
            seen.add(slug)
            granted.append(slug)
    return granted
