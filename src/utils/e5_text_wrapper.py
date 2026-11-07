E5_PASSAGE_PREFIX = "passage: "
E5_QUERY_PREFIX = "query: "


def add_passage(texts: list[str]) -> list[str]:
    """Prefixes texts for E5 passage embedding without touching the source documents."""
    return [f"{E5_PASSAGE_PREFIX}{text}" for text in texts]


def add_query(texts: list[str]) -> list[str]:
    """Prefixes texts for E5 query embedding without touching the source documents."""
    return [f"{E5_QUERY_PREFIX}{text}" for text in texts]
