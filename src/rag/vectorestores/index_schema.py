from __future__ import annotations

import re

from opensearchpy import AsyncOpenSearch

from src.utils.logger import logger


FAQ_INDEX_PROPERTIES = {
    "answer": {"type": "text"},
    "source": {"type": "keyword"},
}

DOCUMENTS_INDEX_PROPERTIES = {
    "link": {"type": "keyword"},
    "title": {
        "type": "text",
        "analyzer": "russian_index_analyzer",
        "search_analyzer": "russian_search_analyzer",
        "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
    },
    "text": {
        "type": "text",
        "analyzer": "russian_index_analyzer",
        "search_analyzer": "russian_search_analyzer",
    },
}


def sanitize_embedding_field_name(field_name: str) -> str:
    sanitized = re.sub(r"[\s\"*\\<>|,/?]+", "_", field_name).strip("_")
    return f"embedding_{sanitized}" if sanitized else "embedding_vector"


def _russian_analysis_settings() -> dict:
    return {
        "filter": {
            "russian_stop": {"type": "stop", "stopwords": "_russian_"},
            "russian_stemmer": {"type": "stemmer", "language": "russian"},
        },
        "analyzer": {
            "russian_index_analyzer": {
                "tokenizer": "standard",
                "filter": ["lowercase", "russian_stop", "russian_stemmer"],
            },
            "russian_search_analyzer": {
                "tokenizer": "standard",
                "filter": ["lowercase", "russian_stop", "russian_stemmer"],
            },
        },
    }


def _corpus_index_properties(embedding_fields: dict[str, int]) -> dict:
    properties: dict = {
        "id": {"type": "keyword"},
        "doc_id": {"type": "keyword"},
        "path": {"type": "keyword"},
        "source_format": {"type": "keyword"},
        "title": {
            "type": "text",
            "analyzer": "russian_index_analyzer",
            "search_analyzer": "russian_search_analyzer",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
        },
        "url": {"type": "keyword"},
        "chunk_index": {"type": "integer"},
        "chunk_total": {"type": "integer"},
        "text": {
            "type": "text",
            "analyzer": "russian_index_analyzer",
            "search_analyzer": "russian_search_analyzer",
        },
    }
    for field_name, dimension in embedding_fields.items():
        properties[field_name] = {
            "type": "knn_vector",
            "dimension": dimension,
            "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "faiss"},
        }
    return properties


async def initialize_index(
    index_name: str,
    embedding_fields: dict[str, int],
    client: AsyncOpenSearch,
    space_type: str = "l2",
    is_delete_existing: bool = True,
    additional_properties: dict | None = None,
    analysis: dict | None = None,
) -> None:
    exists = await client.indices.exists(index=index_name)
    if exists and is_delete_existing:
        logger.info(f"Index already exists, deleting index: {index_name}")
        await client.indices.delete(index=index_name)
    elif exists and not is_delete_existing:
        logger.info(f"Index already exists, using index: {index_name}")
        return

    analysis_block = _russian_analysis_settings()
    if analysis:
        for key, value in analysis.items():
            if isinstance(value, dict) and isinstance(analysis_block.get(key), dict):
                analysis_block[key] = {**analysis_block[key], **value}
            else:
                analysis_block[key] = value

    settings = {
        "index": {
            "knn": True,
            "knn.space_type": space_type,
            "refresh_interval": "-1",
            "number_of_replicas": 0
        },
        "analysis": analysis_block,
    }

    safe_fields = {
        sanitize_embedding_field_name(name): dimension
        for name, dimension in embedding_fields.items()
    }
    properties = _corpus_index_properties(safe_fields)
    if additional_properties:
        properties.update(additional_properties)

    index_body = {
        "settings": settings,
        "mappings": {"properties": properties},
    }

    try:
        await client.indices.create(index=index_name, body=index_body)
        logger.info(
            f"Created OpenSearch index '{index_name}' with embedding fields: "
            f"{list(embedding_fields.keys())}"
        )
    except Exception as e:
        logger.error(f"Failed to create OpenSearch index '{index_name}': {e}")
        raise