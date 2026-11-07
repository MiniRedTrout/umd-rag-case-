from collections import defaultdict

from langchain_core.documents import Document

from src.core import settings
from src.core.config.agent_config import get_agent_config


def group_chunk_indices(chunks: list[tuple[str, int]]) -> list[tuple[str, list[int]]]:
    """Group selected chunks into contiguous, widened read windows.

    Args:
        chunks: list[tuple[str, int]] — Paths and indices of chunks to read.

    Returns:
        list[tuple[str, list[int]]] — Path and ascending indices for each run.
    """
    ids_expansion: dict[str, set[int]] = defaultdict(set)
    routing_ids: dict[str, list[int]] = defaultdict(list)

    expansion = get_agent_config().retrieval.chunk_expansion
    for path, index in chunks:
        routing_ids[path].append(int(index))
        for offset in range(-expansion, expansion + 1):
            ids_expansion[path].add(int(index) + offset)

    grouped: dict[str, list[list[int]]] = {
        path: _group_indicies(indices, routing_ids[path])
        for path, indices in ids_expansion.items()
    }

    merged_id_source_pairs = [
        (path, ids) for path, runs in grouped.items() for ids in runs
    ]

    if not get_agent_config().retrieval.group_one_document_context:
        sorted_merged_id_source_pairs = []
        for path, index in chunks:
            for pair in merged_id_source_pairs:
                if (
                    path == pair[0]
                    and int(index) in pair[1]
                    and pair not in sorted_merged_id_source_pairs
                ):
                    sorted_merged_id_source_pairs.append(pair)
        return sorted_merged_id_source_pairs

    return merged_id_source_pairs


def get_chunk_indices_list(docs: list[Document]) -> list[tuple[str, list[int]]]:
    """Groups ids with window expansion.

    Capable of sorting within one document and within the whole payload.

    Args:
        docs: The list of documents.

    Returns:
        A list of tuples - filename of the document and respective chunk ids.
    """
    ids_expansion = defaultdict(set)
    routing_ids = defaultdict(list)

    for doc in docs:
        routing_ids[doc.metadata["source"]].append(int(doc.metadata["id"]))
        for offset in range(-settings.CHUNK_WINDOW_SIZE, settings.CHUNK_WINDOW_SIZE + 1):
            ids_expansion[doc.metadata["source"]].add(int(doc.metadata["id"]) + offset)

    for key in ids_expansion:
        ids_expansion[key] = _group_indicies_old(ids_expansion[key], routing_ids[key])
        #                                            👆 _old

    merged_id_source_pairs = [(key, ids) for key, vals in ids_expansion.items() for ids in vals]

    if not settings.GROUP_ONE_DOCUMENT_CONTEXT:
        sorted_merged_id_source_pairs = []
        for doc in docs:
            for pair in merged_id_source_pairs:
                if (
                    doc.metadata["source"] == pair[0]
                    and int(doc.metadata["id"]) in pair[1]
                    and pair not in sorted_merged_id_source_pairs
                ):
                    sorted_merged_id_source_pairs.append(pair)
        return sorted_merged_id_source_pairs

    return merged_id_source_pairs


def _group_indicies_old(ids: set[int], routing_ids: list[int]) -> list[int]:
    ids = sorted(ids)

    grouped = [[ids[0]]]

    for x in ids[1:]:
        if x == grouped[-1][-1] + 1:
            grouped[-1].append(x)
        else:
            grouped.append([x])

    if not settings.SORT_ASCENDING_ONE_DOCUMENT_CONTEXT:
        sorted_groups = []

        for route_id in routing_ids:
            for gr in grouped:
                if route_id in gr and gr not in sorted_groups:
                    sorted_groups.append(gr)

        return sorted_groups

    return grouped


def _group_indicies(ids: set[int], routing_ids: list[int]) -> list[list[int]]:
    ordered = sorted(ids)

    grouped = [[ordered[0]]]

    for x in ordered[1:]:
        if x == grouped[-1][-1] + 1:
            grouped[-1].append(x)
        else:
            grouped.append([x])

    if not get_agent_config().retrieval.sort_ascending_one_document_context:
        sorted_groups = []

        for route_id in routing_ids:
            for gr in grouped:
                if route_id in gr and gr not in sorted_groups:
                    sorted_groups.append(gr)

        return sorted_groups

    return grouped