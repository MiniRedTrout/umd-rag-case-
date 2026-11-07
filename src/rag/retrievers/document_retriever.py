from __future__ import annotations

import asyncio
from collections.abc import Sequence

from langchain_core.documents import Document
from scipy.spatial.distance import cosine

from src.core import settings
from src.rag.rankers.document_reciprocal_rank import (
    document_reciprocal_rank_fusion as reciprocal_rank_fusion,
)
from src.rag.vectorestores.opensearch import AsyncOpenSearchWrapper
from src.utils.document_codes import extract_codes
from src.utils.logger import logger
from src.utils.splitter import reduce_length


class DocumentRetriever:
    def __init__(self, opensearch_wrapper: AsyncOpenSearchWrapper, index_name: str) -> None:
        self.opensearch_wrapper = opensearch_wrapper
        self.index_name = index_name

    def invoke(self, question: str, knowledge_bases: Sequence[str]) -> dict:
        return asyncio.run(self.ainvoke(question, knowledge_bases))

    async def find_by_code(
        self, code: str, knowledge_bases: Sequence[str], k: int
    ) -> list[dict]:
        """Return the cards whose cipher is ``code`` or starts with it by whole groups.

        No knowledge bases means nothing is visible, the same reading
        ``SearchGateway`` gives an empty grant.
        """
        if not code or not knowledge_bases:
            return []
        documents = await self.opensearch_wrapper.query_documents_by_code(
            code,
            index_name=self.index_name,
            k=k,
            knowledge_bases=knowledge_bases,
        )
        logger.info(
            f"Document retriever: {len(documents)} cards for code={code!r}"
        )
        return [self._as_match(document, score=0.0, reason="code") for document in documents]

    async def find_by_code_prefix(
            self, code: str, knowledge_bases: Sequence[str], k: int
        ) -> list[dict]:
        """Return the cards whose cipher starts with ``code``.

        A prefix match, not an exact one answers with every document
        whose cipher begins with those groups. No knowledge bases means nothing is
        visible, the same reading ``SearchGateway`` gives an empty grant.
        """
        if not code or not knowledge_bases:
            return []
        documents = await self.opensearch_wrapper.query_documents_by_code_prefix(
            code,
            index_name=self.index_name,
            k=k,
            knowledge_bases=knowledge_bases,
        )
        logger.info(
            f"Document retriever: {len(documents)} cards for prefix={code!r}"
        )
        return [self._as_match(document, score=0.0, reason="prefix") for document in documents]
    
    async def ainvoke(self, question: str, knowledge_bases: Sequence[str]) -> dict:
        if not knowledge_bases:
            logger.info("Document retriever: no knowledge bases granted")
            return {"matched": False}

        query_codes = extract_codes(question)

        user_e5, user_bge = await asyncio.gather(
            self.opensearch_wrapper.embeddings_e5.aembed_query(
                reduce_length(question)
            ),
            self.opensearch_wrapper.embeddings_bge.aembed_query(
                reduce_length(question)
            ),
        )

        e5_docs, bge_docs, keyword_docs = await asyncio.gather(
            self.opensearch_wrapper.query_documents_vector(
                question=question,
                index_name=self.index_name,
                k=settings.DOCS_RETRIEVAL_K,
                use_e5=True,
                vector=user_e5,
                knowledge_bases=knowledge_bases,
            ),
            self.opensearch_wrapper.query_documents_vector(
                question=question,
                index_name=self.index_name,
                k=settings.DOCS_RETRIEVAL_K,
                use_e5=False,
                vector=user_bge,
                knowledge_bases=knowledge_bases,
            ),
            self.opensearch_wrapper.query_documents_keyword(
                question=question,
                index_name=self.index_name,
                k=settings.DOCS_KEYWORD_RETRIEVAL_K,
                codes=sorted(query_codes),
                knowledge_bases=knowledge_bases,
            ),
        )

        logger.info(
            f"Document retriever: candidates e5={len(e5_docs)}, bge={len(bge_docs)}, "
            f"keyword={len(keyword_docs)}, query codes={sorted(query_codes)}"
        )

        documents = reciprocal_rank_fusion([e5_docs, bge_docs, keyword_docs])
        if not documents:
            logger.info("Document retriever: no candidates found")
            return {"matched": False}

        if settings.DOCS_CODE_EXACT_MATCH and query_codes:
            for document in documents:
                matched_codes = query_codes & set(document.metadata.get("codes") or [])
                if matched_codes:
                    logger.info(
                        f"Document match by code: codes={sorted(matched_codes)}, "
                        f"title='{document.metadata['title']}'"
                    )
                    return self._as_match(document, score=0.0, reason="code")

        candidates = documents[: settings.DOCS_RERANK_TOP_N]

        distances = await self._get_distances(
            user_e5,
            user_bge,
            [document.page_content for document in candidates],
        )

        for document, distance in zip(candidates, distances):
            logger.info(
                f"Document candidate distance={distance:.4f}, "
                f"title='{document.metadata['title']}'"
            )

        best_index = min(range(len(distances)), key=lambda index: distances[index])
        best_distance = distances[best_index]
        best_document = candidates[best_index]

        if best_distance <= settings.DOCS_SIMILARITY_THRESHOLD:
            logger.info(
                f"Document match by similarity distance={best_distance:.4f}, "
                f"threshold={settings.DOCS_SIMILARITY_THRESHOLD}, "
                f"title='{best_document.metadata['title']}'"
            )
            return self._as_match(best_document, score=best_distance, reason="similarity")

        logger.info(
            f"Document retriever: no match, best distance={best_distance:.4f} > "
            f"threshold={settings.DOCS_SIMILARITY_THRESHOLD}, "
            f"candidate='{best_document.metadata['title']}'"
        )
        return {"matched": False}

    @staticmethod
    def _as_match(document: Document, score: float, reason: str) -> dict:
        return {
            "matched": True,
            "title": document.metadata["title"],
            "link": document.metadata["link"],
            "description": document.page_content,
            "score": score,
            "reason": reason,
        }

    async def _get_distances(
        self,
        user_e5: list[float],
        user_bge: list[float],
        document_texts: list[str],
    ) -> list[float]:
        document_texts = [reduce_length(text) for text in document_texts]

        docs_e5, docs_bge = await asyncio.gather(
            self.opensearch_wrapper.embeddings_e5.aembed_documents(document_texts),
            self.opensearch_wrapper.embeddings_bge.aembed_documents(document_texts),
        )

        return [
            (cosine(user_e5, doc_e5) + cosine(user_bge, doc_bge)) / 2
            for doc_e5, doc_bge in zip(docs_e5, docs_bge)
        ]
