from dotenv import load_dotenv
from opensearchpy import AsyncOpenSearch
from langchain_openai import OpenAIEmbeddings

from src.core import settings
from src.models.embedders.e5_instructed import E5Instructed
from src.rag.vectorestores.opensearch import AsyncOpenSearchWrapper
from src.rag.vectorestores.index_schema import sanitize_embedding_field_name
from src.security.ssl_manager import ssl_manager
from src.rag.search import ChunkRepository
load_dotenv()


class LLMSettings:
    """Settings for the RAG components."""
    E5_EMBEDDINGS = OpenAIEmbeddings(
        model=settings.VLLM_E5_NAME,
        base_url=settings.VLLM_EMB_URL_E5,
        api_key="not-needed",
        chunk_size=4,
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
    )
    if settings.VLLM_E5_NAME == "intfloat/multilingual-e5-base":
        E5_EMBEDDINGS = E5Instructed(E5_EMBEDDINGS)
    BGE_EMBEDDINGS = OpenAIEmbeddings(
        model=settings.VLLM_BGE_NAME,
        base_url=settings.VLLM_EMB_URL_BGE,
        api_key="not-needed",
        chunk_size=4,
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
    )
    client = AsyncOpenSearch(
        hosts=settings.OPENSEARCH_HOSTS,
        http_auth=(
            f"{settings.OPENSEARCH_USER}:{settings.OPENSEARCH_PASSWORD}"
            if settings.OPENSEARCH_USE_AUTH
            else None
        ),
        use_ssl=settings.OPENSEARCH_USE_SSL,
        verify_certs=settings.OPENSEARCH_VERIFY_CERTS,
        ssl_assert_hostname=settings.OPENSEARCH_SSL_ASSERT_HOSTNAME,
        ssl_show_warn=settings.OPENSEARCH_SSL_SHOW_WARN,
        ssl_context=ssl_manager.opensearch,
    )

    opensearch_wrapper = AsyncOpenSearchWrapper(
        embeddings_e5=E5_EMBEDDINGS,
        embeddings_bge=BGE_EMBEDDINGS,
        embedding_field_name_e5=sanitize_embedding_field_name(settings.VLLM_E5_NAME),
        embedding_field_name_bge=sanitize_embedding_field_name(settings.VLLM_BGE_NAME),
        index_name=settings.OPENSEARCH_COLLECTION,
        client=client,
    )

    documents_opensearch_wrapper = AsyncOpenSearchWrapper(
        embeddings_e5=E5_EMBEDDINGS,
        embeddings_bge=BGE_EMBEDDINGS,
        embedding_field_name_e5=sanitize_embedding_field_name(settings.VLLM_E5_NAME),
        embedding_field_name_bge=sanitize_embedding_field_name(settings.VLLM_BGE_NAME),
        index_name=settings.DOCS_OPENSEARCH_COLLECTION,
        client=client,
    )

    REPOSITORY = ChunkRepository(opensearch_wrapper)


llm_settings = LLMSettings()
