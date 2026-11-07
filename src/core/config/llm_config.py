from dotenv import load_dotenv
from opensearchpy import AsyncOpenSearch

from src.core import settings
from src.models.embedders.embedding_loader import GPNOpenAIEmbeddings, load_embeddings
from src.models.embedders.vllm_embedding_loader import load_vllm_embeddings
from src.models.embedders.e5_instructed import E5Instructed
from src.rag.vectorestores.opensearch import AsyncOpenSearchWrapper
from src.rag.vectorestores.index_schema import sanitize_embedding_field_name
from src.security.ssl_manager import ssl_manager
from src.utils.env_wrapper import get_env_var
from src.rag.search import ChunkRepository

load_dotenv()


class LLMSettings:
    """Settings for the RAG components."""

    E5_EMBEDDINGS: E5Instructed = E5Instructed(
    load_embeddings(settings.EMB_E5_MODEL, base_url=settings.VLLM_EMB_URL_E5)
    )
    BGE_EMBEDDINGS: GPNOpenAIEmbeddings = load_vllm_embeddings(
        settings.EMB_BGE_MODEL, base_url=settings.VLLM_EMB_URL_BGE
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
        embedding_field_name_e5=settings.VLLM_E5_NAME,
        embedding_field_name_bge=settings.VLLM_BGE_NAME,
        index_name=settings.OPENSEARCH_COLLECTION,
        client=client,
    )

    documents_opensearch_wrapper = AsyncOpenSearchWrapper(
        embeddings_e5=E5_EMBEDDINGS,
        embeddings_bge=BGE_EMBEDDINGS,
        embedding_field_name_e5=settings.VLLM_E5_NAME,
        embedding_field_name_bge=settings.VLLM_BGE_NAME,
        index_name=settings.DOCS_OPENSEARCH_COLLECTION,
        client=client,
    )

    embedding_field_name_e5=sanitize_embedding_field_name(settings.VLLM_E5_NAME),
    embedding_field_name_bge=sanitize_embedding_field_name(settings.VLLM_BGE_NAME),

    REPOSITORY = ChunkRepository(opensearch_wrapper)


llm_settings = LLMSettings()