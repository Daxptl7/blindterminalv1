"""
main.py — RAG Pipeline Orchestrator
======================================
The primary entry point that ties all components together:
- EmbeddingService (Gemini / local)
- PineconeService (vector database)
- RankingService (Cohere Rerank v3)
- GeminiAgent (LLM response generation)
- GuardrailsEngine (NeMo Guardrails safety layer)
- IngestionProcessor (document ingestion)
"""

import sys
import logging
from pathlib import Path

# Ensure the Blindterminal root is importable
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rag_pipeline.services.config import Config
from rag_pipeline.services.retrieval.embedding import EmbeddingService
from rag_pipeline.services.retrieval.pinecone_service import PineconeService
from rag_pipeline.services.retrieval.ranking_service import RankingService
from rag_pipeline.ingestion.chunking.splitter import TextSplitter
from rag_pipeline.ingestion.loaders.md_loader import MarkdownLoader
from rag_pipeline.ingestion.processor import IngestionProcessor
from rag_pipeline.guardrails.rails import GuardrailsEngine

# Import GeminiAgent from the existing services module
from Blindterminal.services.gemini_agent import GeminiAgent

logger = logging.getLogger("RAGMain")


class RAGPipelineMain:
    """
    Central orchestrator that initializes all RAG components
    and provides high-level methods for ingestion and querying.
    """

    def __init__(self):
        self.config = Config()

        # ── Initialize Retrieval Services ────────────────────
        self.embedding_service = EmbeddingService(
            provider=self.config.embedding_provider,
            api_key=self.config.gemini_api_key
        )

        self.pinecone_service = PineconeService(
            api_key=self.config.pinecone_api_key,
            index_name=self.config.pinecone_index_name
        )

        self.ranking_service = RankingService(
            api_key=self.config.cohere_api_key
        )

        # ── Initialize LLM Agent ────────────────────────────
        self.gemini_agent = GeminiAgent(
            api_key=self.config.gemini_api_key,
            model_name=self.config.gemini_model_name
        )

        # ── Initialize Ingestion Components ──────────────────
        self.splitter = TextSplitter(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap
        )

        self.md_loader = MarkdownLoader(gemini_agent=self.gemini_agent)

        self.ingestion_processor = IngestionProcessor(
            embedding_service=self.embedding_service,
            pinecone_service=self.pinecone_service,
            splitter=self.splitter,
            md_loader=self.md_loader
        )

        # ── Initialize Guardrails ────────────────────────────
        self.guardrails = GuardrailsEngine(
            retriever_fn=self._retrieve_context,
            generator_fn=self._generate_response
        )

        logger.info("RAG Pipeline Main orchestrator initialized.")

    # ── Query Interface ──────────────────────────────────────

    def ask_question(self, query: str, top_k: int = 3, filters: dict = None) -> str:
        """
        Processes a user query through guardrails, retrieval, reranking, and LLM generation.

        Args:
            query: The user's question.
            top_k: Number of top results to retrieve.
            filters: Optional metadata filters (standard, subject, chapter).

        Returns:
            The generated response string.
        """
        if self.guardrails.is_active:
            return self.guardrails.process_query(query)

        # Direct RAG execution if guardrails are inactive
        context = self._retrieve_context(query=query, top_k=top_k, filters=filters)
        if not context or not context.strip():
            return "I couldn't find relevant information in the textbooks for your question."

        return self._generate_response(query=query, context=context)

    # ── Ingestion Interface ──────────────────────────────────

    def ingest_textbooks(self, directory: str = None) -> dict:
        """
        Ingests all textbook files from a directory.

        Args:
            directory: Path to the textbooks directory. Defaults to config textbooks_dir.

        Returns:
            Summary dict with 'indexed' and 'failed' counts.
        """
        target_dir = directory or str(self.config.textbooks_dir)
        return self.ingestion_processor.ingest_directory(target_dir)

    def ingest_file(self, file_path: str, tags: dict = None) -> bool:
        """Ingests a single file into the vector database."""
        return self.ingestion_processor.ingest_file(file_path, tags=tags)

    # ── Internal RAG Methods (registered with Guardrails) ────

    def _retrieve_context(self, query: str, top_k: int = 3, filters: dict = None) -> str:
        """
        Retrieves and reranks relevant context from Pinecone.
        This method is registered as a custom action in NeMo Guardrails.
        """
        try:
            # 1. Embed the query
            query_vec = self.embedding_service.get_embedding(query)
            if query_vec.size == 0:
                return ""

            # 2. Query Pinecone for candidates
            fetch_k = max(top_k * 3, 10)
            results = self.pinecone_service.query(query_vec, top_k=fetch_k, filters=filters)
            if not results:
                return ""

            candidate_chunks = [text for text, score in results]

            # 3. Rerank with Cohere
            if self.ranking_service.is_available:
                reranked = self.ranking_service.rerank(
                    query=query,
                    candidate_chunks=candidate_chunks,
                    top_n=top_k,
                    relevance_threshold=0.35
                )
                return "\n---\n".join(reranked)

            # Fallback: return top-k by vector similarity
            return "\n---\n".join(candidate_chunks[:top_k])

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""

    def _generate_response(self, query: str, context: str) -> str:
        """
        Generates a response using the Gemini agent.
        This method is registered as a custom action in NeMo Guardrails.
        """
        return self.gemini_agent.generate_response(query, context)
