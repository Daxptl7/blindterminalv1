"""
client.py — RAG Pipeline Gateway Client
==========================================
The user-facing entry point for the RAG pipeline.
Provides simple functions to:
1. Query the textbook database.
2. Ingest new textbook files.
3. Check pipeline status.
"""

import sys
import logging
from pathlib import Path

# Ensure the Blindterminal root is importable
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rag_pipeline.services.main import RAGPipelineMain

logger = logging.getLogger("GatewayClient")

# ── Singleton Pipeline Instance ──────────────────────────────
_pipeline_instance = None


def get_pipeline() -> RAGPipelineMain:
    """
    Returns the singleton RAG pipeline instance.
    Initializes it on first call.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        try:
            _pipeline_instance = RAGPipelineMain()
            logger.info("RAG Pipeline initialized via Gateway Client.")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Pipeline: {e}")
    return _pipeline_instance


def ask(query: str, top_k: int = 3, filters: dict = None) -> str:
    """
    Sends a question to the RAG pipeline and returns the answer.

    Args:
        query: The user's question about textbook content.
        top_k: Number of relevant chunks to retrieve.
        filters: Optional metadata filters (e.g., {"standard": "10", "subject": "science"}).

    Returns:
        The generated answer string.

    Example:
        >>> from rag_pipeline.gateway.client import ask
        >>> answer = ask("What is a chemical reaction?")
        >>> print(answer)
    """
    pipeline = get_pipeline()
    if not pipeline:
        return "RAG Pipeline failed to initialize. Please check configuration."
    return pipeline.ask_question(query, top_k=top_k, filters=filters)


def ingest(directory: str = None) -> dict:
    """
    Ingests textbook files from a directory into the vector database.

    Args:
        directory: Path to the textbooks directory. Uses default if not specified.

    Returns:
        Summary dict with 'indexed' and 'failed' counts.

    Example:
        >>> from rag_pipeline.gateway.client import ingest
        >>> result = ingest("data/textbooks")
        >>> print(result)
    """
    pipeline = get_pipeline()
    if not pipeline:
        return {"indexed": 0, "failed": 0, "error": "Pipeline not initialized"}
    return pipeline.ingest_textbooks(directory)


def ingest_file(file_path: str, tags: dict = None) -> bool:
    """
    Ingests a single textbook file.

    Args:
        file_path: Path to the textbook file.
        tags: Optional metadata tags.

    Returns:
        True if ingestion succeeded.
    """
    pipeline = get_pipeline()
    if not pipeline:
        return False
    return pipeline.ingest_file(file_path, tags=tags)


def status() -> dict:
    """
    Returns the current pipeline status.
    """
    pipeline = get_pipeline()
    if not pipeline:
        return {"status": "offline", "error": "Pipeline not initialized"}

    return {
        "status": "online",
        "pinecone_connected": pipeline.pinecone_service.is_connected,
        "cohere_available": pipeline.ranking_service.is_available,
        "guardrails_active": pipeline.guardrails.is_active,
        "embedding_provider": pipeline.config.embedding_provider,
        "gemini_model": pipeline.config.gemini_model_name,
    }
