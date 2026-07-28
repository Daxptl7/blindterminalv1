import os
import logging
from typing import List, Optional
import numpy as np
from dotenv import load_dotenv
from .embedder import Embedder
from .vector_store import VectorStore

load_dotenv()
logger = logging.getLogger("RetrieverService")

class Retriever:
    """
    Handles semantic search by coordinating Embedder and VectorStore.
    Integrates Cohere Rerank v3 (Item 4 in purchase sheet) to improve retrieval accuracy.
    """
    def __init__(self, embedder: Embedder, vector_store: VectorStore, cohere_api_key: Optional[str] = None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.cohere_key = cohere_api_key or os.getenv("COHERE_API_KEY")
        self.cohere_client = None

        if self.cohere_key and not self.cohere_key.startswith("YOUR_"):
            try:
                import cohere
                self.cohere_client = cohere.ClientV2(api_key=self.cohere_key)
                logger.info("Cohere Rerank v3 service initialized successfully.")
            except Exception as e:
                logger.warning(f"Failed to initialize Cohere Rerank, falling back to vector distance sorting: {e}")

    def get_relevant_context(self, query: str, k: int = 3, filters: Optional[dict] = None) -> str:
        """
        Retrieves top-k relevant text chunks for a query, reranking with Cohere if configured.
        """
        if not query or not query.strip():
            return ""

        try:
            # 1. Embed query
            query_vec = self.embedder.get_embedding(query)
            if query_vec.size == 0:
                return ""

            # 2. Retrieve candidate chunks (fetch top 10 for reranking)
            fetch_k = max(k * 3, 10)
            results = self.vector_store.retrieve(query_vec, query_text=query, k=fetch_k, filters=filters)
            if not results:
                return ""

            candidate_chunks = [text for _, _, text in results]

            # 3. Apply Cohere Rerank v3 if available
            if self.cohere_client and len(candidate_chunks) > 1:
                try:
                    rerank_res = self.cohere_client.rerank(
                        model="rerank-v3.5",
                        query=query,
                        documents=candidate_chunks,
                        top_n=k
                    )
                    top_chunks = [candidate_chunks[result.index] for result in rerank_res.results]
                    logger.info(f"Successfully reranked {len(candidate_chunks)} candidates down to top {len(top_chunks)} using Cohere.")
                    return "\n---\n".join(top_chunks)
                except Exception as e:
                    logger.warning(f"Cohere rerank API error, using vector scores: {e}")

            # Fallback: return top-k by vector similarity
            top_chunks = candidate_chunks[:k]
            return "\n---\n".join(top_chunks)

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""

