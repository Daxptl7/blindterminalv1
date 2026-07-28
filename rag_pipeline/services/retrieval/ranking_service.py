"""
ranking_service.py — Cohere Reranking Service
================================================
Integrates Cohere Rerank v3 to improve retrieval accuracy
by reranking candidate chunks based on semantic relevance to the query.
"""

import os
import logging
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("RankingService")


class RankingService:
    """
    Reranks candidate text chunks using Cohere Rerank v3
    to surface the most contextually relevant results.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.cohere_key = api_key or os.getenv("COHERE_API_KEY", "")
        self.client = None

        if self.cohere_key and not self.cohere_key.startswith("YOUR_"):
            try:
                import cohere
                self.client = cohere.ClientV2(api_key=self.cohere_key)
                logger.info("Cohere Rerank v3 service initialized successfully.")
            except Exception as e:
                logger.warning(f"Failed to initialize Cohere Rerank: {e}")
        else:
            logger.info("Cohere API key not configured. Reranking will be skipped.")

    @property
    def is_available(self) -> bool:
        """Check if Cohere reranking is available."""
        return self.client is not None

    def rerank(
        self,
        query: str,
        candidate_chunks: List[str],
        top_n: int = 3,
        relevance_threshold: float = 0.0
    ) -> List[str]:
        """
        Reranks candidate chunks using Cohere Rerank v3 and returns the top-n most relevant.

        Args:
            query: The user's search query.
            candidate_chunks: List of text chunks to rerank.
            top_n: Number of top results to return.
            relevance_threshold: Minimum relevance score to include (0.0 = no threshold).

        Returns:
            List of reranked text chunks, ordered by relevance.
        """
        if not self.is_available or len(candidate_chunks) <= 1:
            return candidate_chunks[:top_n]

        try:
            rerank_res = self.client.rerank(
                model="rerank-v3.5",
                query=query,
                documents=candidate_chunks,
                top_n=top_n
            )

            top_chunks = []
            for result in rerank_res.results:
                if relevance_threshold > 0 and result.relevance_score < relevance_threshold:
                    logger.debug(
                        f"Chunk discarded by relevance threshold "
                        f"(score: {result.relevance_score:.3f} < {relevance_threshold})"
                    )
                    continue
                top_chunks.append(candidate_chunks[result.index])

            logger.info(
                f"Reranked {len(candidate_chunks)} candidates down to "
                f"{len(top_chunks)} using Cohere (threshold={relevance_threshold})."
            )
            return top_chunks

        except Exception as e:
            logger.warning(f"Cohere rerank API error, returning original order: {e}")
            return candidate_chunks[:top_n]
