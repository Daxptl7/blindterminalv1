"""
pinecone_service.py — Pinecone Vector Database Service
========================================================
Handles all interactions with Pinecone Cloud Vector DB:
- Upserting document chunks with metadata
- Querying vectors with optional metadata filters
- Index management
"""

import os
import uuid
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("PineconeService")


class PineconeService:
    """
    Production-grade Pinecone Cloud Vector Database wrapper.
    Manages upsert, query, and index operations for the RAG pipeline.
    """

    def __init__(self, api_key: Optional[str] = None, index_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_KEY", "")
        self.index_name = index_name or os.getenv("PINECONE_INDEX", "ncert-books")
        self.index = None

        if self.api_key and not self.api_key.startswith("YOUR_"):
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=self.api_key)
                self.index = pc.Index(self.index_name)
                logger.info(f"Connected to Pinecone Cloud Vector DB: {self.index_name}")
            except Exception as e:
                logger.warning(f"Pinecone Cloud setup failed: {e}")
        else:
            logger.warning("No valid Pinecone API key provided. PineconeService is inactive.")

    @property
    def is_connected(self) -> bool:
        """Check if Pinecone index is connected."""
        return self.index is not None

    def upsert_batch(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        tags: Optional[List[dict]] = None
    ) -> int:
        """
        Upserts a batch of text chunks with their embeddings and metadata to Pinecone.

        Args:
            texts: List of text chunks to store.
            embeddings: Numpy array of embedding vectors (shape: [N, dimension]).
            tags: Optional list of metadata dicts (e.g., standard, subject, chapter).

        Returns:
            Number of vectors successfully upserted.
        """
        if not self.is_connected:
            logger.warning("Pinecone not connected. Skipping upsert.")
            return 0

        if embeddings.size == 0 or not texts:
            return 0

        vecs = embeddings.astype('float32')
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        try:
            vectors_to_upsert = []
            for idx, (vec, text) in enumerate(zip(vecs, texts)):
                doc_id = str(uuid.uuid4())
                tag = tags[idx] if tags and idx < len(tags) else {}
                metadata_payload = {"text": text}
                if tag:
                    metadata_payload.update(tag)
                vectors_to_upsert.append((doc_id, vec.tolist(), metadata_payload))

            self.index.upsert(vectors=vectors_to_upsert)
            logger.info(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone Cloud.")
            return len(vectors_to_upsert)
        except Exception as e:
            logger.error(f"Pinecone Cloud upsert error: {e}")
            return 0

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[dict] = None
    ) -> List[Tuple[str, float]]:
        """
        Queries Pinecone for the most similar vectors.

        Args:
            query_embedding: The query vector.
            top_k: Number of top results to retrieve.
            filters: Optional metadata filters (e.g., {"standard": "10", "subject": "science"}).

        Returns:
            List of (text, score) tuples sorted by relevance.
        """
        if not self.is_connected:
            logger.warning("Pinecone not connected. Returning empty results.")
            return []

        vec = query_embedding.astype('float32').reshape(-1)

        try:
            # Build Pinecone filter from metadata
            filter_dict = {}
            if filters:
                for key, val in filters.items():
                    if val:
                        filter_dict[key] = {"$eq": val}

            res = self.index.query(
                vector=vec.tolist(),
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )

            results = []
            for match in res.matches:
                text = match.metadata.get("text", "")
                score = float(match.score)
                results.append((text, score))

            logger.info(f"Pinecone query returned {len(results)} results.")
            return results

        except Exception as e:
            logger.error(f"Pinecone Cloud query error: {e}")
            return []

    def delete_by_filter(self, filters: Dict[str, str]) -> bool:
        """Deletes vectors matching the given metadata filters."""
        if not self.is_connected:
            return False

        try:
            filter_dict = {key: {"$eq": val} for key, val in filters.items() if val}
            self.index.delete(filter=filter_dict)
            logger.info(f"Deleted vectors matching filter: {filters}")
            return True
        except Exception as e:
            logger.error(f"Pinecone delete error: {e}")
            return False
