import os
import re
import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("VectorStoreService")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("FAISS not installed. VectorStore will use numpy cosine similarity as local fallback.")

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logger.warning("rank-bm25 not installed. Hybrid search will rely solely on dense vectors.")


class VectorStore:
    """
    Production-grade Vector Storage & Hybrid Search Engine.
    Supports:
    1. Pinecone Cloud Vector DB
    2. FAISS Index Flat (local dense vector search)
    3. BM25 Keyword Search (local sparse search)
    4. Reciprocal Rank Fusion (RRF) combining Dense + Sparse rankings.
    """
    def __init__(self, store_path: str, dimension: int = 384):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.store_path / "index.faiss"
        self.metadata_file = self.store_path / "metadata.json"
        self.metadata_tags_file = self.store_path / "metadata_tags.json"
        self.legacy_metadata_file = self.store_path / "metadata.pkl"
        self.vectors_file = self.store_path / "vectors.npy"

        self.dimension = dimension
        self.metadata = []
        self.metadata_tags = []
        self.vectors_list = []
        self.bm25 = None
        
        self.pinecone_key = os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX", "ncert-books")
        self.pinecone_index = None

        if self.pinecone_key and not self.pinecone_key.startswith("YOUR_"):
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=self.pinecone_key)
                self.pinecone_index = pc.Index(self.pinecone_index_name)
                logger.info(f"Connected to Pinecone Cloud Vector DB: {self.pinecone_index_name}")
            except Exception as e:
                logger.warning(f"Pinecone Cloud setup fallback: {e}")

        self.index = None
        self.load_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple alphanumeric tokenizer for BM25 keyword matching."""
        return re.findall(r'\w+', text.lower())

    def _rebuild_bm25(self):
        """Rebuilds BM25 index over self.metadata."""
        if HAS_BM25 and self.metadata:
            tokenized_corpus = [self._tokenize(doc) for doc in self.metadata]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.debug(f"BM25 index rebuilt over {len(self.metadata)} documents.")

    def add_text(self, text: str, embedding: np.ndarray, tags: Optional[dict] = None):
        if embedding.size == 0:
            return
        self.add_batch([text], embedding.reshape(1, -1), tags=[tags] if tags else None)

    def add_batch(self, texts: List[str], embeddings: np.ndarray, tags: Optional[List[dict]] = None):
        if embeddings.size == 0 or not texts:
            return

        vecs = embeddings.astype('float32')
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        # Dynamic dimension adjustment for FAISS if needed
        actual_dim = vecs.shape[1]
        if actual_dim != self.dimension:
            logger.warning(
                "Embedding dimension changed from %s to %s. "
                "Dropping incompatible local vectors; re-index documents for best RAG results.",
                self.dimension,
                actual_dim,
            )
            self.dimension = actual_dim
            compatible_vectors = []
            compatible_metadata = []
            compatible_metadata_tags = []
            for i, (existing_vec, text) in enumerate(zip(self.vectors_list, self.metadata)):
                if np.asarray(existing_vec).reshape(-1).shape[0] == self.dimension:
                    compatible_vectors.append(existing_vec)
                    compatible_metadata.append(text)
                    if hasattr(self, 'metadata_tags') and i < len(self.metadata_tags):
                        compatible_metadata_tags.append(self.metadata_tags[i])
                    else:
                        compatible_metadata_tags.append({})
            self.vectors_list = compatible_vectors
            self.metadata = compatible_metadata
            self.metadata_tags = compatible_metadata_tags
            if HAS_FAISS:
                self.index = faiss.IndexFlatL2(self.dimension)
                if self.vectors_list:
                    existing_mat = np.array(self.vectors_list, dtype=np.float32)
                    if existing_mat.ndim > 1 and existing_mat.shape[1] == self.dimension:
                        self.index.add(existing_mat)

        if self.pinecone_index is not None:
            try:
                import uuid
                vectors_to_upsert = []
                for idx, (vec, text) in enumerate(zip(vecs, texts)):
                    doc_id = str(uuid.uuid4())
                    tag = tags[idx] if tags and idx < len(tags) else {}
                    metadata_payload = {"text": text}
                    if tag:
                        metadata_payload.update(tag)
                    vectors_to_upsert.append((doc_id, vec.tolist(), metadata_payload))
                self.pinecone_index.upsert(vectors=vectors_to_upsert)
                logger.info(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone Cloud.")
            except Exception as e:
                logger.error(f"Pinecone Cloud upsert error: {e}")

        if HAS_FAISS:
            if self.index is None:
                self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(vecs)
        
        for idx, (vec, text) in enumerate(zip(vecs, texts)):
            self.vectors_list.append(vec)
            self.metadata.append(text)
            tag = tags[idx] if tags and idx < len(tags) else {}
            self.metadata_tags.append(tag)

        self._rebuild_bm25()
        self.save_index()

    def retrieve(self, query_embedding: np.ndarray, query_text: Optional[str] = None, k: int = 3, filters: Optional[dict] = None) -> List[Tuple[int, float, str]]:
        if len(self.metadata) == 0 and self.pinecone_index is None:
            return []

        vec = query_embedding.astype('float32').reshape(-1)
        if vec.shape[0] != self.dimension:
            logger.warning(
                "Query embedding dimension %s does not match vector store dimension %s. "
                "Re-index documents with the current embedding model.",
                vec.shape[0],
                self.dimension,
            )
            return []
        vector_results = []

        # 1. Pinecone Retrieval
        if self.pinecone_index is not None:
            try:
                filter_dict = {}
                if filters:
                    for key, val in filters.items():
                        if val:
                            filter_dict[key] = {"$eq": val}
                res = self.pinecone_index.query(
                    vector=vec.tolist(),
                    top_k=max(k*3, 10),
                    include_metadata=True,
                    filter=filter_dict if filter_dict else None
                )
                for idx, match in enumerate(res.matches):
                    text = match.metadata.get("text", "")
                    score = match.score
                    vector_results.append((idx, float(1.0 - score), text))
            except Exception as e:
                logger.error(f"Pinecone Cloud query error, falling back: {e}")

        # 2. Local Dense Retrieval (FAISS or Numpy) if Pinecone returned empty
        if not vector_results and self.metadata:
            vec_mat = vec.reshape(1, -1)
            if HAS_FAISS and self.index is not None and self.index.ntotal > 0:
                search_k = min(max(k * 5, 50), self.index.ntotal)
                distances, indices = self.index.search(vec_mat, search_k)
                for dist, idx in zip(distances[0], indices[0]):
                    if idx != -1 and idx < len(self.metadata):
                        if filters:
                            tags = self.metadata_tags[idx] if idx < len(self.metadata_tags) else {}
                            match = True
                            for key, val in filters.items():
                                if val and tags.get(key) != val:
                                    match = False
                                    break
                            if not match:
                                continue
                        vector_results.append((int(idx), float(dist), self.metadata[idx]))
            elif self.vectors_list:
                matrix = np.array(self.vectors_list)
                if matrix.shape[1] == vec.shape[0]:
                    norm_q = vec / (np.linalg.norm(vec) + 1e-10)
                    norm_m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
                    similarities = np.dot(norm_m, norm_q.T).flatten()
                    top_idx = np.argsort(similarities)[::-1]
                    for idx in top_idx:
                        if idx < len(self.metadata):
                            if filters:
                                tags = self.metadata_tags[idx] if idx < len(self.metadata_tags) else {}
                                match = True
                                for key, val in filters.items():
                                    if val and tags.get(key) != val:
                                        match = False
                                        break
                                if not match:
                                    continue
                            vector_results.append((int(idx), float(1.0 - similarities[idx]), self.metadata[idx]))

        # 3. Hybrid Search Fusion (Reciprocal Rank Fusion) if query_text & BM25 available
        if query_text and HAS_BM25 and self.bm25 and self.metadata:
            tokenized_query = self._tokenize(query_text)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_top_indices = np.argsort(bm25_scores)[::-1]

            filtered_bm25_indices = []
            for idx in bm25_top_indices:
                if idx < len(self.metadata) and bm25_scores[idx] > 0:
                    if filters:
                        tags = self.metadata_tags[idx] if idx < len(self.metadata_tags) else {}
                        match = True
                        for key, val in filters.items():
                            if val and tags.get(key) != val:
                                match = False
                                break
                        if not match:
                            continue
                    filtered_bm25_indices.append(idx)
                    if len(filtered_bm25_indices) >= max(k * 3, 10):
                        break

            # Map doc text -> RRF score
            rrf_scores = {}
            c = 60 # standard RRF constant

            # Vector ranks
            for rank, item in enumerate(vector_results):
                doc_text = item[2]
                rrf_scores[doc_text] = rrf_scores.get(doc_text, 0.0) + (1.0 / (c + rank + 1))

            # BM25 ranks
            for rank, idx in enumerate(filtered_bm25_indices):
                doc_text = self.metadata[idx]
                rrf_scores[doc_text] = rrf_scores.get(doc_text, 0.0) + (1.0 / (c + rank + 1))

            if rrf_scores:
                sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
                return [(i, float(1.0 - score), text) for i, (text, score) in enumerate(sorted_docs)]

        # Fallback to standard vector results
        return vector_results[:k]

    def save_index(self):
        try:
            if HAS_FAISS and self.index is not None:
                faiss.write_index(self.index, str(self.index_file))
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False)
            with open(self.metadata_tags_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata_tags, f, ensure_ascii=False)
            if self.vectors_list:
                np.save(self.vectors_file, np.array(self.vectors_list))
            logger.info(f"Vector store saved to {self.store_path}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")

    def load_index(self):
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    loaded_metadata = json.load(f)
                if not isinstance(loaded_metadata, list) or not all(isinstance(item, str) for item in loaded_metadata):
                    raise ValueError("metadata.json must contain a list of strings")
                self.metadata = loaded_metadata

                if self.metadata_tags_file.exists():
                    try:
                        with open(self.metadata_tags_file, 'r', encoding='utf-8') as f:
                            self.metadata_tags = json.load(f)
                    except Exception as e:
                        logger.error(f"Failed to load metadata tags: {e}")
                        self.metadata_tags = [{} for _ in self.metadata]
                else:
                    self.metadata_tags = [{} for _ in self.metadata]

                if len(self.metadata_tags) != len(self.metadata):
                    if len(self.metadata_tags) < len(self.metadata):
                        self.metadata_tags.extend([{} for _ in range(len(self.metadata) - len(self.metadata_tags))])
                    else:
                        self.metadata_tags = self.metadata_tags[:len(self.metadata)]

                if self.vectors_file.exists():
                    vecs = np.load(self.vectors_file)
                    self.vectors_list = list(vecs)
                    if vecs.ndim > 1:
                        self.dimension = vecs.shape[1]

                if HAS_FAISS and self.index_file.exists():
                    self.index = faiss.read_index(str(self.index_file))
                    self.dimension = self.index.d
                elif HAS_FAISS and self.vectors_list:
                    self.index = faiss.IndexFlatL2(self.dimension)
                    self.index.add(np.array(self.vectors_list, dtype=np.float32))

                self._rebuild_bm25()
                logger.info(f"Vector store loaded successfully from {self.store_path}")
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
                self.metadata = []
                self.metadata_tags = []
                self.vectors_list = []
        elif self.legacy_metadata_file.exists():
            logger.warning(
                "Ignoring legacy metadata.pkl because pickle can execute code. "
                "Set BLINDASSIST_TRUST_LEGACY_PICKLE=1 once to migrate a trusted local store."
            )
            if os.getenv("BLINDASSIST_TRUST_LEGACY_PICKLE") == "1":
                try:
                    import pickle
                    with open(self.legacy_metadata_file, 'rb') as f:
                        loaded_metadata = pickle.load(f)
                    if not isinstance(loaded_metadata, list) or not all(isinstance(item, str) for item in loaded_metadata):
                        raise ValueError("legacy metadata must contain a list of strings")
                    self.metadata = loaded_metadata
                    if self.vectors_file.exists():
                        vecs = np.load(self.vectors_file)
                        self.vectors_list = list(vecs)
                        if vecs.ndim > 1:
                            self.dimension = vecs.shape[1]
                    if HAS_FAISS and self.vectors_list:
                        self.index = faiss.IndexFlatL2(self.dimension)
                        self.index.add(np.array(self.vectors_list, dtype=np.float32))
                    self._rebuild_bm25()
                    self.save_index()
                    logger.info("Migrated trusted legacy metadata.pkl to metadata.json.")
                except Exception as e:
                    logger.error(f"Failed to migrate legacy vector store: {e}")
                    self.metadata = []
                    self.vectors_list = []
