import os
import re
import json
import numpy as np
import logging
import hashlib
import tempfile
import threading
import fcntl
from contextlib import contextmanager
from modules.privacy_policy import current_policy
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
    def __init__(self, store_path: str, dimension: int = 384, model_id: str = None, use_cloud: bool = False):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.store_path / "index.faiss"
        self.metadata_file = self.store_path / "metadata.json"
        self.metadata_tags_file = self.store_path / "metadata_tags.json"
        self.legacy_metadata_file = self.store_path / "metadata.pkl"
        self.vectors_file = self.store_path / "vectors.npy"

        self.dimension = dimension
        self.model_id = model_id
        self._lock = threading.RLock()
        self.snapshot_file = self.store_path / "snapshot.json"
        self.backup_file = self.store_path / "snapshot.previous.json"
        self.metadata = []
        self.metadata_tags = []
        self.vectors_list = []
        self.bm25 = None
        
        self.pinecone_key = os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX", "ncert-books")
        self.pinecone_index = None

        if use_cloud and current_policy().allow_cloud and self.pinecone_key and not self.pinecone_key.startswith("YOUR_"):
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

    @contextmanager
    def _writer(self):
        with self._lock, open(self.store_path / ".write.lock", "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    @staticmethod
    def _chunk_id(text, tag):
        return hashlib.sha256(json.dumps([text, tag], ensure_ascii=False,
                                        sort_keys=True).encode()).hexdigest()

    def _rebuild_dense(self):
        self.index = None
        if HAS_FAISS:
            self.index = faiss.IndexFlatL2(self.dimension)
            if self.vectors_list:
                self.index.add(np.asarray(self.vectors_list, dtype=np.float32))
        self._rebuild_bm25()

    def add_batch(self, texts: List[str], embeddings: np.ndarray, tags: Optional[List[dict]] = None):
        if not current_policy().allow_persistence:
            raise PermissionError("Private content cannot be indexed persistently.")
        if not texts:
            return False
        vecs = np.asarray(embeddings, dtype=np.float32)
        if vecs.ndim == 1 and len(texts) == 1:
            vecs = vecs.reshape(1, -1)
        if (vecs.ndim != 2 or vecs.shape != (len(texts), self.dimension)
                or not np.isfinite(vecs).all()
                or any(not isinstance(text, str) or not text.strip() for text in texts)):
            raise ValueError("Embedding rows, text and index dimension must match exactly.")
        if tags is not None and (len(tags) != len(texts) or any(not isinstance(t, dict) for t in tags)):
            raise ValueError("Every chunk must have exactly one metadata object.")
        tags = tags if tags is not None else [{} for _ in texts]
        # Validate JSON before changing any in-memory or persisted state.
        json.dumps(tags, allow_nan=False)
        with self._writer():
            if self.snapshot_file.exists() or self.backup_file.exists():
                self.load_index()
            previous = (self.metadata[:], self.metadata_tags[:], self.vectors_list[:])
            known = {self._chunk_id(text, tag) for text, tag in zip(self.metadata, self.metadata_tags)}
            try:
                for text, vec, tag in zip(texts, vecs, tags):
                    identity = self._chunk_id(text, tag)
                    if identity not in known:
                        self.metadata.append(text)
                        self.metadata_tags.append(dict(tag))
                        self.vectors_list.append(vec.copy())
                        known.add(identity)
                self._rebuild_dense()
                self.save_index()
            except BaseException:
                self.metadata, self.metadata_tags, self.vectors_list = previous
                self._rebuild_dense()
                raise
        if self.pinecone_index is not None and current_policy().allow_cloud:
            try:
                records = [(self._chunk_id(text, tag), vec.tolist(), {"text": text, **tag})
                           for text, vec, tag in zip(texts, vecs, tags)]
                self.pinecone_index.upsert(vectors=records, namespace=self.model_id or "legacy")
            except Exception:
                logger.warning("Cloud index unavailable; local committed copy retained.")
        return True

    def retrieve(self, query_embedding: np.ndarray, query_text: Optional[str] = None, k: int = 3, filters: Optional[dict] = None) -> List[Tuple[int, float, str]]:
        with self._lock:
            return self._retrieve(query_embedding, query_text, k, filters)

    def _retrieve(self, query_embedding: np.ndarray, query_text: Optional[str] = None, k: int = 3, filters: Optional[dict] = None) -> List[Tuple[int, float, str]]:
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
            vec = np.array([], dtype=np.float32)
        vector_results = []

        # 1. Pinecone Retrieval
        if self.pinecone_index is not None and current_policy().allow_cloud and vec.size:
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
                    namespace=self.model_id or "legacy",
                    filter=filter_dict if filter_dict else None
                )
                for idx, match in enumerate(res.matches):
                    text = match.metadata.get("text", "")
                    score = match.score
                    vector_results.append((idx, float(1.0 - score), text))
            except Exception as e:
                logger.error(f"Pinecone Cloud query error, falling back: {e}")

        # 2. Local Dense Retrieval (FAISS or Numpy) if Pinecone returned empty
        if not vector_results and self.metadata and vec.size:
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
        if not vector_results and query_text and self.metadata:
            tokens = set(self._tokenize(query_text))
            lexical = []
            for i, text in enumerate(self.metadata):
                tag = self.metadata_tags[i]
                if filters and any(value and tag.get(key) != value for key, value in filters.items()):
                    continue
                overlap = len(tokens.intersection(self._tokenize(text)))
                if overlap:
                    lexical.append((i, 1.0 / (1 + overlap), text))
            return sorted(lexical, key=lambda item: item[1])[:k]
        return vector_results[:k]

    def _snapshot_payload(self):
        return {"schema_version": 1, "model_id": self.model_id,
                "dimension": self.dimension, "texts": self.metadata,
                "tags": self.metadata_tags,
                "vectors": [np.asarray(v).tolist() for v in self.vectors_list]}

    def _decode_snapshot(self, path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported index snapshot schema.")
        dimension = payload["dimension"]
        if dimension != self.dimension or payload.get("model_id") != self.model_id:
            raise ValueError("Index model identity or dimension is incompatible; use a separate index.")
        texts, tags = payload["texts"], payload["tags"]
        if (not isinstance(texts, list) or any(not isinstance(t, str) for t in texts)
                or not isinstance(tags, list) or len(tags) != len(texts)
                or any(not isinstance(t, dict) for t in tags)):
            raise ValueError("Index metadata is inconsistent.")
        vectors = np.asarray(payload["vectors"], dtype=np.float32)
        if not texts and vectors.size == 0:
            vectors = np.empty((0, dimension), dtype=np.float32)
        if vectors.shape != (len(texts), dimension) or not np.isfinite(vectors).all():
            raise ValueError("Index vectors are inconsistent.")
        return texts, tags, list(vectors)

    def _atomic_write(self, path, data):
        descriptor, temporary = tempfile.mkstemp(prefix=".index-", dir=self.store_path)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def save_index(self):
        if not current_policy().allow_persistence:
            raise PermissionError("Private sessions cannot persist an index.")
        with self._lock:
            payload = json.dumps(self._snapshot_payload(), ensure_ascii=False,
                                 allow_nan=False).encode("utf-8")
            if self.snapshot_file.exists():
                try:
                    self._decode_snapshot(self.snapshot_file)
                except (ValueError, KeyError, TypeError):
                    pass  # Do not replace a valid backup with a corrupt primary.
                else:
                    self._atomic_write(self.backup_file, self.snapshot_file.read_bytes())
            self._atomic_write(self.snapshot_file, payload)
            # Publish is complete only once its directory entry is durable.
            descriptor = os.open(self.store_path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return True

    def load_index(self):
        with self._lock:
            found = False
            for path in (self.snapshot_file, self.backup_file):
                if not path.exists():
                    continue
                found = True
                try:
                    state = self._decode_snapshot(path)
                except (ValueError, KeyError, TypeError, OSError):
                    logger.warning("Index snapshot could not be loaded.")
                    continue
                self.metadata, self.metadata_tags, self.vectors_list = state
                self._rebuild_dense()
                return
            if found:
                raise ValueError("No valid compatible index snapshot; original files retained.")
            # Preserve legacy files. Import validated JSON/NumPy only; never pickle.
            if self.metadata_file.exists() and self.vectors_file.exists():
                texts = json.loads(self.metadata_file.read_text(encoding="utf-8"))
                tags = (json.loads(self.metadata_tags_file.read_text(encoding="utf-8"))
                        if self.metadata_tags_file.exists() else [{} for _ in texts])
                vectors = np.load(self.vectors_file, allow_pickle=False)
                if (not isinstance(texts, list) or any(not isinstance(t, str) for t in texts)
                        or len(tags) != len(texts) or any(not isinstance(t, dict) for t in tags)
                        or vectors.shape != (len(texts), self.dimension) or not np.isfinite(vectors).all()
                        or self.model_id is not None):
                    raise ValueError("Legacy index needs explicit model-specific reingestion; files retained.")
                self.metadata, self.metadata_tags, self.vectors_list = texts, tags, list(vectors)
            elif self.legacy_metadata_file.exists():
                logger.warning("Ignoring unsafe legacy pickle; reingest original documents.")
            self._rebuild_dense()
