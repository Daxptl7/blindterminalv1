import os
import logging
from typing import List, Optional
import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("EmbedderService")

class Embedder:
    """
    Dual-mode embedding service supporting:
    1. Gemini Embedding 2 (cloud API)
    2. sentence-transformers / all-MiniLM-L6-v2 (local offline fallback)
    """
    def __init__(self, provider: str = 'local', model_name: str = 'all-MiniLM-L6-v2', api_key: Optional[str] = None):
        self.provider = provider
        self.model_name = model_name
        self.local_model = None
        self.use_gemini = False
        self.gemini_client = None
        self.gemini_embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
        try:
            self.gemini_output_dim = int(os.getenv("GEMINI_EMBEDDING_DIMENSION", "768"))
        except ValueError:
            self.gemini_output_dim = 768

        if provider == 'gemini':
            key = api_key or os.getenv("GEMINI_API_KEY")
            if key and not key.startswith("YOUR_"):
                try:
                    from google import genai
                    self.gemini_client = genai.Client(api_key=key)
                    self.use_gemini = True
                    logger.info(f"Embedder initialized with Gemini API model: {self.gemini_embedding_model}.")
                except Exception as e:
                    logger.error(f"Failed to initialize Gemini Embeddings, falling back to local: {e}")

        if not self.use_gemini:
            self._init_local_model()

        # Dynamic dimension property
        if self.use_gemini:
            self.dimension = self.gemini_output_dim
        elif self.local_model:
            try:
                if hasattr(self.local_model, 'get_embedding_dimension'):
                    self.dimension = self.local_model.get_embedding_dimension()
                else:
                    self.dimension = self.local_model.get_sentence_embedding_dimension()
            except Exception:
                self.dimension = 384
        else:
            self.dimension = 384

    def _init_local_model(self):
        if self.local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Initializing local embedding model: {self.model_name}...")
                self.local_model = SentenceTransformer(self.model_name)
                logger.info("Local embedding model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load local embedding model: {e}")

    def _extract_gemini_embedding(self, result) -> np.ndarray:
        if isinstance(result, dict):
            embedding = result.get("embedding")
            if isinstance(embedding, dict):
                embedding = embedding.get("values")
            return np.array(embedding or [], dtype=np.float32)

        embeddings = getattr(result, "embeddings", None)
        if embeddings:
            first = embeddings[0]
            values = getattr(first, "values", None)
            if values is None and isinstance(first, dict):
                values = first.get("values")
            return np.array(values or [], dtype=np.float32)

        embedding = getattr(result, "embedding", None)
        values = getattr(embedding, "values", embedding)
        return np.array(values or [], dtype=np.float32)

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Converts a single string into a dense vector embedding.
        """
        if not text or not text.strip():
            return np.array([])

        if self.use_gemini:
            try:
                if self.gemini_client:
                    from google.genai import types
                    result = self.gemini_client.models.embed_content(
                        model=self.gemini_embedding_model,
                        contents=text,
                        config=types.EmbedContentConfig(output_dimensionality=self.dimension)
                    )
                    return self._extract_gemini_embedding(result)
            except Exception as e:
                logger.error(f"Gemini embedding error, falling back to local: {e}")

        self._init_local_model()
        if self.local_model:
            try:
                return np.array(self.local_model.encode(text), dtype=np.float32)
            except Exception as e:
                logger.error(f"Local embedding error: {e}")

        return np.array([])

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Converts a list of strings into a matrix of dense vectors.
        """
        if not texts:
            return np.array([])

        embeddings = [self.get_embedding(t) for t in texts if t]
        valid = [e for e in embeddings if e.size > 0]
        if valid:
            return np.vstack(valid)
        return np.array([])
