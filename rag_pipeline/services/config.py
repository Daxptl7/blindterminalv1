"""
config.py — RAG Pipeline Configuration Loader
================================================
Loads settings from config/settings.json and environment variables.
Provides a single source of truth for all API keys and pipeline parameters.
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("RAGConfig")

# Resolve the base directory of the Blindterminal project
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config" / "settings.json"
DATA_DIR = BASE_DIR / "data"
TEXTBOOKS_DIR = DATA_DIR / "textbooks"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"


class Config:
    """
    Centralized configuration for the RAG pipeline.
    Loads from settings.json with environment variable overrides.
    """

    def __init__(self):
        self._settings = {}
        try:
            with open(CONFIG_PATH, 'r') as f:
                self._settings = json.load(f)
            logger.info(f"Configuration loaded from {CONFIG_PATH}")
        except Exception as e:
            logger.warning(f"Settings load failed, using env vars only: {e}")

    # ── API Keys ────────────────────────────────────────────
    @property
    def gemini_api_key(self) -> str:
        return self._settings.get("gemini_api_key") or os.getenv("GEMINI_API_KEY", "")

    @property
    def pinecone_api_key(self) -> str:
        return self._settings.get("pinecone_api_key") or os.getenv("PINECONE_API_KEY", "")

    @property
    def cohere_api_key(self) -> str:
        return self._settings.get("cohere_api_key") or os.getenv("COHERE_API_KEY", "")

    # ── Model Settings ──────────────────────────────────────
    @property
    def gemini_model_name(self) -> str:
        return self._settings.get("gemini_model_name", "gemini-3.5-flash")

    @property
    def embedding_provider(self) -> str:
        return self._settings.get("embedding_provider", "local")

    @property
    def pinecone_index_name(self) -> str:
        return os.getenv("PINECONE_INDEX", "ncert-books")

    @property
    def use_cohere_rerank(self) -> bool:
        return self._settings.get("use_cohere_rerank", True)

    # ── Ingestion Settings ──────────────────────────────────
    @property
    def chunk_size(self) -> int:
        return self._settings.get("chunk_size", 500)

    @property
    def chunk_overlap(self) -> int:
        return self._settings.get("chunk_overlap", 50)

    @property
    def max_index_file_bytes(self) -> int:
        return self._settings.get("max_index_file_bytes", 5 * 1024 * 1024)

    @property
    def allowed_suffixes(self) -> set:
        return {".txt", ".md", ".text"}

    # ── Paths ───────────────────────────────────────────────
    @property
    def base_dir(self) -> Path:
        return BASE_DIR

    @property
    def textbooks_dir(self) -> Path:
        return TEXTBOOKS_DIR

    @property
    def vector_store_dir(self) -> Path:
        return VECTOR_STORE_DIR

    def is_valid_key(self, key: str) -> bool:
        """Check if an API key is set and not a placeholder."""
        return bool(key) and not key.startswith("YOUR_")
