"""
processor.py — Ingestion Processor
=====================================
Orchestrates the full ingestion pipeline:
1. Crawls the textbooks directory for supported files.
2. Auto-detects metadata tags (standard, subject, chapter) from folder hierarchy.
3. Loads documents (with image processing for Markdown files).
4. Chunks the text into semantic segments.
5. Generates embeddings and upserts them into Pinecone.
"""

import logging
from pathlib import Path
from typing import Optional, List

from .chunking.splitter import TextSplitter
from .loaders.md_loader import MarkdownLoader

logger = logging.getLogger("IngestionProcessor")

MAX_INDEX_FILE_BYTES = 5 * 1024 * 1024
ALLOWED_SUFFIXES = {".txt", ".md", ".text"}


class IngestionProcessor:
    """
    Coordinates document loading, chunking, embedding, and vector storage
    for the textbook ingestion pipeline.
    """

    def __init__(
        self,
        embedding_service,
        pinecone_service,
        splitter: TextSplitter,
        md_loader: MarkdownLoader
    ):
        self.embedding_service = embedding_service
        self.pinecone_service = pinecone_service
        self.splitter = splitter
        self.md_loader = md_loader

    def ingest_directory(self, directory: str) -> dict:
        """
        Scans a directory recursively and ingests all supported textbook files.

        Args:
            directory: Path to the textbooks directory.

        Returns:
            Summary dict with 'indexed' and 'failed' counts.
        """
        dir_path = Path(directory).expanduser().resolve()
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error(f"Textbooks directory not found: {directory}")
            return {"indexed": 0, "failed": 0}

        indexed_count = 0
        failed_count = 0

        logger.info(f"Scanning directory: {dir_path}")
        for path in dir_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES:
                logger.info(f"Found file: {path.name}")
                success = self.ingest_file(str(path))
                if success:
                    indexed_count += 1
                else:
                    failed_count += 1

        logger.info(f"Ingestion complete. Indexed: {indexed_count}, Failed: {failed_count}")
        return {"indexed": indexed_count, "failed": failed_count}

    def ingest_file(self, file_path: str, tags: Optional[dict] = None) -> bool:
        """
        Ingests a single textbook file into the vector database.

        Args:
            file_path: Path to the textbook file.
            tags: Optional metadata tags. If None, auto-detected from folder structure.

        Returns:
            True if ingestion succeeded, False otherwise.
        """
        path = Path(file_path).expanduser().resolve()

        # ── Validation ──────────────────────────────────────
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        if not path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            return False
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            logger.error(f"Unsupported file type for indexing: {path.suffix}")
            return False
        if path.stat().st_size > MAX_INDEX_FILE_BYTES:
            logger.error(f"File too large to index safely: {file_path}")
            return False

        # ── Auto-detect tags from folder hierarchy ──────────
        if tags is None:
            tags = self._auto_detect_tags(path)

        # ── Load document content ───────────────────────────
        try:
            if path.suffix.lower() == '.md':
                # Use Markdown loader (processes images via vision model)
                content = self.md_loader.load(str(path))
            else:
                # Plain text file
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

            if not content or not content.strip():
                logger.warning(f"Empty content in file: {file_path}")
                return False

        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return False

        # ── Chunk → Embed → Upsert ─────────────────────────
        return self._index_content(content, tags)

    def _index_content(self, text: str, tags: Optional[dict] = None) -> bool:
        """
        Chunks text, generates embeddings, and upserts to Pinecone.

        Args:
            text: Document text (with image descriptions already inserted).
            tags: Metadata tags to attach to each chunk.

        Returns:
            True if indexing succeeded.
        """
        if not text or not text.strip():
            logger.warning("No text provided for indexing.")
            return False

        try:
            # 1. Split into chunks
            chunks = self.splitter.split(text)
            if not chunks:
                logger.warning("No chunks generated from text.")
                return False

            logger.info(f"Indexing text into {len(chunks)} chunks.")

            # 2. Generate embeddings
            embeddings = self.embedding_service.get_embeddings(chunks)
            if embeddings.size == 0:
                logger.warning("Embedding generation returned empty matrix.")
                return False

            # 3. Upsert to Pinecone
            chunk_tags = [tags] * len(chunks) if tags else None
            upserted = self.pinecone_service.upsert_batch(chunks, embeddings, tags=chunk_tags)

            if upserted > 0:
                logger.info(f"Successfully indexed {upserted} chunks in Pinecone.")
                return True
            else:
                logger.warning("No chunks were upserted to Pinecone.")
                return False

        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            return False

    def _auto_detect_tags(self, path: Path) -> dict:
        """
        Auto-detects metadata tags (standard, subject, chapter) from folder hierarchy.

        Expected folder structure:
            textbooks/standard_10/science/chemical_reactions.txt
            textbooks/class_12/physics/electrostatics.md
        """
        tags = {}
        parts = path.parts

        for i, part in enumerate(parts):
            part_lower = part.lower()
            if "standard_" in part_lower or "class_" in part_lower or "std_" in part_lower:
                tags["standard"] = part.split("_")[-1]
                if i + 1 < len(parts) - 1:
                    tags["subject"] = parts[i + 1].lower()
                break

        tags["chapter"] = path.stem.lower()
        return tags
