"""
splitter.py — Semantic Text Splitter
=======================================
Splits long text hierarchically across natural semantic boundaries
(paragraphs, sentences, spaces) to prevent severing words or thoughts mid-sentence.
"""

import logging
from typing import List

logger = logging.getLogger("TextSplitter")


class TextSplitter:
    """
    Hierarchical text chunker that splits documents along natural boundaries.
    Uses a cascade of separators: paragraphs → newlines → sentences → words → characters.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def split(self, text: str) -> List[str]:
        """
        Splits text into chunks respecting semantic boundaries.

        Args:
            text: Raw document text to split.

        Returns:
            List of text chunks, each no longer than chunk_size characters.
        """
        if not text or not text.strip():
            return []

        chunks = self._split_text(text, self.separators)
        logger.info(f"Split text into {len(chunks)} chunks (chunk_size={self.chunk_size}, overlap={self.overlap}).")
        return chunks

    def _split_text(self, text_to_split: str, seps: List[str]) -> List[str]:
        """Recursively splits text using progressively finer separators."""
        if len(text_to_split) <= self.chunk_size or not seps:
            return [text_to_split.strip()] if text_to_split.strip() else []

        sep = seps[0]
        next_seps = seps[1:]

        # Last resort: character-level splitting with overlap
        if sep == "":
            splits = [
                text_to_split[i:i + self.chunk_size]
                for i in range(0, len(text_to_split), self.chunk_size - self.overlap)
            ]
            return [s.strip() for s in splits if s.strip()]

        parts = text_to_split.split(sep)
        docs = []
        current_chunk = []
        current_length = 0

        for part in parts:
            # Re-attach the separator for natural reading
            if sep in ["\n\n", "\n", ". ", "? ", "! "]:
                part_text = part + sep
            elif sep == " ":
                part_text = part + " "
            else:
                part_text = part

            part_len = len(part_text)

            if part_len > self.chunk_size:
                # This single part is too large — flush current chunk and recurse
                if current_chunk:
                    joined = "".join(current_chunk).strip()
                    if joined:
                        docs.append(joined)
                    current_chunk = []
                    current_length = 0
                docs.extend(self._split_text(part, next_seps))

            elif current_length + part_len > self.chunk_size:
                # Adding this part would exceed the limit — flush and start new chunk
                joined = "".join(current_chunk).strip()
                if joined:
                    docs.append(joined)
                current_chunk = [part_text]
                current_length = part_len

            else:
                # Accumulate into current chunk
                current_chunk.append(part_text)
                current_length += part_len

        # Flush any remaining content
        if current_chunk:
            joined = "".join(current_chunk).strip()
            if joined:
                docs.append(joined)

        return docs
