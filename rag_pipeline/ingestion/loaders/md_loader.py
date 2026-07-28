"""
md_loader.py — Markdown Document Loader with Vision Processing
================================================================
Loads Markdown (.md) files and uses Gemini's multimodal capability
to extract and describe images referenced in the document.
Replaces image markdown syntax with detailed text descriptions
so they can be embedded and searched in the vector database.
"""

import re
import logging
from pathlib import Path
from typing import Optional
from PIL import Image

logger = logging.getLogger("MarkdownLoader")

# Regex pattern to match Markdown image syntax: ![Alt Text](path/to/image.png)
IMAGE_PATTERN = re.compile(r'!\[(.*?)\]\((.*?)\)')


class MarkdownLoader:
    """
    Loads Markdown files, detects embedded image references,
    and uses a vision model to generate text descriptions of those images.
    """

    def __init__(self, gemini_agent=None):
        """
        Args:
            gemini_agent: An instance of GeminiAgent with describe_image() capability.
                          If None, images will be replaced with their alt-text only.
        """
        self.gemini_agent = gemini_agent

    def load(self, file_path: str) -> str:
        """
        Reads a Markdown file and processes any embedded images.

        Args:
            file_path: Absolute or relative path to the .md file.

        Returns:
            The document text with image references replaced by text descriptions.
        """
        path = Path(file_path).expanduser().resolve()

        if not path.exists() or not path.is_file():
            logger.error(f"File not found: {file_path}")
            return ""

        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return ""

        # Process image references if this is a Markdown file
        if path.suffix.lower() == '.md':
            content = self._process_images(content, parent_dir=path.parent)

        return content

    def _process_images(self, content: str, parent_dir: Path) -> str:
        """
        Finds all Markdown image references, loads the images from disk,
        and replaces each reference with a detailed text description.

        Args:
            content: Raw Markdown text.
            parent_dir: Directory where the Markdown file is located (for resolving relative paths).

        Returns:
            Updated content with image references replaced by descriptions.
        """
        matches = list(IMAGE_PATTERN.finditer(content))

        if not matches:
            return content

        logger.info(f"Found {len(matches)} image reference(s) to process.")

        for match in matches:
            alt_text = match.group(1)
            img_rel_path = match.group(2)

            # Resolve the image path relative to the Markdown file's directory
            img_abs_path = parent_dir / img_rel_path

            if img_abs_path.exists() and img_abs_path.is_file():
                description = self._describe_image(img_abs_path, context_hint=alt_text)
                replacement = f"\n[Diagram Reference: {alt_text}. Description: {description}]\n"
                content = content.replace(match.group(0), replacement)
                logger.info(f"Processed image: {img_rel_path}")
            else:
                # Keep alt-text as fallback context
                replacement = f"\n[Image: {alt_text} (file not found)]\n"
                content = content.replace(match.group(0), replacement)
                logger.warning(f"Referenced image not found: {img_abs_path}")

        return content

    def _describe_image(self, image_path: Path, context_hint: str = "") -> str:
        """
        Generates a text description of an image using the vision model.

        Args:
            image_path: Absolute path to the image file.
            context_hint: Alt text or caption to help the vision model.

        Returns:
            Detailed text description of the image content.
        """
        if not self.gemini_agent:
            return context_hint or "No description available."

        try:
            with Image.open(image_path) as img:
                description = self.gemini_agent.describe_image(img, context_hint=context_hint)
                return description if description else context_hint
        except Exception as e:
            logger.warning(f"Failed to describe image {image_path}: {e}")
            return context_hint or "Image description failed."
