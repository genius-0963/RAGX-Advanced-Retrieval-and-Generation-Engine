"""
RAGX Markdown Loader — Parse Markdown files into clean-text documents.

Strips inline formatting (bold, italic, links, images, code spans) while
preserving the heading hierarchy so that section context is available in
document metadata.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from ragx.config.logging_config import get_logger
from ragx.ingestion.loaders.base import BaseLoader, Document

logger = get_logger(__name__)

# ── Regex helpers for stripping Markdown syntax ─────────────────────────────
_RE_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_RE_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_RE_BOLD_ITALIC = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1")
_RE_STRIKETHROUGH = re.compile(r"~~(.+?)~~")
_RE_INLINE_CODE = re.compile(r"`([^`]+)`")
_RE_HEADING = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
_RE_BLOCKQUOTE = re.compile(r"^>\s?", re.MULTILINE)
_RE_UNORDERED_LIST = re.compile(r"^[\s]*[-*+]\s+", re.MULTILINE)
_RE_ORDERED_LIST = re.compile(r"^[\s]*\d+\.\s+", re.MULTILINE)
_RE_HORIZONTAL_RULE = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)
_RE_FENCED_CODE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_RE_HTML_TAG = re.compile(r"<[^>]+>")


def _strip_markdown(text: str) -> str:
    """Remove common Markdown formatting from *text*, returning plain text."""
    # Remove fenced code blocks (keep content)
    text = _RE_FENCED_CODE.sub(lambda m: m.group(0).strip("`").strip(), text)
    # Remove images (keep alt text)
    text = _RE_IMAGE.sub(r"\1", text)
    # Remove links (keep link text)
    text = _RE_LINK.sub(r"\1", text)
    # Remove bold / italic / bold-italic
    text = _RE_BOLD_ITALIC.sub(r"\2", text)
    # Remove strikethrough
    text = _RE_STRIKETHROUGH.sub(r"\1", text)
    # Remove inline code ticks
    text = _RE_INLINE_CODE.sub(r"\1", text)
    # Remove blockquote markers
    text = _RE_BLOCKQUOTE.sub("", text)
    # Remove list markers
    text = _RE_UNORDERED_LIST.sub("", text)
    text = _RE_ORDERED_LIST.sub("", text)
    # Remove horizontal rules
    text = _RE_HORIZONTAL_RULE.sub("", text)
    # Remove residual HTML
    text = _RE_HTML_TAG.sub("", text)
    return text.strip()


class MarkdownLoader(BaseLoader):
    """Loader for Markdown files.

    The file is split on headings so each section becomes its own
    :class:`Document` with the heading hierarchy stored in metadata.
    Inline formatting is stripped for downstream NLP cleanliness.
    """

    supported_extensions: ClassVar[list[str]] = [".md", ".markdown"]

    def load(self, source: str) -> list[Document]:
        """Load a Markdown file, splitting by heading sections.

        Args:
            source: File-system path to a ``.md`` file.

        Returns:
            A list of :class:`Document` instances, one per section.

        Raises:
            FileNotFoundError: If *source* does not exist.
        """
        path = Path(source).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {path}")

        logger.info("Loading Markdown", path=str(path))

        raw_text = path.read_text(encoding="utf-8", errors="replace")

        # ── Split into sections delimited by headings ───────────────────
        sections: list[tuple[str, int, str]] = []  # (heading, level, body)
        current_heading = ""
        current_level = 0
        current_lines: list[str] = []

        for line in raw_text.splitlines():
            heading_match = _RE_HEADING.match(line)
            if heading_match:
                # Flush previous section
                if current_lines or current_heading:
                    body = "\n".join(current_lines).strip()
                    sections.append((current_heading, current_level, body))

                current_heading = heading_match.group(2).strip()
                current_level = len(heading_match.group(1))
                current_lines = []
            else:
                current_lines.append(line)

        # Flush last section
        body = "\n".join(current_lines).strip()
        if body or current_heading:
            sections.append((current_heading, current_level, body))

        # ── Build documents ─────────────────────────────────────────────
        documents: list[Document] = []
        for idx, (heading, level, section_body) in enumerate(sections):
            clean_text = _strip_markdown(section_body)
            if not clean_text and not heading:
                continue

            # Use heading as prefix for context
            full_content = f"{heading}\n\n{clean_text}".strip() if heading else clean_text

            doc = Document(
                content=full_content,
                metadata={
                    "section": heading,
                    "heading_level": level,
                    "section_index": idx,
                    "file_name": path.name,
                },
                source_path=str(path),
            )
            documents.append(doc)

        logger.info(
            "Markdown loaded",
            path=str(path),
            sections=len(documents),
        )
        return documents
