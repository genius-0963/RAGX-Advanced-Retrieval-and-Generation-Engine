"""
RAGX Metadata Generator — Produce validated metadata for ingested documents.

Creates a rich metadata dictionary for every document, including a unique
document ID, word/char counts, timestamps, and optional per-page /
per-section context.  A Pydantic ``DocumentMetadata`` model is used for
validation so that downstream consumers get consistent, typed metadata.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from ragx.config.logging_config import get_logger

logger = get_logger(__name__)


class DocumentMetadata(BaseModel):
    """Pydantic model that validates and serialises document metadata.

    All fields are set by :class:`MetadataGenerator`.  Optional fields
    default to ``None`` and are excluded from the dict when absent.
    """

    document_id: str = Field(
        ..., description="Unique UUID4 identifier for the document."
    )
    source: str = Field(
        ..., description="Original file path or URL."
    )
    upload_time: str = Field(
        ..., description="ISO 8601 UTC timestamp of ingestion."
    )
    file_type: str = Field(
        ..., description="File extension (e.g. '.pdf') or 'url'."
    )
    word_count: int = Field(
        ..., ge=0, description="Number of whitespace-delimited words."
    )
    char_count: int = Field(
        ..., ge=0, description="Total character count."
    )
    page_number: Optional[int] = Field(
        default=None, description="1-based page number (if applicable)."
    )
    section: Optional[str] = Field(
        default=None, description="Section heading (if applicable)."
    )

    model_config = {"extra": "allow"}


class MetadataGenerator:
    """Generate validated metadata dictionaries for ingested documents.

    Usage::

        gen = MetadataGenerator()
        meta = gen.generate(
            source="/data/report.pdf",
            content="Some text ...",
            page_number=3,
        )
    """

    def generate(self, source: str, content: str, **extra: Any) -> dict[str, Any]:
        """Create a metadata dictionary for a single document.

        Args:
            source: The original file path or URL of the document.
            content: The textual content of the document.
            **extra: Arbitrary additional metadata fields.  Well-known
                keys (``page_number``, ``section``) are extracted and
                placed into the validated model.

        Returns:
            A plain ``dict`` produced by the validated
            :class:`DocumentMetadata` model (with ``None`` values
            excluded).
        """
        # ── Derive file type ────────────────────────────────────────────
        parsed = urlparse(source)
        if parsed.scheme in ("http", "https"):
            file_type = "url"
        else:
            file_type = Path(source).suffix.lower() or "unknown"

        # ── Word / char counts ──────────────────────────────────────────
        word_count = len(content.split()) if content else 0
        char_count = len(content) if content else 0

        # ── Extract well-known optional fields from extra ───────────────
        page_number: Optional[int] = extra.pop("page_number", None)
        section: Optional[str] = extra.pop("section", None)

        if page_number is not None:
            page_number = int(page_number)

        # ── Build and validate ──────────────────────────────────────────
        meta = DocumentMetadata(
            document_id=str(uuid.uuid4()),
            source=source,
            upload_time=datetime.now(timezone.utc).isoformat(),
            file_type=file_type,
            word_count=word_count,
            char_count=char_count,
            page_number=page_number,
            section=section,
            **extra,
        )

        # Exclude None values for a clean dict
        result = meta.model_dump(exclude_none=True)
        logger.debug(
            "Metadata generated",
            document_id=result["document_id"],
            source=source,
        )
        return result
