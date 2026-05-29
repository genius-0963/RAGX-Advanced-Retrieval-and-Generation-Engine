"""
RAGX Base Loader — Core Document dataclass and abstract base loader.

Defines the canonical Document representation used throughout the
RAGX pipeline for ingestion, chunking, embedding, and retrieval.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Document:
    """
    Canonical document representation for the RAGX pipeline.

    Attributes:
        content: The textual content of the document or chunk.
        metadata: Arbitrary metadata dict (source, page, timestamps, etc.).
        doc_id: Unique identifier for the document. Auto-generated if not provided.
        content_hash: SHA-256 hash of the content for deduplication.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content_hash: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        """Compute content hash if not already set."""
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.content.encode("utf-8")
            ).hexdigest()

    def to_langchain(self) -> Any:
        """
        Convert to a LangChain Document.

        Returns:
            A ``langchain_core.documents.Document`` instance.
        """
        from langchain_core.documents import Document as LCDocument

        metadata = {**self.metadata, "doc_id": self.doc_id, "content_hash": self.content_hash}
        return LCDocument(page_content=self.content, metadata=metadata)

    @classmethod
    def from_langchain(cls, lc_doc: Any) -> Document:
        """
        Create a RAGX Document from a LangChain Document.

        Args:
            lc_doc: A ``langchain_core.documents.Document`` instance.

        Returns:
            Equivalent RAGX Document.
        """
        metadata = dict(lc_doc.metadata) if lc_doc.metadata else {}
        doc_id = metadata.pop("doc_id", str(uuid.uuid4()))
        content_hash = metadata.pop("content_hash", "")
        return cls(
            content=lc_doc.page_content,
            metadata=metadata,
            doc_id=doc_id,
            content_hash=content_hash,
        )

    @property
    def created_at(self) -> str | None:
        """Return creation timestamp from metadata, if present."""
        return self.metadata.get("created_at")

    @property
    def source(self) -> str | None:
        """Return source path/URL from metadata, if present."""
        return self.metadata.get("source")

from abc import ABC, abstractmethod
from typing import Iterator

class BaseLoader(ABC):
    """
    Abstract base class for all document loaders.
    """
    supported_extensions: list[str] = []

    @abstractmethod
    def load(self, source: str) -> list[Document]:
        """Load documents from a source."""
        pass

    def lazy_load(self, source: str) -> Iterator[Document]:
        """Lazily load documents from a source."""
        yield from self.load(source)
