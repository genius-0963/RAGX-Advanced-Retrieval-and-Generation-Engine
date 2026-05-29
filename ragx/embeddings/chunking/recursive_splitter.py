"""
RAGX Recursive Splitter — Recursive character-based text chunking.

Wraps LangChain's RecursiveCharacterTextSplitter to split documents
into overlapping chunks while preserving metadata and provenance.
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragx.config.logging_config import get_logger
from ragx.ingestion.loaders.base import Document

logger = get_logger(__name__)


class RecursiveChunker:
    """
    Recursive character text splitter for document chunking.

    Splits text hierarchically using a sequence of separators,
    falling through to smaller separators when chunks exceed
    the target size.

    Attributes:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
        separators: Ordered list of separator strings to split on.
    """

    DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
    ) -> None:
        """
        Initialize the recursive chunker.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between consecutive chunks.
            separators: Ordered list of separator strings. Defaults to
                ``['\\n\\n', '\\n', '. ', ' ', '']``.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )
        logger.info(
            "Initialized RecursiveChunker",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            num_separators=len(self.separators),
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """
        Split a list of RAGX Documents into smaller chunks.

        Each resulting chunk inherits the parent document's metadata,
        augmented with ``chunk_index``, ``parent_doc_id``, and
        ``total_chunks`` fields.

        Args:
            documents: List of RAGX Documents to split.

        Returns:
            List of chunked RAGX Documents with preserved and enriched metadata.
        """
        if not documents:
            logger.warning("No documents provided for splitting")
            return []

        all_chunks: list[Document] = []

        for doc in documents:
            try:
                # Convert to LangChain doc for the splitter
                lc_doc = doc.to_langchain()
                lc_chunks = self._splitter.split_documents([lc_doc])

                for idx, lc_chunk in enumerate(lc_chunks):
                    chunk_metadata = dict(lc_chunk.metadata)
                    chunk_metadata["chunk_index"] = idx
                    chunk_metadata["parent_doc_id"] = doc.doc_id
                    chunk_metadata["total_chunks"] = len(lc_chunks)

                    chunk = Document(
                        content=lc_chunk.page_content,
                        metadata=chunk_metadata,
                        doc_id=str(uuid.uuid4()),
                    )
                    all_chunks.append(chunk)

            except Exception:
                logger.exception(
                    "Failed to split document",
                    doc_id=doc.doc_id,
                    source=doc.source,
                )
                # On failure, keep the original document as a single chunk
                all_chunks.append(doc)

        logger.info(
            "Recursive splitting complete",
            input_docs=len(documents),
            output_chunks=len(all_chunks),
        )
        return all_chunks

    def split_text(self, text: str) -> list[str]:
        """
        Split raw text into a list of text chunks.

        Args:
            text: Raw text string to split.

        Returns:
            List of text chunk strings.
        """
        if not text:
            return []

        chunks = self._splitter.split_text(text)
        logger.debug(
            "Split raw text",
            input_length=len(text),
            num_chunks=len(chunks),
        )
        return chunks
