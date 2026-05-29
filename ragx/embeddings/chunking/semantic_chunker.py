"""
RAGX Semantic Chunker — Embedding-aware semantic text chunking.

Wraps LangChain's experimental SemanticChunker to split documents at
semantically meaningful boundaries. Falls back to recursive splitting
if semantic chunking fails.
"""

from __future__ import annotations

import uuid
from typing import Any

from ragx.config.logging_config import get_logger
from ragx.ingestion.loaders.base import Document

logger = get_logger(__name__)


class SemanticChunkerWrapper:
    """
    Semantic text chunker using embedding similarity to find breakpoints.

    Groups consecutive sentences whose embeddings are similar, splitting
    only where the semantic similarity drops below a threshold.

    Falls back to :class:`RecursiveChunker` if semantic chunking raises
    an exception (e.g. missing embedding model, empty documents).

    Attributes:
        breakpoint_threshold_type: The strategy for detecting breakpoints.
            One of ``'percentile'``, ``'standard_deviation'``, or ``'interquartile'``.
    """

    def __init__(
        self,
        embeddings: Any = None,
        breakpoint_threshold_type: str = "percentile",
    ) -> None:
        """
        Initialize the semantic chunker.

        Args:
            embeddings: A LangChain-compatible embeddings object. If ``None``,
                the chunker will use a default :class:`HuggingFaceEmbeddings`
                model (``all-MiniLM-L6-v2``).
            breakpoint_threshold_type: Breakpoint detection strategy.
                One of ``'percentile'``, ``'standard_deviation'``,
                or ``'interquartile'``.
        """
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self._embeddings = embeddings
        self._chunker: Any = None

        self._initialize_chunker()

    def _initialize_chunker(self) -> None:
        """
        Lazily build the underlying LangChain SemanticChunker.

        If the embeddings object was not supplied, a default
        HuggingFaceEmbeddings instance is created.
        """
        try:
            if self._embeddings is None:
                from langchain_huggingface import HuggingFaceEmbeddings

                self._embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )
                logger.info(
                    "SemanticChunker: created default HuggingFaceEmbeddings "
                    "(all-MiniLM-L6-v2)"
                )

            from langchain_experimental.text_splitter import SemanticChunker

            self._chunker = SemanticChunker(
                embeddings=self._embeddings,
                breakpoint_threshold_type=self.breakpoint_threshold_type,
            )
            logger.info(
                "Initialized SemanticChunker",
                breakpoint_threshold_type=self.breakpoint_threshold_type,
            )
        except Exception:
            logger.exception(
                "Failed to initialize SemanticChunker — "
                "will fall back to RecursiveChunker at split time"
            )
            self._chunker = None

    def _fallback_split(self, documents: list[Document]) -> list[Document]:
        """
        Fall back to recursive splitting.

        Args:
            documents: Documents to split.

        Returns:
            Chunked documents produced by :class:`RecursiveChunker`.
        """
        from ragx.embeddings.chunking.recursive_splitter import RecursiveChunker

        logger.warning("Falling back to RecursiveChunker for splitting")
        fallback = RecursiveChunker()
        return fallback.split(documents)

    def split(self, documents: list[Document]) -> list[Document]:
        """
        Split documents using semantic similarity boundaries.

        If the semantic chunker is unavailable or fails for a given
        document, the method transparently falls back to recursive
        character splitting.

        Args:
            documents: List of RAGX Documents to split.

        Returns:
            List of chunked RAGX Documents with enriched metadata.
        """
        if not documents:
            logger.warning("No documents provided for semantic splitting")
            return []

        if self._chunker is None:
            return self._fallback_split(documents)

        all_chunks: list[Document] = []
        failed_docs: list[Document] = []

        for doc in documents:
            try:
                lc_doc = doc.to_langchain()
                lc_chunks = self._chunker.split_documents([lc_doc])

                for idx, lc_chunk in enumerate(lc_chunks):
                    chunk_metadata = dict(lc_chunk.metadata)
                    chunk_metadata["chunk_index"] = idx
                    chunk_metadata["parent_doc_id"] = doc.doc_id
                    chunk_metadata["total_chunks"] = len(lc_chunks)
                    chunk_metadata["chunking_strategy"] = "semantic"

                    chunk = Document(
                        content=lc_chunk.page_content,
                        metadata=chunk_metadata,
                        doc_id=str(uuid.uuid4()),
                    )
                    all_chunks.append(chunk)

            except Exception:
                logger.exception(
                    "Semantic chunking failed for document — "
                    "will attempt recursive fallback",
                    doc_id=doc.doc_id,
                    source=doc.source,
                )
                failed_docs.append(doc)

        # Attempt recursive fallback for any documents that failed
        if failed_docs:
            fallback_chunks = self._fallback_split(failed_docs)
            all_chunks.extend(fallback_chunks)

        logger.info(
            "Semantic splitting complete",
            input_docs=len(documents),
            output_chunks=len(all_chunks),
            fallback_count=len(failed_docs),
        )
        return all_chunks
