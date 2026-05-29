"""
RAGX BM25 Search — Sparse keyword-based retrieval using BM25 Okapi.

Provides the BM25Search class that builds and queries a BM25 index over
a corpus of RAGX Document objects, supporting incremental add/remove
operations and automatic NLTK tokenizer bootstrapping.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from ragx.config.logging_config import get_logger
from ragx.ingestion.loaders.base import Document

logger = get_logger(__name__)


def _ensure_nltk_punkt() -> None:
    """Download the NLTK punkt tokeniser if it is not already available.

    The download is performed silently and only once per environment.
    """
    try:
        import nltk  # type: ignore[import-untyped]

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            logger.info("downloading_nltk_punkt_tab")
            # Suppress NLTK download messages in production
            nltk.download("punkt_tab", quiet=True)
    except Exception:
        logger.exception("nltk_punkt_download_failed")
        raise


def _tokenize(text: str) -> list[str]:
    """Tokenise *text* into lower-cased word tokens using NLTK.

    Args:
        text: The raw text to tokenise.

    Returns:
        A list of lower-cased word tokens.
    """
    from nltk.tokenize import word_tokenize  # type: ignore[import-untyped]

    return [token.lower() for token in word_tokenize(text)]


class BM25Search:
    """BM25 Okapi sparse search over a corpus of Documents.

    The index is built lazily on first search if documents are provided at
    construction time, or explicitly via :meth:`index`.

    Attributes:
        _documents: The indexed document corpus.
        _bm25: The underlying ``rank_bm25.BM25Okapi`` instance.
        _tokenized_corpus: Pre-tokenised corpus used by BM25.
    """

    def __init__(self, documents: list[Document] | None = None) -> None:
        """Initialise BM25Search, optionally indexing *documents*.

        Args:
            documents: Optional list of :class:`Document` objects to index
                immediately.
        """
        self._documents: list[Document] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []
        self._dirty: bool = False

        # Ensure punkt is available before any tokenisation
        _ensure_nltk_punkt()

        if documents:
            self.index(documents)

        logger.info(
            "bm25_search_initialized", document_count=len(self._documents)
        )

    # ── Index management ─────────────────────────────────────────────────

    def index(self, documents: list[Document]) -> None:
        """Build (or rebuild) the BM25 index from *documents*.

        Args:
            documents: The full document corpus to index.  Any previous
                index is discarded.
        """
        if not documents:
            logger.warning("bm25_index_called_with_empty_corpus")
            self._documents = []
            self._tokenized_corpus = []
            self._bm25 = None
            return

        self._documents = list(documents)
        self._tokenized_corpus = [_tokenize(doc.content) for doc in self._documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._dirty = False
        logger.info("bm25_index_built", document_count=len(self._documents))

    def add_documents(self, documents: list[Document]) -> None:
        """Add *documents* to the existing index.

        The BM25 index is rebuilt to incorporate the new documents.

        Args:
            documents: Documents to append to the corpus.
        """
        if not documents:
            return

        self._documents.extend(documents)
        new_tokens = [_tokenize(doc.content) for doc in documents]
        self._tokenized_corpus.extend(new_tokens)
        # Rebuild the BM25 model with the extended corpus
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._dirty = False
        logger.info(
            "bm25_documents_added",
            added=len(documents),
            total=len(self._documents),
        )

    def remove_documents(self, doc_ids: set[str]) -> None:
        """Remove documents by their *doc_id* and rebuild the index.

        Args:
            doc_ids: Set of document IDs to remove.
        """
        if not doc_ids:
            return

        original_count = len(self._documents)
        paired = [
            (doc, tokens)
            for doc, tokens in zip(self._documents, self._tokenized_corpus)
            if doc.doc_id not in doc_ids
        ]

        if paired:
            self._documents, self._tokenized_corpus = map(list, zip(*paired))  # type: ignore[arg-type]
        else:
            self._documents = []
            self._tokenized_corpus = []

        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None

        removed = original_count - len(self._documents)
        logger.info(
            "bm25_documents_removed",
            removed=removed,
            remaining=len(self._documents),
        )

    # ── Search API ───────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[Document]:
        """Return the top-k documents by BM25 score for *query*.

        Args:
            query: Natural-language query string.
            k: Maximum number of results.

        Returns:
            A list of :class:`Document` objects sorted by descending
            BM25 score.
        """
        results = self.search_with_scores(query, k=k)
        return [doc for doc, _score in results]

    def search_with_scores(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        """Return top-k documents with their BM25 scores.

        Args:
            query: Natural-language query string.
            k: Maximum number of results.

        Returns:
            A list of ``(Document, score)`` tuples sorted by descending
            BM25 score.
        """
        if not self._bm25 or not self._documents:
            logger.warning("bm25_search_on_empty_index")
            return []

        tokenized_query = _tokenize(query)
        scores: np.ndarray = self._bm25.get_scores(tokenized_query)

        # Get indices of top-k scores in descending order
        top_k = min(k, len(self._documents))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            (self._documents[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0.0
        ]

        logger.debug(
            "bm25_search_completed",
            query_preview=query[:80],
            k=k,
            results=len(results),
        )
        return results

    @property
    def document_count(self) -> int:
        """Return the number of indexed documents."""
        return len(self._documents)
