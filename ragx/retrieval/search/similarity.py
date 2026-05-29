"""
RAGX Similarity Search — Dense vector similarity search.

Provides the SimilaritySearch class that wraps a vector store to perform
dense embedding-based nearest-neighbour retrieval with optional score
thresholding.
"""

from __future__ import annotations

from typing import Any

from ragx.config.logging_config import get_logger
from ragx.ingestion.loaders.base import Document

logger = get_logger(__name__)


class SimilaritySearch:
    """Dense vector similarity search over a vector store.

    Wraps either a RAGX BaseVectorStore or a LangChain-compatible
    vectorstore to provide a uniform search interface.

    Attributes:
        vectorstore: The underlying vector store used for search.
    """

    def __init__(self, vectorstore: Any) -> None:
        """Initialise the similarity search.

        Args:
            vectorstore: A RAGX BaseVectorStore or LangChain-compatible
                vectorstore instance that exposes ``similarity_search``
                and ``similarity_search_with_score`` (or
                ``similarity_search_with_relevance_scores``) methods.
        """
        self.vectorstore = vectorstore
        logger.info(
            "similarity_search_initialized",
            vectorstore_type=type(vectorstore).__name__,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float | None = None,
    ) -> list[Document]:
        """Return the top-k documents most similar to *query*.

        Args:
            query: Natural-language query string.
            k: Maximum number of results to return.
            score_threshold: If provided, only documents with a relevance
                score **≥** this value are returned.  Scores are normalised
                to [0, 1] where 1 is a perfect match.

        Returns:
            A list of :class:`Document` objects ordered by descending
            relevance.
        """
        if score_threshold is not None:
            results = self.search_with_scores(query, k=k)
            filtered = [
                doc for doc, score in results if score >= score_threshold
            ]
            logger.debug(
                "similarity_search_filtered",
                query_preview=query[:80],
                k=k,
                threshold=score_threshold,
                total=len(results),
                after_filter=len(filtered),
            )
            return filtered

        docs = self._do_search(query, k=k)
        logger.debug(
            "similarity_search_completed",
            query_preview=query[:80],
            k=k,
            results=len(docs),
        )
        return docs

    def search_with_scores(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        """Return top-k documents with their similarity scores.

        Args:
            query: Natural-language query string.
            k: Maximum number of results.

        Returns:
            A list of ``(Document, score)`` tuples ordered by descending
            relevance.  Scores are in [0, 1].
        """
        results = self._do_search_with_scores(query, k=k)
        logger.debug(
            "similarity_search_with_scores_completed",
            query_preview=query[:80],
            k=k,
            results=len(results),
        )
        return results

    # ── Internal helpers ─────────────────────────────────────────────────

    def _do_search(self, query: str, k: int) -> list[Document]:
        """Execute a plain similarity search against the vectorstore."""
        try:
            # LangChain vectorstores expose similarity_search()
            docs = self.vectorstore.similarity_search(query, k=k)
            if docs and isinstance(docs[0], Document):
                return docs
            return [Document.from_langchain(d) for d in docs]
        except Exception:
            logger.exception(
                "similarity_search_error", query_preview=query[:80], k=k
            )
            raise

    def _do_search_with_scores(
        self, query: str, k: int
    ) -> list[tuple[Document, float]]:
        """Execute a scored similarity search.

        Tries ``similarity_search_with_relevance_scores`` first (returns
        normalised scores), then falls back to
        ``similarity_search_with_score`` (distance-based), converting
        distances to relevance scores.
        """
        try:
            # Preferred: normalised relevance scores in [0, 1]
            if hasattr(
                self.vectorstore, "similarity_search_with_relevance_scores"
            ):
                raw = self.vectorstore.similarity_search_with_relevance_scores(
                    query, k=k
                )
                if raw and isinstance(raw[0][0], Document):
                    return [(doc, float(score)) for doc, score in raw]
                return [
                    (Document.from_langchain(doc), float(score))
                    for doc, score in raw
                ]

            # Fallback: raw distance scores (lower = better)
            if hasattr(self.vectorstore, "similarity_search_with_score"):
                raw = self.vectorstore.similarity_search_with_score(
                    query, k=k
                )
                if raw and isinstance(raw[0][0], Document):
                    return [
                        (doc, self._distance_to_relevance(float(score)))
                        for doc, score in raw
                    ]
                return [
                    (
                        Document.from_langchain(doc),
                        self._distance_to_relevance(float(score)),
                    )
                    for doc, score in raw
                ]

            # Last resort: plain search, assign 0.0 scores
            logger.warning(
                "vectorstore_no_score_method",
                vectorstore_type=type(self.vectorstore).__name__,
            )
            docs = self._do_search(query, k=k)
            return [(doc, 0.0) for doc in docs]

        except Exception:
            logger.exception(
                "similarity_search_with_scores_error",
                query_preview=query[:80],
                k=k,
            )
            raise

    @staticmethod
    def _distance_to_relevance(distance: float) -> float:
        """Convert a non-negative distance to a [0, 1] relevance score.

        Uses the formula ``1 / (1 + distance)`` so that distance 0 maps
        to relevance 1.0 and large distances approach 0.
        """
        return 1.0 / (1.0 + distance)
