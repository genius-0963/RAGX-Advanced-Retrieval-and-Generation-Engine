"""
RAGX Cross-Encoder Reranking — Rerank retrieved documents using cross-encoder models.

Uses pairwise (query, document) scoring for high-accuracy relevance ranking.
"""

from __future__ import annotations

from ragx.config.logging_config import get_logger
from ragx.ingestion.loaders.base import Document

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Reranks documents using a cross-encoder model.

    Cross-encoders jointly encode (query, document) pairs for more accurate
    relevance scoring compared to bi-encoder similarity search.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
    ) -> None:
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model name.
            top_n: Default number of top documents to return.
        """
        self.model_name = model_name
        self.top_n = top_n
        self._model = None

    def _get_model(self):
        """Lazily load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info("cross_encoder_loaded", model=self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        """
        Rerank documents by cross-encoder relevance score.

        Args:
            query: Search query.
            documents: List of documents to rerank.
            top_n: Number of top documents to return. Uses default if None.

        Returns:
            Reranked list of Documents (highest relevance first).
        """
        scored = self.rerank_with_scores(query, documents, top_n)
        return [doc for doc, _ in scored]

    def rerank_with_scores(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Rerank documents and return with relevance scores.

        Args:
            query: Search query.
            documents: List of documents to rerank.
            top_n: Number of top documents to return.

        Returns:
            List of (Document, score) tuples sorted by descending score.
        """
        n = top_n if top_n is not None else self.top_n

        if not documents:
            return []

        model = self._get_model()

        # Create (query, document) pairs for scoring
        pairs = [(query, doc.content) for doc in documents]

        try:
            scores = model.predict(pairs)
        except Exception as e:
            logger.error("cross_encoder_scoring_failed", error=str(e))
            return [(doc, 0.0) for doc in documents[:n]]

        # Pair documents with scores and sort descending
        doc_scores = list(zip(documents, [float(s) for s in scores]))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        result = doc_scores[:n]

        logger.info(
            "cross_encoder_reranked",
            input_docs=len(documents),
            output_docs=len(result),
            top_score=result[0][1] if result else 0.0,
        )
        return result
