"""
RAGX Cohere Reranking — Rerank documents using Cohere's Rerank API.

Falls back to returning original order if the API is unavailable.
"""

from __future__ import annotations

from ragx.config.logging_config import get_logger
from ragx.config.settings import get_settings
from ragx.ingestion.loaders.base import Document

logger = get_logger(__name__)


class CohereReranker:
    """
    Reranks documents using Cohere's Rerank API.

    Provides a managed API alternative to local cross-encoder models.
    Falls back gracefully if the API is unavailable.
    """

    def __init__(
        self,
        model: str = "rerank-english-v3.0",
        api_key: str | None = None,
        top_n: int = 5,
    ) -> None:
        """
        Initialize Cohere reranker.

        Args:
            model: Cohere rerank model name.
            api_key: Cohere API key. Falls back to settings/env if None.
            top_n: Default number of top documents to return.
        """
        self.model = model
        self.top_n = top_n
        settings = get_settings()
        self.api_key = api_key or settings.cohere_api_key
        self._reranker = None

    def _get_reranker(self):
        """Lazily initialize the Cohere reranker."""
        if self._reranker is None:
            try:
                import cohere
                self._client = cohere.Client(api_key=self.api_key)
                logger.info("cohere_reranker_initialized", model=self.model)
            except Exception as e:
                logger.error("cohere_init_failed", error=str(e))
                self._client = None
        return self._client

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        """
        Rerank documents using Cohere API.

        Args:
            query: Search query.
            documents: List of documents to rerank.
            top_n: Number of top documents to return.

        Returns:
            Reranked list of Documents.
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

        client = self._get_reranker()

        if client is None:
            logger.warning("cohere_unavailable, returning original order")
            return [(doc, 0.0) for doc in documents[:n]]

        doc_texts = [doc.content for doc in documents]

        try:
            response = client.rerank(
                model=self.model,
                query=query,
                documents=doc_texts,
                top_n=n,
            )

            results: list[tuple[Document, float]] = []
            for item in response.results:
                idx = item.index
                score = float(item.relevance_score)
                results.append((documents[idx], score))

            logger.info(
                "cohere_reranked",
                input_docs=len(documents),
                output_docs=len(results),
                top_score=results[0][1] if results else 0.0,
            )
            return results

        except Exception as e:
            logger.warning("cohere_rerank_failed, returning original order", error=str(e))
            return [(doc, 0.0) for doc in documents[:n]]
