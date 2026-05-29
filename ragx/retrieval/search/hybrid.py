"""
RAGX Hybrid Search — Combines BM25 and dense vector search via RRF.

Uses Reciprocal Rank Fusion (RRF) to merge ranked result lists from
BM25 sparse search and dense similarity search, producing a single
de-duplicated ranking that benefits from both lexical and semantic signals.
"""

from __future__ import annotations

from ragx.config.logging_config import get_logger
from ragx.ingestion.loaders.base import Document
from ragx.retrieval.search.bm25 import BM25Search
from ragx.retrieval.search.similarity import SimilaritySearch

logger = get_logger(__name__)

# RRF constant — controls how much weight is given to highly-ranked
# results relative to lower-ranked ones.  60 is the standard value from
# the original Cormack et al. paper.
_RRF_K = 60


class HybridSearch:
    """Hybrid search combining dense and sparse retrieval via RRF.

    Reciprocal Rank Fusion produces a combined score for each document:
    ``score(d) = Σ  1 / (rank_i(d) + k)`` across all result lists,
    where *k* = 60 by default.

    Attributes:
        similarity_search: Dense vector search component.
        bm25_search: BM25 sparse search component.
        dense_weight: Relative weight for the dense search scores.
    """

    def __init__(
        self,
        similarity_search: SimilaritySearch,
        bm25_search: BM25Search,
        dense_weight: float = 0.6,
    ) -> None:
        """Initialise hybrid search.

        Args:
            similarity_search: Dense vector similarity search instance.
            bm25_search: BM25 sparse search instance.
            dense_weight: Weight for dense search results in [0, 1].
                The BM25 weight is ``1 - dense_weight``.
        """
        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError(
                f"dense_weight must be in [0, 1], got {dense_weight}"
            )

        self.similarity_search = similarity_search
        self.bm25_search = bm25_search
        self.dense_weight = dense_weight
        self._sparse_weight = 1.0 - dense_weight

        logger.info(
            "hybrid_search_initialized",
            dense_weight=dense_weight,
            sparse_weight=self._sparse_weight,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[Document]:
        """Return the top-k documents from a hybrid RRF ranking.

        Args:
            query: Natural-language query string.
            k: Maximum number of results to return.

        Returns:
            De-duplicated list of :class:`Document` objects ranked by
            combined RRF score.
        """
        scored = self.search_with_scores(query, k=k)
        return [doc for doc, _score in scored]

    def search_with_scores(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        """Return top-k documents with their RRF fusion scores.

        Args:
            query: Natural-language query string.
            k: Maximum number of results.

        Returns:
            A list of ``(Document, rrf_score)`` tuples sorted by
            descending fused score.
        """
        # Retrieve more candidates than needed from each source so
        # fusion has a richer pool.
        fetch_k = k * 3

        dense_docs = self.similarity_search.search(query, k=fetch_k)
        sparse_docs = self.bm25_search.search(query, k=fetch_k)

        fused = self._reciprocal_rank_fusion(
            dense_docs, sparse_docs
        )

        top_k = fused[:k]

        logger.debug(
            "hybrid_search_completed",
            query_preview=query[:80],
            k=k,
            dense_count=len(dense_docs),
            sparse_count=len(sparse_docs),
            fused_count=len(fused),
            returned=len(top_k),
        )
        return top_k

    # ── Reciprocal Rank Fusion ───────────────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        dense_docs: list[Document],
        sparse_docs: list[Document],
    ) -> list[tuple[Document, float]]:
        """Fuse two ranked result lists using weighted RRF.

        Args:
            dense_docs: Documents from dense search (ordered by relevance).
            sparse_docs: Documents from BM25 search (ordered by score).

        Returns:
            Merged, de-duplicated list of ``(Document, fused_score)``
            tuples sorted by descending score.
        """
        # Map content hash → (Document, accumulated RRF score)
        score_map: dict[str, tuple[Document, float]] = {}

        # Accumulate dense RRF scores
        for rank, doc in enumerate(dense_docs):
            rrf_score = self.dense_weight / (rank + _RRF_K)
            key = doc.content_hash
            if key in score_map:
                existing_doc, existing_score = score_map[key]
                score_map[key] = (existing_doc, existing_score + rrf_score)
            else:
                score_map[key] = (doc, rrf_score)

        # Accumulate sparse RRF scores
        for rank, doc in enumerate(sparse_docs):
            rrf_score = self._sparse_weight / (rank + _RRF_K)
            key = doc.content_hash
            if key in score_map:
                existing_doc, existing_score = score_map[key]
                score_map[key] = (existing_doc, existing_score + rrf_score)
            else:
                score_map[key] = (doc, rrf_score)

        # Sort by fused score descending
        fused = sorted(
            score_map.values(), key=lambda pair: pair[1], reverse=True
        )
        return fused
