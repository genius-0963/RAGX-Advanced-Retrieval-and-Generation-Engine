"""
RAGX Retrieval Engine — Orchestrates the full retrieval pipeline.

Combines search strategies (similarity, BM25, hybrid), advanced retrieval
(multi-query, parent-child, compression), and reranking into a configurable pipeline.
"""

from __future__ import annotations

from typing import Any

from ragx.config.logging_config import get_logger
from ragx.config.settings import Settings, get_settings
from ragx.ingestion.loaders.base import Document

logger = get_logger(__name__)


class RetrievalEngine:
    """
    Orchestrates the full retrieval pipeline.

    Configurable via Settings with per-call overrides for:
    - Search strategy (similarity, BM25, hybrid)
    - Multi-query retrieval
    - Reranking (cross-encoder, Cohere)
    - Context compression
    """

    def __init__(
        self,
        settings: Settings | None = None,
        vectorstore: Any = None,
        embeddings: Any = None,
        llm: Any = None,
    ) -> None:
        """
        Initialize retrieval engine.

        Args:
            settings: Application settings. Uses defaults if None.
            vectorstore: Vector store for dense retrieval.
            embeddings: Embedding model for query encoding.
            llm: LLM for multi-query and compression strategies.
        """
        self.settings = settings or get_settings()
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.llm = llm

        # Lazy-initialized components
        self._similarity_search = None
        self._bm25_search = None
        self._hybrid_search = None
        self._reranker = None
        self._indexed_for_bm25 = False

    def _get_similarity_search(self):
        """Lazily initialize similarity search."""
        if self._similarity_search is None and self.vectorstore is not None:
            from ragx.retrieval.search.similarity import SimilaritySearch
            self._similarity_search = SimilaritySearch(self.vectorstore)
        return self._similarity_search

    def _get_bm25_search(self):
        """Lazily initialize BM25 search."""
        if self._bm25_search is None:
            from ragx.retrieval.search.bm25 import BM25Search
            self._bm25_search = BM25Search()
        return self._bm25_search

    def _get_hybrid_search(self):
        """Lazily initialize hybrid search."""
        if self._hybrid_search is None:
            sim = self._get_similarity_search()
            bm25 = self._get_bm25_search()
            if sim is not None and bm25 is not None:
                from ragx.retrieval.search.hybrid import HybridSearch
                self._hybrid_search = HybridSearch(sim, bm25, dense_weight=0.6)
        return self._hybrid_search

    def _get_reranker(self, reranker_type: str | None = None):
        """Get or initialize the appropriate reranker."""
        rtype = reranker_type or self.settings.reranker.value

        if rtype == "none":
            return None

        if rtype == "cross-encoder":
            from ragx.retrieval.reranking.cross_encoder import CrossEncoderReranker
            return CrossEncoderReranker(top_n=self.settings.reranker_top_n)
        elif rtype == "cohere":
            from ragx.retrieval.reranking.cohere_rerank import CohereReranker
            return CohereReranker(top_n=self.settings.reranker_top_n)
        else:
            logger.warning("unknown_reranker_type", reranker=rtype)
            return None

    def index_documents(self, documents: list[Document]) -> None:
        """
        Index documents for BM25 search.

        Args:
            documents: Documents to index.
        """
        bm25 = self._get_bm25_search()
        if bm25 is not None:
            bm25.index(documents)
            self._indexed_for_bm25 = True
            logger.info("documents_indexed_for_bm25", count=len(documents))

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        strategy: str | None = None,
        use_reranker: bool = True,
        use_multi_query: bool = False,
        use_compression: bool = False,
    ) -> list[Document]:
        """
        Execute the full retrieval pipeline.

        Pipeline: query → search → (multi-query) → (rerank) → (compress) → top-k

        Args:
            query: User search query.
            k: Number of results to return. Uses settings default if None.
            strategy: Search strategy override ('similarity', 'bm25', 'hybrid').
            use_reranker: Whether to apply reranking.
            use_multi_query: Whether to use multi-query expansion.
            use_compression: Whether to compress retrieved context.

        Returns:
            List of relevant Documents.
        """
        top_k = k if k is not None else self.settings.retrieval_top_k
        search_strategy = strategy or self.settings.retrieval_strategy.value

        logger.info(
            "retrieval_started",
            query_preview=query[:80],
            strategy=search_strategy,
            top_k=top_k,
            use_reranker=use_reranker,
        )

        # Step 1: Multi-query expansion (optional)
        if use_multi_query and self.llm is not None:
            documents = self._multi_query_retrieve(query, search_strategy, top_k)
        else:
            documents = self._search(query, search_strategy, fetch_k=top_k * 3)

        # Step 2: Reranking (optional)
        if use_reranker and documents:
            reranker = self._get_reranker()
            if reranker is not None:
                try:
                    documents = reranker.rerank(query, documents, top_n=top_k)
                except Exception as e:
                    logger.warning("reranking_failed", error=str(e))
                    documents = documents[:top_k]
            else:
                documents = documents[:top_k]
        else:
            documents = documents[:top_k]

        # Step 3: Context compression (optional)
        if use_compression and documents:
            try:
                from ragx.retrieval.strategies.compression import ContextCompressor

                method = "llm" if self.llm is not None else "embeddings"
                compressor = ContextCompressor(
                    retriever=None,
                    llm=self.llm if method == "llm" else None,
                    embeddings=self.embeddings if method == "embeddings" else None,
                    method=method,
                )
                if method == "llm":
                    documents = compressor._compress_with_llm(query, documents)
                else:
                    documents = compressor._compress_with_embeddings(query, documents)
            except Exception as e:
                logger.warning("compression_failed", error=str(e))

        logger.info("retrieval_complete", num_results=len(documents))
        return documents

    def _search(
        self, query: str, strategy: str, fetch_k: int
    ) -> list[Document]:
        """Execute search with the specified strategy."""
        if strategy == "similarity":
            searcher = self._get_similarity_search()
            if searcher is None:
                logger.error("no_vectorstore_for_similarity_search")
                return []
            return searcher.search(query, k=fetch_k)

        elif strategy == "bm25":
            searcher = self._get_bm25_search()
            if searcher is None or not self._indexed_for_bm25:
                logger.warning("bm25_not_indexed, falling back to similarity")
                return self._search(query, "similarity", fetch_k)
            return searcher.search(query, k=fetch_k)

        elif strategy == "hybrid":
            searcher = self._get_hybrid_search()
            if searcher is None:
                logger.warning("hybrid_search_unavailable, falling back to similarity")
                return self._search(query, "similarity", fetch_k)
            if not self._indexed_for_bm25:
                logger.warning("bm25_not_indexed, falling back to similarity")
                return self._search(query, "similarity", fetch_k)
            return searcher.search(query, k=fetch_k)

        else:
            logger.warning("unknown_strategy, falling back to similarity", strategy=strategy)
            return self._search(query, "similarity", fetch_k)

    def _multi_query_retrieve(
        self, query: str, strategy: str, k: int
    ) -> list[Document]:
        """Use multi-query retrieval for better recall."""
        try:
            from ragx.retrieval.strategies.multi_query import MultiQueryRetriever

            class SimpleRetrieverAdapter:
                """Adapter to wrap our search as a retriever."""
                def __init__(self, engine, strategy, k):
                    self.engine = engine
                    self.strategy = strategy
                    self.k = k

                def invoke(self, query: str) -> list:
                    from langchain_core.documents import Document as LCDoc
                    docs = self.engine._search(query, self.strategy, self.k)
                    return [
                        LCDoc(page_content=d.content, metadata=d.metadata)
                        for d in docs
                    ]

            adapter = SimpleRetrieverAdapter(self, strategy, k * 2)
            mq_retriever = MultiQueryRetriever(retriever=adapter, llm=self.llm)
            return mq_retriever.retrieve(query, num_queries=3, k=k * 2)
        except Exception as e:
            logger.warning("multi_query_failed, falling back to standard", error=str(e))
            return self._search(query, strategy, k * 2)

    def get_retriever(self, **kwargs):
        """
        Return a LangChain-compatible retriever.

        Args:
            **kwargs: Passed to the retrieve method.

        Returns:
            Object with an invoke(query) method compatible with LangChain.
        """
        engine = self

        class RAGXRetriever:
            """LangChain-compatible retriever wrapping the RetrievalEngine."""
            def __init__(self, engine, **kwargs):
                self._engine = engine
                self._kwargs = kwargs

            def invoke(self, query: str) -> list:
                from langchain_core.documents import Document as LCDoc
                docs = self._engine.retrieve(query, **self._kwargs)
                return [
                    LCDoc(page_content=d.content, metadata=d.metadata)
                    for d in docs
                ]

            def get_relevant_documents(self, query: str) -> list:
                return self.invoke(query)

        return RAGXRetriever(engine, **kwargs)
