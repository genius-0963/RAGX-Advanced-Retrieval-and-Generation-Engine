"""
Tests for RAGX Retrieval — metrics, search strategies (offline, no API/vector store).
"""

from __future__ import annotations

import pytest

from ragx.retrieval.metrics import (
    evaluate_batch,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ragx.retrieval.engine import RetrievalEngine
from ragx.ingestion.loaders.base import Document


# ── precision_at_k ────────────────────────────────────────────────────────────

class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"x", "y"}, 3) == 0.0

    def test_half_relevant(self):
        result = precision_at_k(["a", "b", "c", "d"], {"a", "c"}, 4)
        assert result == pytest.approx(0.5)

    def test_k_zero_returns_zero(self):
        assert precision_at_k(["a", "b"], {"a"}, 0) == 0.0

    def test_k_larger_than_retrieved(self):
        # Only checks up to len(retrieved), k=10 but only 2 docs
        result = precision_at_k(["a", "b"], {"a"}, 10)
        assert result == pytest.approx(0.1)  # 1/10

    def test_empty_retrieved(self):
        assert precision_at_k([], {"a", "b"}, 5) == 0.0


# ── recall_at_k ───────────────────────────────────────────────────────────────

class TestRecallAtK:
    def test_all_relevant_retrieved(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_none_relevant(self):
        assert recall_at_k(["a", "b", "c"], {"x", "y"}, 3) == 0.0

    def test_partial_recall(self):
        result = recall_at_k(["a", "b", "c"], {"a", "d"}, 3)
        assert result == pytest.approx(0.5)

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k(["a", "b"], set(), 5) == 0.0

    def test_k_zero_returns_zero(self):
        assert recall_at_k(["a", "b"], {"a"}, 0) == 0.0


# ── mrr ──────────────────────────────────────────────────────────────────────

class TestMRR:
    def test_first_result_relevant(self):
        assert mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)

    def test_second_result_relevant(self):
        assert mrr(["x", "a", "c"], {"a"}) == pytest.approx(0.5)

    def test_third_result_relevant(self):
        assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_relevant_found(self):
        assert mrr(["a", "b", "c"], {"x"}) == 0.0

    def test_empty_retrieved(self):
        assert mrr([], {"a"}) == 0.0


# ── ndcg_at_k ────────────────────────────────────────────────────────────────

class TestNDCGAtK:
    def test_perfect_ranking(self):
        # All relevant at top positions
        assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(1.0)

    def test_zero_k(self):
        assert ndcg_at_k(["a", "b"], {"a"}, 0) == 0.0

    def test_empty_relevant(self):
        assert ndcg_at_k(["a", "b"], set(), 3) == 0.0

    def test_worst_ranking(self):
        # Relevant doc at last position: NDCG < 1
        result = ndcg_at_k(["x", "y", "a"], {"a"}, 3)
        assert 0.0 < result < 1.0

    def test_no_relevant_in_retrieved(self):
        result = ndcg_at_k(["x", "y", "z"], {"a", "b"}, 3)
        assert result == 0.0


# ── evaluate_batch ────────────────────────────────────────────────────────────

class TestEvaluateBatch:
    def test_empty_input(self):
        result = evaluate_batch([])
        assert "aggregate" in result
        assert "per_query" in result
        assert result["per_query"] == []

    def test_single_perfect_query(self):
        queries = [{"retrieved": ["a", "b", "c"], "relevant": {"a", "b", "c"}, "k": 3}]
        result = evaluate_batch(queries)
        assert result["aggregate"]["avg_precision_at_k"] == pytest.approx(1.0)
        assert result["aggregate"]["avg_recall_at_k"] == pytest.approx(1.0)
        assert result["aggregate"]["avg_mrr"] == pytest.approx(1.0)
        assert result["aggregate"]["avg_ndcg_at_k"] == pytest.approx(1.0)

    def test_multiple_queries(self):
        queries = [
            {"retrieved": ["a", "b", "c"], "relevant": ["a"], "k": 3},
            {"retrieved": ["x", "b"], "relevant": ["b"], "k": 2},
        ]
        result = evaluate_batch(queries)
        assert result["aggregate"]["num_queries"] == 2
        assert len(result["per_query"]) == 2

    def test_default_k(self):
        queries = [{"retrieved": ["a", "b", "c", "d", "e"], "relevant": {"a"}}]
        result = evaluate_batch(queries)
        assert result["per_query"][0]["k"] == 5


# ── RetrievalEngine ──────────────────────────────────────────────────────────

class TestRetrievalEngine:
    def test_instantiation_no_vectorstore(self):
        engine = RetrievalEngine()
        assert engine is not None

    def test_retrieve_without_vectorstore_returns_empty(self):
        engine = RetrievalEngine(vectorstore=None)
        results = engine.retrieve("test query", strategy="similarity")
        assert results == []

    def test_retrieve_bm25_without_index_falls_back(self):
        """BM25 without indexing should fallback gracefully (no crash)."""
        engine = RetrievalEngine(vectorstore=None)
        results = engine.retrieve("test query", strategy="bm25")
        assert isinstance(results, list)

    def test_index_documents_for_bm25(self):
        engine = RetrievalEngine()
        docs = [
            Document(content="Python is a programming language", metadata={"source": "a"}),
            Document(content="RAGX uses ChromaDB for vector storage", metadata={"source": "b"}),
        ]
        engine.index_documents(docs)
        assert engine._indexed_for_bm25 is True

    def test_bm25_retrieve_after_indexing(self):
        engine = RetrievalEngine(vectorstore=None)
        docs = [
            Document(content="Python programming language", metadata={"source": "a"}),
            Document(content="RAGX retrieval engine ChromaDB", metadata={"source": "b"}),
            Document(content="Machine learning neural networks", metadata={"source": "c"}),
        ]
        engine.index_documents(docs)
        results = engine.retrieve("Python", strategy="bm25", use_reranker=False, k=2)
        assert isinstance(results, list)
        assert len(results) <= 2

    def test_get_retriever_returns_adapter(self):
        engine = RetrievalEngine(vectorstore=None)
        retriever = engine.get_retriever()
        assert hasattr(retriever, "invoke")
        assert hasattr(retriever, "get_relevant_documents")

    def test_list_sessions_initially_empty(self):
        """Engine itself doesn't have sessions, but shouldn't crash."""
        engine = RetrievalEngine()
        assert engine is not None
