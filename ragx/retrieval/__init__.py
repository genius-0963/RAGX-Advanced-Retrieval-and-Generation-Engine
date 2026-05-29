"""
RAGX Retrieval — Retrieval engine with search, reranking, and strategy modules.

Provides dense vector similarity search, BM25 sparse search, hybrid search
with Reciprocal Rank Fusion, cross-encoder and Cohere reranking, multi-query
retrieval, parent-child retrieval, context compression, retrieval evaluation
metrics, and a unified orchestration engine.
"""

from __future__ import annotations

from ragx.retrieval.engine import RetrievalEngine
from ragx.retrieval.metrics import (
    evaluate_batch,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ragx.retrieval.reranking.cohere_rerank import CohereReranker
from ragx.retrieval.reranking.cross_encoder import CrossEncoderReranker
from ragx.retrieval.search.bm25 import BM25Search
from ragx.retrieval.search.hybrid import HybridSearch
from ragx.retrieval.search.similarity import SimilaritySearch
from ragx.retrieval.strategies.compression import ContextCompressor
from ragx.retrieval.strategies.multi_query import MultiQueryRetriever
from ragx.retrieval.strategies.parent_child import ParentChildRetriever

__all__ = [
    "BM25Search",
    "CohereReranker",
    "ContextCompressor",
    "CrossEncoderReranker",
    "HybridSearch",
    "MultiQueryRetriever",
    "ParentChildRetriever",
    "RetrievalEngine",
    "SimilaritySearch",
    "evaluate_batch",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
