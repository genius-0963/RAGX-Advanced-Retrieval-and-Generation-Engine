"""
RAGX Reranking — Document reranking with cross-encoder and Cohere models.

Provides CrossEncoderReranker (local) and CohereReranker (API-based)
for improving retrieval precision by rescoring candidate documents.
"""

from __future__ import annotations

from ragx.retrieval.reranking.cohere_rerank import CohereReranker
from ragx.retrieval.reranking.cross_encoder import CrossEncoderReranker

__all__ = ["CohereReranker", "CrossEncoderReranker"]
