"""
RAGX Retrieval Metrics — Precision@K, Recall@K, MRR, and NDCG@K.

Standard information retrieval evaluation metrics for measuring
retrieval quality against gold-standard relevance judgments.
"""

from __future__ import annotations

import math

from ragx.config.logging_config import get_logger

logger = get_logger(__name__)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Calculate Precision@K — fraction of top-k retrieved docs that are relevant.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of relevant document IDs (ground truth).
        k: Cutoff position.

    Returns:
        Precision score between 0.0 and 1.0.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant)
    return relevant_in_top_k / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Calculate Recall@K — fraction of relevant docs found in top-k results.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of relevant document IDs (ground truth).
        k: Cutoff position.

    Returns:
        Recall score between 0.0 and 1.0.
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant)
    return relevant_in_top_k / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """
    Calculate Mean Reciprocal Rank — inverse of the rank of the first relevant result.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of relevant document IDs (ground truth).

    Returns:
        MRR score between 0.0 and 1.0.
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    Uses binary relevance (1 if relevant, 0 otherwise).

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of relevant document IDs (ground truth).
        k: Cutoff position.

    Returns:
        NDCG score between 0.0 and 1.0.
    """
    if k <= 0 or not relevant:
        return 0.0

    top_k = retrieved[:k]

    # DCG: sum of relevance / log2(rank + 1)
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        rel = 1.0 if doc_id in relevant else 0.0
        dcg += rel / math.log2(i + 2)  # +2 because ranks start at 1 and log2(1)=0

    # Ideal DCG: all relevant docs at top positions
    ideal_k = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_batch(queries: list[dict]) -> dict:
    """
    Evaluate multiple queries and return aggregate metrics.

    Args:
        queries: List of dicts, each with:
            - retrieved (list[str]): Retrieved document IDs.
            - relevant (set[str] or list[str]): Relevant document IDs.
            - k (int, optional): Cutoff position. Defaults to 5.

    Returns:
        Dict with per-query and aggregate metrics.
    """
    per_query: list[dict] = []
    totals = {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}

    for i, q in enumerate(queries):
        retrieved_list = q["retrieved"]
        relevant_set = set(q["relevant"]) if isinstance(q["relevant"], list) else q["relevant"]
        k = q.get("k", 5)

        p = precision_at_k(retrieved_list, relevant_set, k)
        r = recall_at_k(retrieved_list, relevant_set, k)
        m = mrr(retrieved_list, relevant_set)
        n = ndcg_at_k(retrieved_list, relevant_set, k)

        per_query.append({
            "query_index": i,
            "precision_at_k": round(p, 4),
            "recall_at_k": round(r, 4),
            "mrr": round(m, 4),
            "ndcg_at_k": round(n, 4),
            "k": k,
        })

        totals["precision"] += p
        totals["recall"] += r
        totals["mrr"] += m
        totals["ndcg"] += n

    num_queries = len(queries) if queries else 1
    aggregate = {
        "num_queries": len(queries),
        "avg_precision_at_k": round(totals["precision"] / num_queries, 4),
        "avg_recall_at_k": round(totals["recall"] / num_queries, 4),
        "avg_mrr": round(totals["mrr"] / num_queries, 4),
        "avg_ndcg_at_k": round(totals["ndcg"] / num_queries, 4),
    }

    logger.info("batch_evaluation_complete", **aggregate)
    return {"aggregate": aggregate, "per_query": per_query}
