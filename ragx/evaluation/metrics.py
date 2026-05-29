"""
RAGX Custom Evaluation Metrics — Lightweight metrics that don't require external APIs.

Provides token-overlap similarity, context utilization, and response completeness.
"""

from __future__ import annotations

import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def calculate_answer_similarity(answer: str, reference: str) -> float:
    """
    Calculate token-overlap similarity between answer and reference.

    Uses F1-score of token overlap (similar to BLEU-1 / ROUGE-1 concept).

    Args:
        answer: Generated answer text.
        reference: Reference/gold answer text.

    Returns:
        Float score between 0.0 and 1.0.
    """
    if not answer or not reference:
        return 0.0

    answer_tokens = Counter(_tokenize(answer))
    reference_tokens = Counter(_tokenize(reference))

    common = answer_tokens & reference_tokens
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / max(sum(answer_tokens.values()), 1)
    recall = num_common / max(sum(reference_tokens.values()), 1)

    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return round(f1, 4)


def calculate_context_utilization(answer: str, contexts: list[str]) -> float:
    """
    Calculate how much of the retrieved context was used in the answer.

    Measures what fraction of context tokens appear in the answer.

    Args:
        answer: Generated answer text.
        contexts: List of context strings.

    Returns:
        Float score between 0.0 and 1.0.
    """
    if not answer or not contexts:
        return 0.0

    answer_tokens = set(_tokenize(answer))
    all_context_tokens: list[str] = []
    for ctx in contexts:
        all_context_tokens.extend(_tokenize(ctx))

    if not all_context_tokens:
        return 0.0

    context_token_set = set(all_context_tokens)

    # Remove common stopwords from consideration
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "nor", "not", "so", "yet", "both", "either", "neither", "each",
        "every", "all", "any", "few", "more", "most", "other", "some", "such",
        "no", "only", "own", "same", "than", "too", "very", "just", "because",
        "it", "its", "this", "that", "these", "those", "i", "me", "my",
        "we", "our", "you", "your", "he", "him", "his", "she", "her",
        "they", "them", "their", "what", "which", "who", "whom", "how",
        "when", "where", "why",
    }
    context_content_tokens = context_token_set - stopwords

    if not context_content_tokens:
        return 0.0

    used_tokens = answer_tokens & context_content_tokens
    utilization = len(used_tokens) / len(context_content_tokens)

    return round(min(utilization, 1.0), 4)


def calculate_response_completeness(answer: str, query: str) -> float:
    """
    Estimate how completely the answer addresses the query.

    Uses query keyword coverage as a proxy for completeness.

    Args:
        answer: Generated answer text.
        query: The original user query.

    Returns:
        Float score between 0.0 and 1.0.
    """
    if not answer or not query:
        return 0.0

    query_tokens = set(_tokenize(query))
    answer_tokens = set(_tokenize(answer))

    # Remove common question words and stopwords
    question_words = {
        "what", "which", "who", "whom", "where", "when", "why", "how",
        "is", "are", "was", "were", "do", "does", "did", "can", "could",
        "will", "would", "should", "the", "a", "an", "of", "in", "to",
        "for", "and", "or", "me", "my", "i", "please", "tell", "explain",
        "describe", "list", "give", "show",
    }
    query_content_tokens = query_tokens - question_words

    if not query_content_tokens:
        # If query is all question words, check if answer is non-trivial
        return 1.0 if len(answer_tokens) > 5 else 0.5

    covered = query_content_tokens & answer_tokens
    coverage = len(covered) / len(query_content_tokens)

    # Bonus for answer length (longer answers tend to be more complete)
    length_bonus = min(len(answer_tokens) / 50, 0.2)  # cap at 0.2

    completeness = min(coverage + length_bonus, 1.0)
    return round(completeness, 4)
