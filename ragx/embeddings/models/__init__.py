"""
RAGX Embedding Models — Embedding provider wrappers and factory.

Provides a unified interface to multiple embedding providers
(OpenAI, BGE, Sentence Transformers) via a factory function.
"""

from __future__ import annotations

from typing import Any

from ragx.config.logging_config import get_logger
from ragx.embeddings.models.bge_embeddings import BGEEmbeddingModel
from ragx.embeddings.models.openai_embeddings import OpenAIEmbeddingModel
from ragx.embeddings.models.sentence_transformer import SentenceTransformerModel

logger = get_logger(__name__)

__all__ = [
    "OpenAIEmbeddingModel",
    "BGEEmbeddingModel",
    "SentenceTransformerModel",
    "get_embedding_model",
]


def get_embedding_model(
    provider: str,
    model_name: str | None = None,
) -> OpenAIEmbeddingModel | BGEEmbeddingModel | SentenceTransformerModel:
    """
    Factory function to create an embedding model by provider name.

    Args:
        provider: Embedding provider identifier. One of
            ``'openai'``, ``'bge'``, or ``'sentence-transformer'``.
        model_name: Optional model name override. If ``None``, each
            provider uses its default model.

    Returns:
        An initialised embedding model instance.

    Raises:
        ValueError: If the provider is not recognised.
    """
    provider_lower = provider.lower().strip()

    if provider_lower == "openai":
        kwargs: dict[str, Any] = {}
        if model_name:
            kwargs["model"] = model_name
        model = OpenAIEmbeddingModel(**kwargs)
        logger.info("Created OpenAI embedding model", model=model_name or "text-embedding-3-small")
        return model

    if provider_lower == "bge":
        kwargs = {}
        if model_name:
            kwargs["model_name"] = model_name
        model = BGEEmbeddingModel(**kwargs)
        logger.info("Created BGE embedding model", model=model_name or "BAAI/bge-small-en-v1.5")
        return model

    if provider_lower in ("sentence-transformer", "sentence_transformer"):
        kwargs = {}
        if model_name:
            kwargs["model_name"] = model_name
        model = SentenceTransformerModel(**kwargs)
        logger.info(
            "Created SentenceTransformer embedding model",
            model=model_name or "all-MiniLM-L6-v2",
        )
        return model

    raise ValueError(
        f"Unknown embedding provider '{provider}'. "
        f"Supported: 'openai', 'bge', 'sentence-transformer'."
    )
