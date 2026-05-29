"""
RAGX OpenAI Embeddings — OpenAI embedding model wrapper.

Wraps ``langchain_openai.OpenAIEmbeddings`` to provide batch embedding
with configurable batch sizes and a consistent interface for the
RAGX embedding pipeline.
"""

from __future__ import annotations

from typing import Any

from ragx.config.logging_config import get_logger
from ragx.config.settings import get_settings

logger = get_logger(__name__)


class OpenAIEmbeddingModel:
    """
    OpenAI embedding model wrapper.

    Uses the OpenAI Embeddings API (via LangChain) to generate dense
    vector representations of text.

    Attributes:
        model: The OpenAI model identifier (e.g. ``text-embedding-3-small``).
        batch_size: Number of texts to embed in a single API call.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 512,
    ) -> None:
        """
        Initialize the OpenAI embedding model.

        Args:
            model: OpenAI embedding model name.
            api_key: OpenAI API key. Falls back to ``Settings.openai_api_key``
                or the ``OPENAI_API_KEY`` environment variable.
            batch_size: Maximum texts per embedding API call.

        Raises:
            ValueError: If no API key is available.
        """
        from langchain_openai import OpenAIEmbeddings

        self.model = model
        self.batch_size = batch_size

        resolved_key = api_key or get_settings().openai_api_key
        if not resolved_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY env var "
                "or pass api_key explicitly."
            )

        self._embeddings = OpenAIEmbeddings(
            model=self.model,
            openai_api_key=resolved_key,
            chunk_size=self.batch_size,
        )
        logger.info(
            "Initialized OpenAIEmbeddingModel",
            model=self.model,
            batch_size=self.batch_size,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of document texts.

        Texts are processed in batches of ``self.batch_size`` for
        efficient API usage.

        Args:
            texts: List of document text strings to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            RuntimeError: If the OpenAI API call fails.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(texts))
            batch = texts[start:end]

            try:
                batch_embeddings = self._embeddings.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                logger.debug(
                    "Embedded batch",
                    batch=f"{batch_idx + 1}/{total_batches}",
                    batch_size=len(batch),
                )
            except Exception as exc:
                logger.exception(
                    "OpenAI embedding batch failed",
                    batch_index=batch_idx,
                    batch_size=len(batch),
                )
                raise RuntimeError(
                    f"OpenAI embedding failed on batch {batch_idx + 1}/{total_batches}"
                ) from exc

        logger.info(
            "Document embedding complete",
            total_texts=len(texts),
            total_batches=total_batches,
            embedding_dim=len(all_embeddings[0]) if all_embeddings else 0,
        )
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.

        Uses the query-optimized embedding endpoint when available.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector for the query.

        Raises:
            RuntimeError: If the OpenAI API call fails.
        """
        try:
            embedding = self._embeddings.embed_query(text)
            logger.debug("Embedded query", text_length=len(text))
            return embedding
        except Exception as exc:
            logger.exception("OpenAI query embedding failed")
            raise RuntimeError("OpenAI query embedding failed") from exc

    def get_langchain_embeddings(self) -> Any:
        """
        Return the underlying LangChain embeddings object.

        Returns:
            The ``langchain_openai.OpenAIEmbeddings`` instance used internally.
        """
        return self._embeddings
