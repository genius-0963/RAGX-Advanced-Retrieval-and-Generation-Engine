"""
RAGX BGE Embeddings — BGE (BAAI General Embedding) model wrapper.

Wraps ``langchain_huggingface.HuggingFaceEmbeddings`` configured for
the BGE family of embedding models with L2-normalised output vectors
and automatic hardware acceleration.
"""

from __future__ import annotations

import platform
from typing import Any

from ragx.config.logging_config import get_logger

logger = get_logger(__name__)


def _detect_device() -> str:
    """
    Detect the best available compute device.

    Returns:
        ``'cuda'`` if NVIDIA GPU is available, ``'mps'`` on Apple Silicon,
        otherwise ``'cpu'``.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available() and platform.system() == "Darwin":
            return "mps"
    except ImportError:
        logger.debug("PyTorch not available — falling back to CPU")
    return "cpu"


class BGEEmbeddingModel:
    """
    BGE (BAAI General Embedding) model wrapper.

    Uses ``HuggingFaceEmbeddings`` with ``normalize_embeddings=True``
    to produce L2-normalised vectors suitable for cosine similarity
    and inner-product search.

    Attributes:
        model_name: HuggingFace model identifier for the BGE model.
        device: Compute device (``cuda``, ``mps``, or ``cpu``).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        batch_size: int = 256,
    ) -> None:
        """
        Initialize the BGE embedding model.

        Args:
            model_name: HuggingFace model identifier.
            device: Compute device override. Auto-detected if ``None``.
            batch_size: Number of texts to encode per forward pass.
        """
        from langchain_huggingface import HuggingFaceEmbeddings

        self.model_name = model_name
        self.device = device or _detect_device()
        self.batch_size = batch_size

        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": self.batch_size,
            },
        )
        logger.info(
            "Initialized BGEEmbeddingModel",
            model_name=self.model_name,
            device=self.device,
            batch_size=self.batch_size,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of document texts.

        Args:
            texts: List of document text strings to embed.

        Returns:
            List of L2-normalised embedding vectors.

        Raises:
            RuntimeError: If embedding fails.
        """
        if not texts:
            return []

        try:
            embeddings = self._embeddings.embed_documents(texts)
            logger.info(
                "BGE document embedding complete",
                total_texts=len(texts),
                embedding_dim=len(embeddings[0]) if embeddings else 0,
            )
            return embeddings
        except Exception as exc:
            logger.exception("BGE document embedding failed", total_texts=len(texts))
            raise RuntimeError("BGE document embedding failed") from exc

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.

        Args:
            text: Query text to embed.

        Returns:
            L2-normalised embedding vector for the query.

        Raises:
            RuntimeError: If embedding fails.
        """
        try:
            embedding = self._embeddings.embed_query(text)
            logger.debug("Embedded query with BGE", text_length=len(text))
            return embedding
        except Exception as exc:
            logger.exception("BGE query embedding failed")
            raise RuntimeError("BGE query embedding failed") from exc

    def get_langchain_embeddings(self) -> Any:
        """
        Return the underlying LangChain embeddings object.

        Returns:
            The ``langchain_huggingface.HuggingFaceEmbeddings`` instance.
        """
        return self._embeddings
