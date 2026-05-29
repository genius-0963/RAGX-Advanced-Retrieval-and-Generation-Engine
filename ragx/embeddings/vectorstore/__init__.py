"""
RAGX Vector Store — Abstract base and concrete vector store implementations.

Provides a factory function and concrete stores (FAISS, ChromaDB)
for persisting and querying document embeddings.
"""

from __future__ import annotations

from typing import Any

from ragx.config.logging_config import get_logger
from ragx.embeddings.vectorstore.base import BaseVectorStore
from ragx.embeddings.vectorstore.chroma_store import ChromaVectorStore
from ragx.embeddings.vectorstore.faiss_store import FAISSVectorStore

logger = get_logger(__name__)

__all__ = [
    "BaseVectorStore",
    "FAISSVectorStore",
    "ChromaVectorStore",
    "get_vectorstore",
]


def get_vectorstore(
    store_type: str,
    embeddings: Any,
    **kwargs: Any,
) -> BaseVectorStore:
    """
    Factory function to create a vector store by type.

    Args:
        store_type: Vector store backend. One of ``'chroma'`` or ``'faiss'``.
        embeddings: A LangChain-compatible embeddings object used for
            encoding queries and documents.
        **kwargs: Additional keyword arguments forwarded to the store
            constructor (e.g. ``persist_directory``, ``collection_name``).

    Returns:
        An initialised :class:`BaseVectorStore` subclass instance.

    Raises:
        ValueError: If ``store_type`` is not recognised.
    """
    store_type_lower = store_type.lower().strip()

    if store_type_lower == "faiss":
        store = FAISSVectorStore(embeddings=embeddings, **kwargs)
        logger.info("Created FAISS vector store", **kwargs)
        return store

    if store_type_lower == "chroma":
        store = ChromaVectorStore(embeddings=embeddings, **kwargs)
        logger.info("Created ChromaDB vector store", **kwargs)
        return store

    raise ValueError(
        f"Unknown vector store type '{store_type}'. "
        f"Supported: 'faiss', 'chroma'."
    )
