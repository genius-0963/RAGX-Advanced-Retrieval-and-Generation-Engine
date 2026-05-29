"""
RAGX Base Vector Store — Abstract interface for vector store backends.

Defines the contract that all vector store implementations must fulfil,
including CRUD operations, similarity search, persistence, and
LangChain retriever integration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ragx.ingestion.loaders.base import Document


class BaseVectorStore(ABC):
    """
    Abstract base class for vector store implementations.

    Subclasses must implement every abstract method to provide a
    complete document storage and retrieval backend.
    """

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of RAGX Documents to ingest.

        Returns:
            List of assigned document IDs.
        """

    @abstractmethod
    def delete_documents(self, ids: list[str]) -> bool:
        """
        Delete documents by their IDs.

        Args:
            ids: List of document IDs to remove.

        Returns:
            ``True`` if deletion succeeded.
        """

    @abstractmethod
    def update_documents(self, ids: list[str], documents: list[Document]) -> bool:
        """
        Update existing documents in-place.

        Semantically equivalent to deleting the old documents and
        adding the new ones.

        Args:
            ids: IDs of documents to replace.
            documents: Replacement documents (same order as ``ids``).

        Returns:
            ``True`` if the update succeeded.
        """

    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """
        Find the *k* most similar documents to a query.

        Args:
            query: Natural-language query string.
            k: Number of results to return.

        Returns:
            List of RAGX Documents ranked by similarity (most similar first).
        """

    @abstractmethod
    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        """
        Find the *k* most similar documents together with their scores.

        Args:
            query: Natural-language query string.
            k: Number of results to return.

        Returns:
            List of ``(Document, score)`` tuples ranked by similarity.
        """

    @abstractmethod
    def as_retriever(self, **kwargs: Any) -> Any:
        """
        Return a LangChain-compatible retriever backed by this store.

        Args:
            **kwargs: Retriever configuration (e.g. ``search_type``,
                ``search_kwargs``).

        Returns:
            A LangChain ``BaseRetriever`` instance.
        """

    @abstractmethod
    def save(self) -> None:
        """Persist the vector store to disk."""

    @abstractmethod
    def load(self) -> None:
        """Load a previously persisted vector store from disk."""

    @abstractmethod
    def count(self) -> int:
        """
        Return the number of documents in the store.

        Returns:
            Document count.
        """
