"""
RAGX ChromaDB Vector Store — ChromaDB-backed document storage and retrieval.

Wraps ``langchain_chroma.Chroma`` to provide persistent vector storage
with collection management and metadata-filtering support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragx.config.logging_config import get_logger
from ragx.embeddings.vectorstore.base import BaseVectorStore
from ragx.ingestion.loaders.base import Document

logger = get_logger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB vector store implementation.

    Stores document embeddings in a named ChromaDB collection with
    automatic persistence to disk.

    Attributes:
        collection_name: Name of the ChromaDB collection.
        persist_directory: Directory for on-disk persistence.
    """

    def __init__(
        self,
        embeddings: Any,
        collection_name: str = "ragx_default",
        persist_directory: str = "./data/vectorstore/chroma",
    ) -> None:
        """
        Initialize the ChromaDB vector store.

        Creates (or opens) a named collection backed by a persistent
        ChromaDB database on disk.

        Args:
            embeddings: A LangChain-compatible embeddings object.
            collection_name: Name of the ChromaDB collection to use.
            persist_directory: Path to the directory for persistence.
        """
        from langchain_chroma import Chroma

        self._embeddings = embeddings
        self.collection_name = collection_name
        self.persist_directory = str(Path(persist_directory).resolve())

        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self._store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=self.persist_directory,
        )
        logger.info(
            "Initialized ChromaVectorStore",
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            count=self.count(),
        )

    # ── CRUD ────────────────────────────────────────────────────────────────

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to the ChromaDB collection.

        Args:
            documents: RAGX Documents to ingest.

        Returns:
            List of document IDs.
        """
        if not documents:
            logger.warning("No documents to add")
            return []

        lc_docs = [doc.to_langchain() for doc in documents]
        ids = [doc.doc_id for doc in documents]

        try:
            returned_ids = self._store.add_documents(documents=lc_docs, ids=ids)
            logger.info(
                "Added documents to ChromaDB",
                added=len(documents),
                collection=self.collection_name,
                total=self.count(),
            )
            return returned_ids
        except Exception as exc:
            logger.exception("Failed to add documents to ChromaDB")
            raise RuntimeError("ChromaDB add_documents failed") from exc

    def delete_documents(self, ids: list[str]) -> bool:
        """
        Delete documents from the ChromaDB collection by ID.

        Args:
            ids: Document IDs to remove.

        Returns:
            ``True`` if deletion succeeded.
        """
        if not ids:
            return True

        try:
            self._store.delete(ids=ids)
            logger.info(
                "Deleted documents from ChromaDB",
                count=len(ids),
                collection=self.collection_name,
            )
            return True
        except Exception as exc:
            logger.exception("Failed to delete documents from ChromaDB")
            raise RuntimeError("ChromaDB delete_documents failed") from exc

    def update_documents(self, ids: list[str], documents: list[Document]) -> bool:
        """
        Update documents by deleting and re-adding them.

        Args:
            ids: IDs of documents to replace.
            documents: Replacement RAGX Documents.

        Returns:
            ``True`` if the update succeeded.
        """
        if len(ids) != len(documents):
            raise ValueError(
                f"Length mismatch: {len(ids)} IDs vs {len(documents)} documents"
            )

        self.delete_documents(ids)

        # Override doc_ids to match the original IDs
        for doc, doc_id in zip(documents, ids):
            doc.doc_id = doc_id

        self.add_documents(documents)
        logger.info(
            "Updated documents in ChromaDB",
            count=len(ids),
            collection=self.collection_name,
        )
        return True

    # ── Search ──────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Find the *k* most similar documents.

        Args:
            query: Natural-language query.
            k: Number of results.
            filter: Optional ChromaDB metadata filter dict.

        Returns:
            List of RAGX Documents ranked by similarity.
        """
        try:
            kwargs: dict[str, Any] = {"k": k}
            if filter is not None:
                kwargs["filter"] = filter

            lc_docs = self._store.similarity_search(query, **kwargs)
            results = [Document.from_langchain(d) for d in lc_docs]
            logger.debug(
                "ChromaDB similarity search",
                query_len=len(query),
                k=k,
                results=len(results),
            )
            return results
        except Exception as exc:
            logger.exception("ChromaDB similarity search failed")
            raise RuntimeError("ChromaDB similarity_search failed") from exc

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Find the *k* most similar documents with distance scores.

        Args:
            query: Natural-language query.
            k: Number of results.
            filter: Optional ChromaDB metadata filter dict.

        Returns:
            List of ``(Document, score)`` tuples.
        """
        try:
            kwargs: dict[str, Any] = {"k": k}
            if filter is not None:
                kwargs["filter"] = filter

            results_with_score = self._store.similarity_search_with_score(
                query, **kwargs
            )
            converted = [
                (Document.from_langchain(lc_doc), score)
                for lc_doc, score in results_with_score
            ]
            logger.debug(
                "ChromaDB similarity search with score",
                query_len=len(query),
                k=k,
                results=len(converted),
            )
            return converted
        except Exception as exc:
            logger.exception("ChromaDB similarity search with score failed")
            raise RuntimeError(
                "ChromaDB similarity_search_with_score failed"
            ) from exc

    # ── Retriever ───────────────────────────────────────────────────────────

    def as_retriever(self, **kwargs: Any) -> Any:
        """
        Return a LangChain retriever backed by this ChromaDB store.

        Args:
            **kwargs: Forwarded to ``Chroma.as_retriever()``.
                Common keys: ``search_type``, ``search_kwargs``.

        Returns:
            A LangChain ``VectorStoreRetriever``.
        """
        return self._store.as_retriever(**kwargs)

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self) -> None:
        """
        Persist the ChromaDB collection to disk.

        ChromaDB with a ``persist_directory`` auto-persists, so this
        method is a no-op kept for interface compliance.
        """
        # ChromaDB auto-persists when a persist_directory is configured.
        logger.debug(
            "ChromaDB auto-persists — explicit save is a no-op",
            collection=self.collection_name,
        )

    def load(self) -> None:
        """
        Reload the ChromaDB collection from disk.

        Re-initialises the Chroma client pointing at the same
        ``persist_directory`` and ``collection_name``.
        """
        from langchain_chroma import Chroma

        try:
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._embeddings,
                persist_directory=self.persist_directory,
            )
            logger.info(
                "Reloaded ChromaDB collection",
                collection=self.collection_name,
                persist_directory=self.persist_directory,
                count=self.count(),
            )
        except Exception as exc:
            logger.exception("Failed to reload ChromaDB collection")
            raise RuntimeError("ChromaDB load failed") from exc

    # ── Utilities ───────────────────────────────────────────────────────────

    def count(self) -> int:
        """
        Return the number of documents in the collection.

        Returns:
            Document count.
        """
        try:
            collection = self._store._collection
            return collection.count()
        except Exception:
            logger.debug("Could not get ChromaDB count — returning 0")
            return 0

    # ── Collection Management ───────────────────────────────────────────────

    def list_collections(self) -> list[str]:
        """
        List all collection names in the ChromaDB instance.

        Returns:
            List of collection name strings.
        """
        try:
            client = self._store._client
            collections = client.list_collections()
            names = [c.name for c in collections]
            logger.debug("Listed ChromaDB collections", count=len(names))
            return names
        except Exception:
            logger.exception("Failed to list ChromaDB collections")
            return []

    def delete_collection(self, collection_name: str | None = None) -> bool:
        """
        Delete a ChromaDB collection.

        Args:
            collection_name: Name of the collection to delete. Defaults
                to the current collection.

        Returns:
            ``True`` if deletion succeeded.
        """
        target = collection_name or self.collection_name
        try:
            client = self._store._client
            client.delete_collection(name=target)
            logger.info("Deleted ChromaDB collection", collection=target)

            # If we deleted our own collection, reinitialise
            if target == self.collection_name:
                self.load()

            return True
        except Exception as exc:
            logger.exception(
                "Failed to delete ChromaDB collection",
                collection=target,
            )
            raise RuntimeError(
                f"ChromaDB delete_collection('{target}') failed"
            ) from exc

    def create_collection(self, collection_name: str) -> None:
        """
        Create a new ChromaDB collection and switch to it.

        Args:
            collection_name: Name for the new collection.
        """
        from langchain_chroma import Chroma

        try:
            self.collection_name = collection_name
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._embeddings,
                persist_directory=self.persist_directory,
            )
            logger.info(
                "Created and switched to ChromaDB collection",
                collection=self.collection_name,
            )
        except Exception as exc:
            logger.exception(
                "Failed to create ChromaDB collection",
                collection=collection_name,
            )
            raise RuntimeError(
                f"ChromaDB create_collection('{collection_name}') failed"
            ) from exc
