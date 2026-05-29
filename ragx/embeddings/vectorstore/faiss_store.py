"""
RAGX FAISS Vector Store — FAISS-backed document storage and retrieval.

Wraps ``langchain_community.vectorstores.FAISS`` to provide persistent,
high-performance approximate nearest-neighbour search using
``IndexFlatIP`` (inner-product) for small-to-medium datasets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ragx.config.logging_config import get_logger
from ragx.embeddings.vectorstore.base import BaseVectorStore
from ragx.ingestion.loaders.base import Document

logger = get_logger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS vector store implementation.

    Stores document embeddings in a FAISS ``IndexFlatIP`` index and
    persists the index to disk for later reloading.

    Attributes:
        persist_directory: Directory for saving/loading the FAISS index.
    """

    def __init__(
        self,
        embeddings: Any,
        persist_directory: str = "./data/vectorstore/faiss",
    ) -> None:
        """
        Initialize the FAISS vector store.

        If a previously saved index exists at ``persist_directory``,
        it is loaded automatically.

        Args:
            embeddings: A LangChain-compatible embeddings object.
            persist_directory: Path to the directory used for persistence.
        """
        from langchain_community.vectorstores import FAISS

        self._embeddings = embeddings
        self.persist_directory = str(Path(persist_directory).resolve())
        self._faiss_class = FAISS
        self._store: FAISS | None = None

        # Ensure the persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Attempt to load an existing index
        index_path = Path(self.persist_directory) / "index.faiss"
        if index_path.exists():
            try:
                self.load()
                logger.info(
                    "Loaded existing FAISS index",
                    persist_directory=self.persist_directory,
                    count=self.count(),
                )
            except Exception:
                logger.exception(
                    "Failed to load existing FAISS index — starting fresh"
                )
                self._store = None
        else:
            logger.info(
                "No existing FAISS index found — will create on first add",
                persist_directory=self.persist_directory,
            )

    # ── CRUD ────────────────────────────────────────────────────────────────

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to the FAISS index.

        On the very first call (when the index is empty), a new FAISS
        store is created from the documents. Subsequent calls merge
        new documents into the existing index.

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
            if self._store is None:
                # Create new FAISS store from the first batch
                self._store = self._faiss_class.from_documents(
                    documents=lc_docs,
                    embedding=self._embeddings,
                    ids=ids,
                )
            else:
                self._store.add_documents(documents=lc_docs, ids=ids)

            # Auto-persist after every add
            self.save()

            logger.info(
                "Added documents to FAISS",
                added=len(documents),
                total=self.count(),
            )
            return ids

        except Exception as exc:
            logger.exception("Failed to add documents to FAISS")
            raise RuntimeError("FAISS add_documents failed") from exc

    def delete_documents(self, ids: list[str]) -> bool:
        """
        Delete documents from the FAISS index by ID.

        Args:
            ids: Document IDs to remove.

        Returns:
            ``True`` if deletion succeeded.
        """
        if self._store is None:
            logger.warning("FAISS store is empty — nothing to delete")
            return False

        try:
            deleted = self._store.delete(ids)
            self.save()
            logger.info("Deleted documents from FAISS", count=len(ids))
            return deleted if isinstance(deleted, bool) else True
        except Exception as exc:
            logger.exception("Failed to delete documents from FAISS")
            raise RuntimeError("FAISS delete_documents failed") from exc

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
        logger.info("Updated documents in FAISS", count=len(ids))
        return True

    # ── Search ──────────────────────────────────────────────────────────────

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """
        Find the *k* most similar documents.

        Args:
            query: Natural-language query.
            k: Number of results.

        Returns:
            List of RAGX Documents ranked by similarity.
        """
        if self._store is None:
            logger.warning("FAISS store is empty — returning no results")
            return []

        try:
            lc_docs = self._store.similarity_search(query, k=k)
            results = [Document.from_langchain(d) for d in lc_docs]
            logger.debug("FAISS similarity search", query_len=len(query), k=k, results=len(results))
            return results
        except Exception as exc:
            logger.exception("FAISS similarity search failed")
            raise RuntimeError("FAISS similarity_search failed") from exc

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        """
        Find the *k* most similar documents with scores.

        Args:
            query: Natural-language query.
            k: Number of results.

        Returns:
            List of ``(Document, score)`` tuples.
        """
        if self._store is None:
            logger.warning("FAISS store is empty — returning no results")
            return []

        try:
            results_with_score = self._store.similarity_search_with_score(query, k=k)
            converted = [
                (Document.from_langchain(lc_doc), score)
                for lc_doc, score in results_with_score
            ]
            logger.debug(
                "FAISS similarity search with score",
                query_len=len(query),
                k=k,
                results=len(converted),
            )
            return converted
        except Exception as exc:
            logger.exception("FAISS similarity search with score failed")
            raise RuntimeError("FAISS similarity_search_with_score failed") from exc

    # ── Retriever ───────────────────────────────────────────────────────────

    def as_retriever(self, **kwargs: Any) -> Any:
        """
        Return a LangChain retriever backed by this FAISS store.

        Args:
            **kwargs: Forwarded to ``FAISS.as_retriever()``.
                Common keys: ``search_type``, ``search_kwargs``.

        Returns:
            A LangChain ``VectorStoreRetriever``.

        Raises:
            RuntimeError: If the store is uninitialised.
        """
        if self._store is None:
            raise RuntimeError(
                "Cannot create retriever — FAISS store has no documents. "
                "Add documents first."
            )
        return self._store.as_retriever(**kwargs)

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the FAISS index to ``self.persist_directory``."""
        if self._store is None:
            logger.warning("FAISS store is empty — nothing to save")
            return

        try:
            self._store.save_local(self.persist_directory)
            logger.info(
                "Saved FAISS index",
                persist_directory=self.persist_directory,
            )
        except Exception as exc:
            logger.exception("Failed to save FAISS index")
            raise RuntimeError("FAISS save failed") from exc

    def load(self) -> None:
        """Load a FAISS index from ``self.persist_directory``."""
        try:
            self._store = self._faiss_class.load_local(
                self.persist_directory,
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(
                "Loaded FAISS index",
                persist_directory=self.persist_directory,
            )
        except Exception as exc:
            logger.exception("Failed to load FAISS index")
            raise RuntimeError("FAISS load failed") from exc

    # ── Utilities ───────────────────────────────────────────────────────────

    def count(self) -> int:
        """
        Return the number of vectors in the FAISS index.

        Returns:
            Document count, or ``0`` if the store is uninitialised.
        """
        if self._store is None:
            return 0
        try:
            return self._store.index.ntotal
        except AttributeError:
            # Fallback: try docstore
            try:
                return len(self._store.docstore._dict)
            except Exception:
                return 0
