"""
RAGX Parent-Child Document Retrieval — Index small chunks, return parent chunks.

Indexes small child chunks for high retrieval precision, but returns the
larger parent chunks to provide complete context to the LLM.
"""

from __future__ import annotations

import uuid
from typing import Any

from ragx.config.logging_config import get_logger
from ragx.ingestion.loaders.base import Document

logger = get_logger(__name__)


class ParentChildRetriever:
    """
    Parent-child document retriever.

    Indexes small child chunks for precise matching but returns
    the larger parent chunks for richer context.
    """

    def __init__(
        self,
        vectorstore: Any,
        parent_splitter: Any,
        child_splitter: Any,
        docstore: dict[str, Document] | None = None,
    ) -> None:
        """
        Initialize parent-child retriever.

        Args:
            vectorstore: Vector store for indexing child chunks.
            parent_splitter: Text splitter for creating parent chunks.
            child_splitter: Text splitter for creating child chunks (smaller).
            docstore: Optional dict mapping parent_id -> parent Document.
                     Uses in-memory dict if None.
        """
        self.vectorstore = vectorstore
        self.parent_splitter = parent_splitter
        self.child_splitter = child_splitter
        self.docstore: dict[str, Document] = docstore if docstore is not None else {}

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Index documents using parent-child strategy.

        Creates parent chunks, then splits each parent into child chunks.
        Child chunks are indexed in the vector store with parent_id metadata.
        Parent chunks are stored in the docstore.

        Args:
            documents: List of documents to index.

        Returns:
            List of parent document IDs.
        """
        from langchain_core.documents import Document as LCDocument

        parent_ids: list[str] = []
        child_lc_docs: list[LCDocument] = []

        for doc in documents:
            # Create parent chunks
            parent_lc_doc = LCDocument(
                page_content=doc.content,
                metadata=doc.metadata.copy(),
            )
            parent_chunks = self.parent_splitter.split_documents([parent_lc_doc])

            for parent_chunk in parent_chunks:
                parent_id = str(uuid.uuid4())
                parent_ids.append(parent_id)

                # Store parent in docstore
                self.docstore[parent_id] = Document(
                    content=parent_chunk.page_content,
                    metadata={**parent_chunk.metadata, "parent_id": parent_id},
                    source_path=doc.source_path,
                )

                # Create child chunks from this parent
                child_chunks = self.child_splitter.split_documents([parent_chunk])
                for child_chunk in child_chunks:
                    child_chunk.metadata["parent_id"] = parent_id
                    child_chunk.metadata["source_path"] = doc.source_path
                    child_lc_docs.append(child_chunk)

        # Index all child chunks in vector store
        if child_lc_docs:
            if hasattr(self.vectorstore, "add_documents"):
                self.vectorstore.add_documents(child_lc_docs)
            else:
                logger.warning("vectorstore_missing_add_documents")

        logger.info(
            "parent_child_indexed",
            num_parents=len(parent_ids),
            num_children=len(child_lc_docs),
        )
        return parent_ids

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        """
        Retrieve parent documents by searching child chunks.

        Searches child chunks for the query, then returns the
        corresponding parent chunks (deduplicated).

        Args:
            query: Search query.
            k: Number of parent documents to return.

        Returns:
            List of parent Documents.
        """
        # Retrieve more children than k to ensure enough unique parents
        fetch_k = k * 3

        if hasattr(self.vectorstore, "similarity_search"):
            child_results = self.vectorstore.similarity_search(query, k=fetch_k)
        else:
            child_results = []

        # Map children back to parents (deduplicate)
        seen_parent_ids: set[str] = set()
        parent_docs: list[Document] = []

        for child in child_results:
            parent_id = child.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                parent_doc = self.docstore.get(parent_id)
                if parent_doc is not None:
                    parent_docs.append(parent_doc)

                if len(parent_docs) >= k:
                    break

        logger.info(
            "parent_child_retrieved",
            query_preview=query[:80],
            num_children_fetched=len(child_results),
            num_parents_returned=len(parent_docs),
        )
        return parent_docs

    def get_parent_count(self) -> int:
        """Return number of parent documents in the docstore."""
        return len(self.docstore)
