"""
RAGX Embedding Pipeline — Orchestrates chunking, embedding, and vector storage.

Processes documents through: chunk → embed → store, supporting incremental
updates and batch processing.
"""

from __future__ import annotations

from typing import Any

from tqdm import tqdm

from ragx.config.logging_config import get_logger
from ragx.config.settings import Settings, get_settings
from ragx.ingestion.loaders.base import Document

logger = get_logger(__name__)


class EmbeddingPipeline:
    """
    Orchestrates the embedding pipeline: chunk → embed → store.

    Supports incremental updates and batch processing with
    configurable chunker, embedding model, and vector store.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize embedding pipeline from settings.

        Args:
            settings: Application settings. Uses defaults if None.
        """
        self.settings = settings or get_settings()
        self._chunker = None
        self._embedding_model = None
        self._vectorstore = None
        self._processed_ids: set[str] = set()

    def _get_chunker(self):
        """Lazily initialize the text chunker."""
        if self._chunker is None:
            strategy = self.settings.chunking_strategy.value
            if strategy == "semantic":
                try:
                    from ragx.embeddings.chunking.semantic_chunker import SemanticChunkerWrapper
                    embeddings = self._get_embedding_model().get_langchain_embeddings()
                    self._chunker = SemanticChunkerWrapper(embeddings=embeddings)
                    logger.info("semantic_chunker_initialized")
                except Exception as e:
                    logger.warning("semantic_chunker_failed, falling back to recursive", error=str(e))
                    from ragx.embeddings.chunking.recursive_splitter import RecursiveChunker
                    self._chunker = RecursiveChunker(
                        chunk_size=self.settings.chunk_size,
                        chunk_overlap=self.settings.chunk_overlap,
                    )
            else:
                from ragx.embeddings.chunking.recursive_splitter import RecursiveChunker
                self._chunker = RecursiveChunker(
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                )
                logger.info("recursive_chunker_initialized")
        return self._chunker

    def _get_embedding_model(self):
        """Lazily initialize the embedding model."""
        if self._embedding_model is None:
            from ragx.embeddings.models import get_embedding_model
            self._embedding_model = get_embedding_model(
                provider=self.settings.embedding_provider.value,
                model_name=self.settings.embedding_model,
            )
            logger.info(
                "embedding_model_initialized",
                provider=self.settings.embedding_provider.value,
                model=self.settings.embedding_model,
            )
        return self._embedding_model

    def _get_vectorstore(self):
        """Lazily initialize the vector store."""
        if self._vectorstore is None:
            from ragx.embeddings.vectorstore import get_vectorstore
            embeddings = self._get_embedding_model().get_langchain_embeddings()
            self._vectorstore = get_vectorstore(
                store_type=self.settings.vectorstore_type.value,
                embeddings=embeddings,
                collection_name=self.settings.chroma_collection,
                persist_directory=str(
                    self.settings.vectorstore_path / self.settings.vectorstore_type.value
                ),
            )
            logger.info(
                "vectorstore_initialized",
                type=self.settings.vectorstore_type.value,
            )
        return self._vectorstore

    def process(
        self,
        documents: list[Document],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> list[str]:
        """
        Process documents: chunk → embed → store.

        Args:
            documents: List of documents to process.
            batch_size: Number of chunks to process at once.
            show_progress: Whether to show a progress bar.

        Returns:
            List of stored document/chunk IDs.
        """
        if not documents:
            logger.warning("no_documents_to_process")
            return []

        logger.info("embedding_pipeline_started", num_documents=len(documents))

        # Step 1: Chunk documents
        chunker = self._get_chunker()
        chunks = chunker.split(documents)
        logger.info("chunking_complete", num_chunks=len(chunks))

        vectorstore = self._get_vectorstore()
        all_ids: list[str] = []

        # Process in batches
        iterator = range(0, len(chunks), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding batches", unit="batch")

        for i in iterator:
            batch = chunks[i : i + batch_size]
            try:
                ids = vectorstore.add_documents(batch)
                all_ids.extend(ids)
            except Exception as e:
                logger.error("batch_embedding_failed", batch_start=i, error=str(e))

        # Track processed document IDs
        for doc in documents:
            doc_id = doc.metadata.get("document_id")
            if doc_id:
                self._processed_ids.add(doc_id)

        logger.info(
            "embedding_pipeline_complete",
            total_chunks=len(chunks),
            stored_ids=len(all_ids),
        )
        return all_ids

    def process_incremental(
        self,
        documents: list[Document],
        batch_size: int = 100,
    ) -> list[str]:
        """
        Process only new/unprocessed documents.

        Args:
            documents: List of documents (some may already be processed).
            batch_size: Batch size for processing.

        Returns:
            List of newly stored chunk IDs.
        """
        new_docs = [
            doc for doc in documents
            if doc.metadata.get("document_id") not in self._processed_ids
        ]

        if not new_docs:
            logger.info("no_new_documents_to_process")
            return []

        logger.info(
            "incremental_processing",
            total=len(documents),
            new=len(new_docs),
            skipped=len(documents) - len(new_docs),
        )
        return self.process(new_docs, batch_size=batch_size)

    def get_vectorstore(self) -> Any:
        """Get the underlying vector store instance."""
        return self._get_vectorstore()

    def get_retriever(self, **kwargs):
        """
        Get a LangChain-compatible retriever from the vector store.

        Args:
            **kwargs: Passed to the vector store's as_retriever method.

        Returns:
            LangChain retriever.
        """
        vs = self._get_vectorstore()
        if hasattr(vs, "as_retriever"):
            return vs.as_retriever(**kwargs)
        raise AttributeError("Vector store does not support as_retriever()")
