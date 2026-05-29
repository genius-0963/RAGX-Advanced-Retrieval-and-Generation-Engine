"""
RAGX Context Compression — Extract only query-relevant content from retrieved chunks.

Supports LLM-based extraction and embeddings-based filtering to reduce
token usage and improve answer quality.
"""

from __future__ import annotations

from typing import Any

from ragx.config.logging_config import get_logger
from ragx.ingestion.loaders.base import Document

logger = get_logger(__name__)

COMPRESSION_PROMPT = """Given the following question and context, extract only the sentences 
from the context that are directly relevant to answering the question. 
Return only the relevant sentences, preserving their original wording.
If no sentences are relevant, respond with "No relevant information found."

Question: {question}

Context:
{context}

Relevant sentences:"""


class ContextCompressor:
    """
    Compresses retrieved context to include only query-relevant information.

    Two modes:
    - 'llm': Uses an LLM to extract relevant sentences from each chunk.
    - 'embeddings': Uses embedding similarity to filter chunks below a threshold.
    """

    def __init__(
        self,
        retriever: Any,
        llm: Any = None,
        embeddings: Any = None,
        method: str = "llm",
        similarity_threshold: float = 0.76,
    ) -> None:
        """
        Initialize context compressor.

        Args:
            retriever: Base retriever to fetch initial documents.
            llm: LangChain LLM for LLM-based compression.
            embeddings: LangChain embeddings for embeddings-based filtering.
            method: Compression method ('llm' or 'embeddings').
            similarity_threshold: Threshold for embeddings-based filtering.
        """
        self.retriever = retriever
        self.llm = llm
        self.embeddings = embeddings
        self.method = method
        self.similarity_threshold = similarity_threshold

    def _compress_with_llm(self, query: str, documents: list[Document]) -> list[Document]:
        """Use LLM to extract relevant sentences from each document."""
        if self.llm is None:
            logger.warning("no_llm_for_compression, returning uncompressed")
            return documents

        from langchain_core.messages import HumanMessage

        compressed: list[Document] = []
        for doc in documents:
            prompt = COMPRESSION_PROMPT.format(question=query, context=doc.content)
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                content = response.content.strip()
                if content and content.lower() != "no relevant information found.":
                    compressed.append(
                        Document(
                            content=content,
                            metadata={**doc.metadata, "compressed": True},
                            source_path=doc.source_path,
                        )
                    )
            except Exception as e:
                logger.warning("compression_failed_for_doc", error=str(e))
                compressed.append(doc)  # Keep original on failure

        return compressed

    def _compress_with_embeddings(
        self, query: str, documents: list[Document]
    ) -> list[Document]:
        """Use embedding similarity to filter low-relevance documents."""
        if self.embeddings is None:
            logger.warning("no_embeddings_for_compression, returning uncompressed")
            return documents

        try:
            query_embedding = self.embeddings.embed_query(query)
            doc_texts = [doc.content for doc in documents]
            doc_embeddings = self.embeddings.embed_documents(doc_texts)

            filtered: list[Document] = []
            for doc, doc_emb in zip(documents, doc_embeddings):
                # Cosine similarity
                dot_product = sum(a * b for a, b in zip(query_embedding, doc_emb))
                norm_q = sum(a * a for a in query_embedding) ** 0.5
                norm_d = sum(a * a for a in doc_emb) ** 0.5
                similarity = dot_product / max(norm_q * norm_d, 1e-8)

                if similarity >= self.similarity_threshold:
                    doc.metadata["similarity_score"] = round(similarity, 4)
                    filtered.append(doc)

            logger.info(
                "embeddings_compression",
                original=len(documents),
                filtered=len(filtered),
                threshold=self.similarity_threshold,
            )
            return filtered

        except Exception as e:
            logger.error("embeddings_compression_failed", error=str(e))
            return documents

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        """
        Retrieve and compress documents.

        Args:
            query: Search query.
            k: Number of documents to retrieve before compression.

        Returns:
            List of compressed/filtered Documents.
        """
        # Fetch more than k to allow for filtering
        fetch_k = k * 2

        # Get documents from base retriever
        if hasattr(self.retriever, "retrieve"):
            documents = self.retriever.retrieve(query, k=fetch_k)
        elif hasattr(self.retriever, "invoke"):
            lc_docs = self.retriever.invoke(query)[:fetch_k]
            documents = [
                Document(
                    content=d.page_content,
                    metadata=d.metadata,
                    source_path=d.metadata.get("source_path", ""),
                )
                for d in lc_docs
            ]
        else:
            logger.error("unsupported_retriever_type")
            return []

        # Compress based on method
        if self.method == "llm":
            compressed = self._compress_with_llm(query, documents)
        elif self.method == "embeddings":
            compressed = self._compress_with_embeddings(query, documents)
        else:
            logger.warning("unknown_compression_method", method=self.method)
            compressed = documents

        result = compressed[:k]
        logger.info(
            "context_compressed",
            method=self.method,
            original=len(documents),
            compressed=len(result),
        )
        return result
