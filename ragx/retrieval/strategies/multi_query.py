"""
RAGX Multi-Query Retriever — LLM-powered query expansion.

Generates multiple alternative formulations of the user's query using
an LLM, retrieves results for each variant, and merges the de-duplicated
result sets to improve recall.
"""

from __future__ import annotations

from typing import Any

from ragx.config.logging_config import get_logger
from ragx.ingestion.loaders.base import Document

logger = get_logger(__name__)

_QUERY_GENERATION_TEMPLATE = """\
You are an AI assistant helping to improve document retrieval.
Given the user's question, generate {num_queries} alternative formulations
that capture different aspects or phrasings of the same information need.

Each alternative should approach the question from a different angle to
maximise the chance of retrieving relevant documents.

Return ONLY the alternative queries, one per line, without numbering or
bullet points.

Original question: {question}

Alternative queries:"""


class MultiQueryRetriever:
    """Retriever that expands a query into multiple variants via an LLM.

    For each variant the base retriever is invoked and results are merged
    with de-duplication (by ``content_hash``) to boost recall.

    Attributes:
        retriever: Base retriever (any object with a ``search`` or
            ``invoke`` method that returns a list of Documents).
        llm: LLM used to generate alternative queries (must expose an
            ``invoke`` method returning text).
    """

    def __init__(self, retriever: Any, llm: Any) -> None:
        """Initialise the multi-query retriever.

        Args:
            retriever: A base retriever with a ``search(query, k)`` or
                ``invoke(query)`` method.
            llm: An LLM instance with an ``invoke(prompt)`` method that
                returns generated text (or a message with ``.content``).
        """
        self.retriever = retriever
        self.llm = llm
        logger.info(
            "multi_query_retriever_initialized",
            retriever_type=type(retriever).__name__,
            llm_type=type(llm).__name__,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        num_queries: int = 3,
        k: int = 5,
    ) -> list[Document]:
        """Retrieve documents using multiple query variants.

        Args:
            query: The user's original query.
            num_queries: Number of alternative queries to generate.
            k: Number of documents to retrieve per query variant.

        Returns:
            De-duplicated list of :class:`Document` objects from all
            query variants.
        """
        queries = self._generate_queries(query, num_queries=num_queries)
        all_queries = [query, *queries]  # always include the original

        logger.debug(
            "multi_query_variants_generated",
            original=query[:80],
            variants=len(queries),
        )

        seen_hashes: set[str] = set()
        merged: list[Document] = []

        for q in all_queries:
            docs = self._run_retriever(q, k=k)
            for doc in docs:
                if doc.content_hash not in seen_hashes:
                    seen_hashes.add(doc.content_hash)
                    merged.append(doc)

        logger.info(
            "multi_query_retrieval_completed",
            original_query=query[:80],
            total_queries=len(all_queries),
            unique_results=len(merged),
        )
        return merged

    # ── Internal helpers ─────────────────────────────────────────────────

    def _generate_queries(
        self, query: str, num_queries: int
    ) -> list[str]:
        """Use the LLM to generate alternative query formulations.

        Args:
            query: The original user query.
            num_queries: Number of alternatives to produce.

        Returns:
            A list of alternative query strings.
        """
        prompt = _QUERY_GENERATION_TEMPLATE.format(
            num_queries=num_queries, question=query
        )

        try:
            response = self.llm.invoke(prompt)

            # Handle both raw string and LangChain message objects
            if hasattr(response, "content"):
                text = response.content
            else:
                text = str(response)

            # Parse one query per line, ignoring blanks
            alternatives = [
                line.strip()
                for line in text.strip().splitlines()
                if line.strip()
            ]

            # Limit to the requested number
            return alternatives[:num_queries]

        except Exception:
            logger.exception(
                "multi_query_generation_failed",
                query_preview=query[:80],
            )
            # Graceful degradation: return empty list so original query
            # is still used.
            return []

    def _run_retriever(self, query: str, k: int) -> list[Document]:
        """Run the base retriever for a single query.

        Handles both RAGX-style retrievers (``search(query, k)``) and
        LangChain-style retrievers (``invoke(query)``).

        Args:
            query: Query string to search for.
            k: Number of results to request.

        Returns:
            List of Documents from the base retriever.
        """
        try:
            # RAGX retrievers
            if hasattr(self.retriever, "search"):
                return self.retriever.search(query, k=k)

            # LangChain retrievers
            if hasattr(self.retriever, "invoke"):
                lc_docs = self.retriever.invoke(query)
                return [Document.from_langchain(d) for d in lc_docs[:k]]

            # Callable retriever
            if callable(self.retriever):
                return self.retriever(query, k=k)

            raise TypeError(
                f"Retriever of type {type(self.retriever).__name__} does not "
                "support search(), invoke(), or __call__()."
            )
        except TypeError:
            raise
        except Exception:
            logger.exception(
                "multi_query_retriever_run_error",
                query_preview=query[:80],
            )
            return []
