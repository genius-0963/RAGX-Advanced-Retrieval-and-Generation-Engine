"""
RAGX Citation Extraction — Parse and format source citations from LLM responses.

Extracts [1], [2], [Source 1] patterns from answers and maps them back
to the original source documents.
"""

from __future__ import annotations

import re

from ragx.config.logging_config import get_logger
from ragx.ingestion.loaders.base import Document

logger = get_logger(__name__)


class CitationExtractor:
    """Extracts and formats source citations from LLM-generated answers."""

    # Patterns to match citations like [1], [2], [Source 1], [Ref 2], etc.
    CITATION_PATTERNS = [
        re.compile(r"\[(\d+)\]"),
        re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE),
        re.compile(r"\[Ref\s+(\d+)\]", re.IGNORECASE),
        re.compile(r"\[Reference\s+(\d+)\]", re.IGNORECASE),
    ]

    def extract_citations(
        self,
        answer: str,
        context_documents: list[Document],
    ) -> list[dict]:
        """
        Extract citations from an answer and map to context documents.

        Args:
            answer: LLM-generated answer text containing citation markers.
            context_documents: List of context Documents (indexed starting at 1).

        Returns:
            List of citation dicts with: citation_id, source_document,
            page_number, section, excerpt.
        """
        if not answer or not context_documents:
            return []

        # Find all citation numbers in the answer
        cited_numbers: set[int] = set()
        for pattern in self.CITATION_PATTERNS:
            matches = pattern.findall(answer)
            for match in matches:
                try:
                    cited_numbers.add(int(match))
                except ValueError:
                    continue

        if not cited_numbers:
            # If no explicit citations, create implicit citations for all context docs
            return self._create_implicit_citations(context_documents)

        citations: list[dict] = []
        for num in sorted(cited_numbers):
            idx = num - 1  # Convert 1-indexed to 0-indexed
            if 0 <= idx < len(context_documents):
                doc = context_documents[idx]
                citation = {
                    "citation_id": num,
                    "source_document": doc.metadata.get("source", doc.source_path),
                    "page_number": doc.metadata.get("page_number"),
                    "section": doc.metadata.get("section"),
                    "excerpt": doc.content[:200] + ("..." if len(doc.content) > 200 else ""),
                    "document_id": doc.metadata.get("document_id"),
                }
                citations.append(citation)
            else:
                logger.warning(
                    "citation_out_of_range",
                    citation_num=num,
                    num_docs=len(context_documents),
                )

        logger.info("citations_extracted", count=len(citations))
        return citations

    def _create_implicit_citations(self, documents: list[Document]) -> list[dict]:
        """Create citations for all context documents when no explicit citations exist."""
        citations: list[dict] = []
        for i, doc in enumerate(documents):
            citations.append({
                "citation_id": i + 1,
                "source_document": doc.metadata.get("source", doc.source_path),
                "page_number": doc.metadata.get("page_number"),
                "section": doc.metadata.get("section"),
                "excerpt": doc.content[:200] + ("..." if len(doc.content) > 200 else ""),
                "document_id": doc.metadata.get("document_id"),
                "implicit": True,
            })
        return citations

    def format_citations(self, citations: list[dict]) -> str:
        """
        Format citations as a readable string.

        Args:
            citations: List of citation dicts from extract_citations.

        Returns:
            Formatted citation text.
        """
        if not citations:
            return ""

        lines = ["\n---\n**Sources:**\n"]
        for citation in citations:
            source = citation.get("source_document", "Unknown")
            page = citation.get("page_number")
            section = citation.get("section")

            parts = [f"[{citation['citation_id']}] {source}"]
            if page is not None:
                parts.append(f"Page {page}")
            if section:
                parts.append(f"Section: {section}")

            lines.append(f"- {', '.join(parts)}")

        return "\n".join(lines)

    def enrich_response(self, answer: str, citations: list[dict]) -> str:
        """
        Append formatted citations to the answer.

        Args:
            answer: Original LLM answer.
            citations: List of citation dicts.

        Returns:
            Answer with appended citation block.
        """
        citation_text = self.format_citations(citations)
        if citation_text:
            return f"{answer}\n{citation_text}"
        return answer
