"""
RAGX Prompt Templates — Pre-built prompt templates for RAG generation.

Centralises all prompt engineering: system prompts with anti-hallucination
rules, citation-enforcing templates, query expansion prompts, context
compression prompts, and helper functions for formatting context and
building message lists.
"""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from ragx.config.logging_config import get_logger
from ragx.ingestion.loaders.base import Document

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT: str = (
    "You are RAGX, a precise and helpful question-answering assistant.\n"
    "You MUST follow these rules:\n"
    "1. Answer ONLY based on the provided context. Do NOT use prior knowledge.\n"
    "2. If the context does not contain enough information, say "
    '"I don\'t have enough information to answer this question."\n'
    "3. Always cite your sources using [Source N] format where N corresponds "
    "to the document number in the context.\n"
    "4. Be concise and factual. Do not speculate or add opinions.\n"
    "5. If multiple sources agree, synthesise the information and cite all "
    "relevant sources.\n"
    "6. If sources conflict, acknowledge the discrepancy and cite each source.\n"
    "7. Never fabricate information, quotes, statistics, or citations.\n"
    "8. Use structured formatting (bullet points, numbered lists) when "
    "appropriate for clarity.\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# RAG Prompt Template (LangChain ChatPromptTemplate)
# ─────────────────────────────────────────────────────────────────────────────

RAG_PROMPT_TEMPLATE: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        (
            "human",
            "Context:\n{context}\n\n"
            "---\n"
            "Question: {query}\n\n"
            "Provide a comprehensive answer based on the context above. "
            "Cite sources using [Source N] format.",
        ),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Citation Prompt
# ─────────────────────────────────────────────────────────────────────────────

CITATION_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a citation-aware assistant. When answering questions, "
            "you MUST cite every claim using [Source N] format, where N "
            "refers to the numbered source in the provided context. "
            "Every factual statement must have at least one citation. "
            "At the end, list all sources you referenced.",
        ),
        (
            "human",
            "Context:\n{context}\n\n"
            "---\n"
            "Question: {query}\n\n"
            "Answer with inline citations [Source N] for every claim:",
        ),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Query Expansion Prompt
# ─────────────────────────────────────────────────────────────────────────────

QUERY_EXPANSION_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at reformulating search queries. "
            "Given a user question, generate 3-5 alternative versions "
            "of the question that capture different aspects or phrasings. "
            "Each alternative should help retrieve different relevant documents. "
            "Return ONLY the alternative queries, one per line, numbered.",
        ),
        (
            "human",
            "Original question: {query}\n\n"
            "Generate alternative search queries:",
        ),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Compression Prompt
# ─────────────────────────────────────────────────────────────────────────────

COMPRESSION_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at extracting relevant information. "
            "Given a user question and a document, extract ONLY the parts "
            "of the document that are directly relevant to answering the "
            "question. Remove all irrelevant content. Preserve exact quotes "
            "and data points. If the document contains no relevant information, "
            'respond with "NOT_RELEVANT".',
        ),
        (
            "human",
            "Question: {query}\n\n"
            "Document:\n{document}\n\n"
            "Extract only the relevant parts:",
        ),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def format_context(documents: list[Document]) -> str:
    """Format a list of documents into a numbered context string.

    Each document is rendered as::

        [1] (source: path/to/file.pdf, page: 3)
        <content>

    Args:
        documents: The retrieved documents to format.

    Returns:
        A single string with all documents numbered and formatted.
    """
    if not documents:
        return "No context available."

    parts: list[str] = []
    for idx, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", doc.metadata.get("page_number"))
        section = doc.metadata.get("section", "")

        header_parts = [f"source: {source}"]
        if page is not None:
            header_parts.append(f"page: {page}")
        if section:
            header_parts.append(f"section: {section}")

        header = ", ".join(header_parts)
        parts.append(f"[{idx}] ({header})\n{doc.content}")

    return "\n\n".join(parts)


def build_prompt(
    query: str,
    context: list[Document],
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Build the full message list for a chat-model invocation.

    Args:
        query:         The user question.
        context:       Retrieved context documents.
        system_prompt: Optional override for the system prompt.
                       Defaults to ``RAG_SYSTEM_PROMPT``.

    Returns:
        A list of ``{"role": ..., "content": ...}`` dicts ready for the
        chat model.
    """
    sys_prompt = system_prompt or RAG_SYSTEM_PROMPT
    formatted_context = format_context(context)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": (
                f"Context:\n{formatted_context}\n\n"
                f"---\n"
                f"Question: {query}\n\n"
                "Provide a comprehensive answer based on the context above. "
                "Cite sources using [Source N] format."
            ),
        },
    ]

    logger.debug(
        "prompt_built",
        query_length=len(query),
        context_docs=len(context),
        system_prompt_length=len(sys_prompt),
    )
    return messages
