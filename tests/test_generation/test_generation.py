"""
Tests for RAGX Generation — citations, memory, prompt templates, confidence scoring.
All offline (no LLM API calls).
"""

from __future__ import annotations

import pytest

from ragx.ingestion.loaders.base import Document
from ragx.generation.citations import CitationExtractor
from ragx.generation.memory import ConversationMemory
from ragx.generation.prompts.templates import (
    RAG_SYSTEM_PROMPT,
    build_prompt,
    format_context,
)
from ragx.generation.llm.base import GenerationResponse


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_doc(content: str, source: str = "test.txt", page: int | None = None) -> Document:
    meta: dict = {"source": source}
    if page is not None:
        meta["page_number"] = page
    return Document(content=content, metadata=meta, source_path=source)


# ── CitationExtractor ─────────────────────────────────────────────────────────

class TestCitationExtractor:
    def setup_method(self):
        self.extractor = CitationExtractor()

    def test_extract_numeric_citation(self):
        docs = [make_doc("RAGX uses ChromaDB.", "doc1.txt")]
        answer = "RAGX uses ChromaDB [1] for vector storage."
        citations = self.extractor.extract_citations(answer, docs)
        assert len(citations) == 1
        assert citations[0]["citation_id"] == 1

    def test_extract_source_citation(self):
        docs = [make_doc("Python is a language.", "py.txt")]
        answer = "Python is popular [Source 1]."
        citations = self.extractor.extract_citations(answer, docs)
        assert len(citations) == 1
        assert citations[0]["citation_id"] == 1

    def test_extract_multiple_citations(self):
        docs = [
            make_doc("Fact A.", "a.txt"),
            make_doc("Fact B.", "b.txt"),
        ]
        answer = "A is true [1] and B is also true [2]."
        citations = self.extractor.extract_citations(answer, docs)
        assert len(citations) == 2
        ids = {c["citation_id"] for c in citations}
        assert 1 in ids
        assert 2 in ids

    def test_out_of_range_citation_ignored(self):
        docs = [make_doc("Only one doc.", "one.txt")]
        answer = "See [1] and [99]."
        citations = self.extractor.extract_citations(answer, docs)
        # Only [1] should be found, [99] is out of range
        assert all(c["citation_id"] == 1 for c in citations)

    def test_no_citations_creates_implicit(self):
        docs = [make_doc("Some content.", "x.txt")]
        answer = "Here is an answer with no citation markers."
        citations = self.extractor.extract_citations(answer, docs)
        # Should create implicit citations for all docs
        assert len(citations) == 1
        assert citations[0].get("implicit") is True

    def test_empty_answer_returns_empty(self):
        docs = [make_doc("content")]
        citations = self.extractor.extract_citations("", docs)
        assert citations == []

    def test_empty_docs_returns_empty(self):
        citations = self.extractor.extract_citations("answer [1]", [])
        assert citations == []

    def test_citation_includes_excerpt(self):
        docs = [make_doc("A" * 300, "long.txt")]
        answer = "Something [1]."
        citations = self.extractor.extract_citations(answer, docs)
        assert len(citations[0]["excerpt"]) <= 205  # 200 chars + "..."

    def test_format_citations(self):
        citations = [
            {"citation_id": 1, "source_document": "a.txt", "page_number": None, "section": None},
        ]
        text = self.extractor.format_citations(citations)
        assert "[1]" in text
        assert "a.txt" in text

    def test_enrich_response(self):
        citations = [
            {"citation_id": 1, "source_document": "a.txt", "page_number": 2, "section": "Intro"},
        ]
        answer = "The answer."
        enriched = self.extractor.enrich_response(answer, citations)
        assert "The answer." in enriched
        assert "[1]" in enriched

    def test_ref_citation_pattern(self):
        docs = [make_doc("Reference content.", "ref.txt")]
        answer = "See [Ref 1] for details."
        citations = self.extractor.extract_citations(answer, docs)
        assert len(citations) == 1

    def test_reference_citation_pattern(self):
        docs = [make_doc("Reference content.", "ref.txt")]
        answer = "See [Reference 1] for details."
        citations = self.extractor.extract_citations(answer, docs)
        assert len(citations) == 1


# ── ConversationMemory ────────────────────────────────────────────────────────

class TestConversationMemory:
    def setup_method(self):
        # Use unique session to avoid cross-test contamination
        import uuid
        self.session_id = str(uuid.uuid4())
        self.memory = ConversationMemory(window_size=5, session_id=self.session_id)

    def test_initial_length_zero(self):
        assert len(self.memory) == 0

    def test_add_message(self):
        self.memory.add_message("user", "Hello!")
        assert len(self.memory) == 1

    def test_add_multiple_messages(self):
        self.memory.add_message("user", "Q1")
        self.memory.add_message("assistant", "A1")
        assert len(self.memory) == 2

    def test_get_history_returns_messages(self):
        self.memory.add_message("user", "What is RAG?")
        self.memory.add_message("assistant", "RAG stands for Retrieval-Augmented Generation.")
        history = self.memory.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_get_context_string(self):
        self.memory.add_message("user", "Hello")
        self.memory.add_message("assistant", "Hi there")
        ctx = self.memory.get_context_string()
        assert "User" in ctx or "user" in ctx.lower()
        assert "Hello" in ctx
        assert "Hi there" in ctx

    def test_clear_resets_memory(self):
        self.memory.add_message("user", "Q")
        self.memory.add_message("assistant", "A")
        self.memory.clear()
        assert len(self.memory) == 0

    def test_window_trimming(self):
        # window_size=5, so 10 non-system messages → keep last 10 (window_size*2)
        for i in range(20):
            self.memory.add_message("user" if i % 2 == 0 else "assistant", f"msg {i}")
        history = self.memory.get_history()
        assert len(history) <= 10  # window_size * 2

    def test_session_isolation(self):
        import uuid
        session_a = str(uuid.uuid4())
        session_b = str(uuid.uuid4())
        mem_a = ConversationMemory(session_id=session_a)
        mem_b = ConversationMemory(session_id=session_b)
        mem_a.add_message("user", "From A")
        assert len(mem_b) == 0

    def test_to_langchain_messages(self):
        self.memory.add_message("user", "question")
        self.memory.add_message("assistant", "answer")
        lc_msgs = self.memory.to_langchain_messages()
        assert len(lc_msgs) == 2

    def test_list_sessions_class_method(self):
        sessions = ConversationMemory.list_sessions()
        assert self.session_id in sessions

    def test_clear_all_sessions(self):
        import uuid
        s = str(uuid.uuid4())
        m = ConversationMemory(session_id=s)
        m.add_message("user", "test")
        ConversationMemory.clear_all_sessions()
        assert ConversationMemory.list_sessions() == []


# ── Prompt Templates ──────────────────────────────────────────────────────────

class TestPromptTemplates:
    def test_system_prompt_not_empty(self):
        assert len(RAG_SYSTEM_PROMPT) > 100

    def test_system_prompt_contains_citation_rule(self):
        assert "[Source" in RAG_SYSTEM_PROMPT or "cite" in RAG_SYSTEM_PROMPT.lower()

    def test_system_prompt_anti_hallucination(self):
        lower = RAG_SYSTEM_PROMPT.lower()
        assert "context" in lower

    def test_format_context_with_docs(self):
        docs = [
            make_doc("First document content.", "a.pdf", page=1),
            make_doc("Second document content.", "b.pdf", page=2),
        ]
        result = format_context(docs)
        assert "[1]" in result
        assert "[2]" in result
        assert "First document" in result
        assert "Second document" in result

    def test_format_context_empty_docs(self):
        result = format_context([])
        assert "No context" in result or result == "No context available."

    def test_format_context_includes_source(self):
        docs = [make_doc("content", "myfile.pdf")]
        result = format_context(docs)
        assert "myfile.pdf" in result

    def test_format_context_includes_page_when_present(self):
        docs = [make_doc("content", "report.pdf", page=5)]
        result = format_context(docs)
        assert "5" in result  # page number appears

    def test_build_prompt_returns_list(self):
        docs = [make_doc("content")]
        messages = build_prompt("What is RAGX?", docs)
        assert isinstance(messages, list)
        assert len(messages) >= 2

    def test_build_prompt_has_system_role(self):
        docs = [make_doc("content")]
        messages = build_prompt("test query", docs)
        roles = [m["role"] for m in messages]
        assert "system" in roles

    def test_build_prompt_has_user_role(self):
        docs = [make_doc("content")]
        messages = build_prompt("test query", docs)
        roles = [m["role"] for m in messages]
        assert "user" in roles

    def test_build_prompt_query_in_user_message(self):
        docs = [make_doc("some context content")]
        query = "What is the meaning of RAG?"
        messages = build_prompt(query, docs)
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert any(query in m["content"] for m in user_msgs)

    def test_build_prompt_custom_system_prompt(self):
        docs = [make_doc("content")]
        custom = "You are a pirate assistant."
        messages = build_prompt("Q", docs, system_prompt=custom)
        sys_msgs = [m for m in messages if m["role"] == "system"]
        assert any(custom in m["content"] for m in sys_msgs)

    def test_build_prompt_empty_context(self):
        messages = build_prompt("query", [])
        assert isinstance(messages, list)
        assert len(messages) >= 2


# ── GenerationResponse dataclass ──────────────────────────────────────────────

class TestGenerationResponse:
    def test_default_values(self):
        resp = GenerationResponse(answer="Hello!")
        assert resp.answer == "Hello!"
        assert resp.confidence_score == 0.0
        assert resp.sources == []
        assert resp.model == ""
        assert resp.latency_ms == 0.0
        assert resp.tokens_used["prompt_tokens"] == 0

    def test_custom_values(self):
        resp = GenerationResponse(
            answer="The answer",
            confidence_score=0.85,
            model="gemini-2.0-flash",
            latency_ms=320.5,
            tokens_used={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        assert resp.confidence_score == 0.85
        assert resp.model == "gemini-2.0-flash"
        assert resp.latency_ms == 320.5
        assert resp.tokens_used["total_tokens"] == 150
