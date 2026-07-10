"""
RAGX Generation Pipeline — Orchestrates the full answer generation flow.

Combines retrieval, prompt building, LLM invocation, citation extraction,
and conversation memory into a unified pipeline.
"""

from __future__ import annotations

from typing import Any, Iterator

from ragx.config.logging_config import get_logger
from ragx.config.settings import Settings, get_settings
from ragx.generation.citations import CitationExtractor
from ragx.generation.llm import get_llm
from ragx.generation.llm.base import BaseLLM, GenerationResponse
from ragx.generation.memory import ConversationMemory
from ragx.generation.prompts.templates import build_prompt, format_context
from ragx.ingestion.loaders.base import Document

logger = get_logger(__name__)


class GenerationPipeline:
    """
    Orchestrates the full answer generation pipeline.

    Flow: query → (retrieve) → build prompt → LLM → extract citations
          → update memory → return response with sources.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        retrieval_engine: Any = None,
    ) -> None:
        """
        Initialize generation pipeline.

        Args:
            settings: Application settings. Uses defaults if None.
            retrieval_engine: Optional RetrievalEngine for auto-retrieval.
        """
        self.settings = settings or get_settings()
        self.retrieval_engine = retrieval_engine
        self._llm: BaseLLM | None = None
        self._citation_extractor = CitationExtractor()
        self._memories: dict[str, ConversationMemory] = {}

    def _get_llm(self) -> BaseLLM:
        """Lazily initialize the LLM."""
        if self._llm is None:
            self._llm = get_llm()
            logger.info("llm_initialized", provider=self.settings.llm_provider.value)
        return self._llm

    def _get_memory(self, session_id: str) -> ConversationMemory:
        """Get or create conversation memory for a session."""
        if session_id not in self._memories:
            self._memories[session_id] = ConversationMemory(
                window_size=self.settings.memory_window_size,
                session_id=session_id,
            )
        return self._memories[session_id]

    def generate(
        self,
        query: str,
        context: list[Document] | None = None,
        use_memory: bool = True,
        session_id: str | None = None,
    ) -> GenerationResponse:
        """
        Generate an answer for a query.

        If no context is provided and a retrieval engine is available,
        context will be auto-retrieved.

        Args:
            query: User question.
            context: Pre-retrieved context documents. Auto-retrieves if None.
            use_memory: Whether to use conversation memory.
            session_id: Session ID for memory isolation.

        Returns:
            GenerationResponse with answer, sources, confidence, etc.
        """
        sid = session_id or "default"

        # Step 1: Retrieve context if not provided
        if context is None and self.retrieval_engine is not None:
            logger.info("auto_retrieving_context", query_preview=query[:80])
            context = self.retrieval_engine.retrieve(query, use_compression=True)
        elif context is None:
            context = []

        # Step 2: Build prompt with context and conversation history
        memory = self._get_memory(sid) if use_memory else None
        conversation_history = ""
        if memory and len(memory) > 0:
            conversation_history = memory.get_context_string()

        prompt_messages = build_prompt(query, context)

        # Inject conversation history if present
        if conversation_history:
            history_msg = f"\n\nConversation History:\n{conversation_history}\n"
            # Insert before the last user message
            if len(prompt_messages) >= 2:
                prompt_messages.insert(-1, {"role": "system", "content": history_msg})

        # Step 3: Generate answer via LLM
        llm = self._get_llm()
        full_prompt = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in prompt_messages
        )
        response = llm.generate(full_prompt, context)

        # Step 4: Extract citations
        citations = self._citation_extractor.extract_citations(
            response.answer, context
        )
        sources = [
            {
                "citation_id": c["citation_id"],
                "source": c["source_document"],
                "page_number": c.get("page_number"),
                "section": c.get("section"),
                "excerpt": c.get("excerpt", ""),
            }
            for c in citations
        ]

        # Step 5: Update memory
        if memory is not None:
            memory.add_message("user", query)
            memory.add_message("assistant", response.answer)

        # Step 6: Enrich response
        response.sources = sources

        logger.info(
            "generation_complete",
            model=response.model,
            confidence=response.confidence_score,
            num_sources=len(sources),
            latency_ms=response.latency_ms,
        )
        return response

    def stream_generate(
        self,
        query: str,
        context: list[Document] | None = None,
        use_memory: bool = True,
        session_id: str | None = None,
    ) -> Iterator[str]:
        """
        Stream-generate an answer for a query.

        Args:
            query: User question.
            context: Pre-retrieved context documents.
            use_memory: Whether to use conversation memory.
            session_id: Session ID for memory isolation.

        Yields:
            Chunks of the generated answer.
        """
        sid = session_id or "default"

        if context is None and self.retrieval_engine is not None:
            context = self.retrieval_engine.retrieve(query, use_compression=True)
        elif context is None:
            context = []

        prompt_messages = build_prompt(query, context)
        full_prompt = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in prompt_messages
        )

        llm = self._get_llm()
        full_answer_chunks: list[str] = []

        for chunk in llm.stream(full_prompt, context):
            full_answer_chunks.append(chunk)
            yield chunk

        # Update memory with complete answer
        if use_memory:
            memory = self._get_memory(sid)
            full_answer = "".join(full_answer_chunks)
            memory.add_message("user", query)
            memory.add_message("assistant", full_answer)

    def query(self, query: str, session_id: str = "default") -> dict:
        """
        High-level convenience method for query → answer.

        Args:
            query: User question.
            session_id: Session ID for conversation continuity.

        Returns:
            Dict with answer, sources, confidence_score, model, latency_ms.
        """
        response = self.generate(query, session_id=session_id)
        return {
            "answer": response.answer,
            "sources": response.sources,
            "confidence_score": response.confidence_score,
            "model": response.model,
            "latency_ms": response.latency_ms,
            "tokens_used": response.tokens_used,
        }

    def clear_session(self, session_id: str) -> None:
        """Clear conversation memory for a session."""
        if session_id in self._memories:
            self._memories[session_id].clear()
            del self._memories[session_id]

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self._memories.keys())
