"""
RAGX LLM Base — Abstract interface for language model providers.

Defines the ``BaseLLM`` abstract class and ``GenerationResponse`` dataclass
that every concrete LLM backend must implement.  This ensures a uniform
contract across OpenAI, Anthropic, Gemini, and Ollama integrations.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ragx.config.logging_config import get_logger
from ragx.ingestion.loaders.base import Document

logger = get_logger(__name__)


# ── Data-transfer object ────────────────────────────────────────────────────


@dataclass
class GenerationResponse:
    """Encapsulates everything returned by a single LLM generation call.

    Attributes:
        answer:           The generated text answer.
        confidence_score: Heuristic confidence in *[0, 1]*.
        sources:          List of source-reference dicts attached to the answer.
        tokens_used:      Token-usage breakdown (prompt / completion / total).
        model:            Model identifier that produced the answer.
        latency_ms:       Wall-clock latency of the generation call in ms.
    """

    answer: str
    confidence_score: float = 0.0
    sources: list[dict[str, Any]] = field(default_factory=list)
    tokens_used: dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    model: str = ""
    latency_ms: float = 0.0


# ── Abstract base ───────────────────────────────────────────────────────────


class BaseLLM(ABC):
    """Abstract base class for all LLM provider integrations.

    Subclasses must implement ``generate``, ``stream``, and
    ``get_langchain_llm``.  Common helpers such as confidence-scoring
    are provided by the base.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        """Initialise common LLM parameters.

        Args:
            model:       Model identifier (e.g. ``gpt-4o``).
            temperature: Sampling temperature.
            max_tokens:  Maximum tokens to generate.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(
            "llm_init",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ── Abstract methods ────────────────────────────────────────────────

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: list[Document] | None = None,
    ) -> GenerationResponse:
        """Generate a complete answer from the given prompt and context.

        Args:
            prompt:  User query or instruction.
            context: Optional list of retrieved documents for grounding.

        Returns:
            A fully-populated ``GenerationResponse``.
        """

    @abstractmethod
    def stream(
        self,
        prompt: str,
        context: list[Document] | None = None,
    ) -> Iterator[str]:
        """Stream tokens as they are generated.

        Args:
            prompt:  User query or instruction.
            context: Optional list of retrieved documents for grounding.

        Yields:
            Individual text chunks / tokens.
        """

    @abstractmethod
    def get_langchain_llm(self) -> Any:
        """Return the underlying LangChain chat-model object.

        Returns:
            A LangChain ``BaseChatModel`` instance.
        """

    # ── Helpers ─────────────────────────────────────────────────────────

    def _calculate_confidence(
        self,
        answer: str,
        context: list[Document],
    ) -> float:
        """Compute a simple heuristic confidence score in *[0, 1]*.

        The heuristic considers:
        * **Coverage** — fraction of context tokens that appear in the answer.
        * **Length ratio** — reasonable answer length relative to context.
        * **Citation presence** — whether the answer contains numbered source
          references such as ``[1]``, ``[Source 2]``, etc.

        Args:
            answer:  The generated answer text.
            context: The context documents that were supplied.

        Returns:
            A float between 0.0 and 1.0.
        """
        if not answer or not context:
            return 0.0

        answer_lower = answer.lower()
        answer_tokens = set(answer_lower.split())

        # 1) Token-overlap coverage (0–0.5)
        context_tokens: set[str] = set()
        for doc in context:
            context_tokens.update(doc.content.lower().split())

        if context_tokens:
            overlap = len(answer_tokens & context_tokens) / len(context_tokens)
        else:
            overlap = 0.0
        coverage_score = min(overlap, 1.0) * 0.5

        # 2) Length ratio (0–0.3) — we penalise extremely short or long answers
        total_context_len = sum(len(d.content) for d in context)
        if total_context_len > 0:
            ratio = len(answer) / total_context_len
            # Sweet-spot: answer is 10%–80% of context length
            if 0.1 <= ratio <= 0.8:
                length_score = 0.3
            elif ratio < 0.1:
                length_score = ratio * 3.0  # ramp up
            else:
                length_score = max(0.0, 0.3 - (ratio - 0.8) * 0.3)
        else:
            length_score = 0.1

        # 3) Citation presence (0–0.2)
        import re

        citation_pattern = re.compile(r"\[\s*(?:source\s*)?\d+\s*\]", re.IGNORECASE)
        citations_found = len(citation_pattern.findall(answer))
        citation_score = min(citations_found / max(len(context), 1), 1.0) * 0.2

        confidence = coverage_score + length_score + citation_score
        return round(min(max(confidence, 0.0), 1.0), 4)

    def _build_messages(
        self,
        prompt: str,
        context: list[Document] | None = None,
    ) -> list[dict[str, str]]:
        """Build a simple message list suitable for chat models.

        Args:
            prompt:  The user query.
            context: Optional grounding documents.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts.
        """
        from ragx.generation.prompts.templates import build_prompt

        return build_prompt(query=prompt, context=context or [])
