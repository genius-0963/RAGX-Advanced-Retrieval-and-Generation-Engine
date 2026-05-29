"""
RAGX Gemini LLM — Google Generative AI integration via LangChain.

Provides ``GeminiLLM``, a concrete implementation of ``BaseLLM`` that
wraps ``langchain_google_genai.ChatGoogleGenerativeAI`` for Gemini models.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ragx.config.logging_config import get_logger
from ragx.config.settings import get_settings
from ragx.generation.llm.base import BaseLLM, GenerationResponse
from ragx.ingestion.loaders.base import Document

logger = get_logger(__name__)


class GeminiLLM(BaseLLM):
    """Google Gemini integration using ``langchain_google_genai.ChatGoogleGenerativeAI``.

    Supports synchronous generation with token-usage reporting and
    token-by-token streaming.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        """Initialise the Google Gemini chat model.

        Args:
            model:       Gemini model identifier.
            api_key:     Google API key.  Falls back to settings / env var.
            temperature: Sampling temperature.
            max_tokens:  Maximum tokens to generate.
        """
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)

        from langchain_google_genai import ChatGoogleGenerativeAI

        settings = get_settings()
        resolved_key = api_key or settings.google_api_key

        self._llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=resolved_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        logger.info("gemini_llm_ready", model=model)

    # ── Public interface ────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        context: list[Document] | None = None,
    ) -> GenerationResponse:
        """Generate a complete answer using Google Gemini.

        Args:
            prompt:  User query.
            context: Optional grounding documents.

        Returns:
            Populated ``GenerationResponse``.
        """
        messages = self._build_messages(prompt, context)
        lc_messages = _dicts_to_lc_messages(messages)

        start = time.perf_counter()
        response = self._llm.invoke(lc_messages)
        latency_ms = (time.perf_counter() - start) * 1000.0

        answer = response.content or ""

        # Token usage from response metadata
        usage_meta = getattr(response, "usage_metadata", None) or {}
        tokens_used = {
            "prompt_tokens": usage_meta.get("input_tokens", 0),
            "completion_tokens": usage_meta.get("output_tokens", 0),
            "total_tokens": usage_meta.get("total_tokens", 0),
        }

        confidence = self._calculate_confidence(answer, context or [])

        logger.info(
            "gemini_generate_complete",
            model=self.model,
            latency_ms=round(latency_ms, 2),
            tokens=tokens_used,
        )

        return GenerationResponse(
            answer=answer,
            confidence_score=confidence,
            sources=[],
            tokens_used=tokens_used,
            model=self.model,
            latency_ms=round(latency_ms, 2),
        )

    def stream(
        self,
        prompt: str,
        context: list[Document] | None = None,
    ) -> Iterator[str]:
        """Stream tokens from Google Gemini.

        Args:
            prompt:  User query.
            context: Optional grounding documents.

        Yields:
            Individual text chunks.
        """
        messages = self._build_messages(prompt, context)
        lc_messages = _dicts_to_lc_messages(messages)

        logger.debug("gemini_stream_start", model=self.model)
        for chunk in self._llm.stream(lc_messages):
            token = chunk.content
            if token:
                yield token

    def get_langchain_llm(self) -> Any:
        """Return the underlying ``ChatGoogleGenerativeAI`` instance.

        Returns:
            The LangChain chat model.
        """
        return self._llm


# ── Utility ─────────────────────────────────────────────────────────────────


def _dicts_to_lc_messages(messages: list[dict[str, str]]) -> list[Any]:
    """Convert ``[{"role": ..., "content": ...}]`` to LangChain message objects.

    Args:
        messages: List of role/content dicts.

    Returns:
        List of LangChain ``BaseMessage`` instances.
    """
    lc_msgs: list[Any] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            lc_msgs.append(SystemMessage(content=content))
        else:
            lc_msgs.append(HumanMessage(content=content))
    return lc_msgs
