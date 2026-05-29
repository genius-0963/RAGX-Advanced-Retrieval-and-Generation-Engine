"""
RAGX Llama LLM — Local Llama via Ollama integration.

Provides ``LlamaLLM``, a concrete implementation of ``BaseLLM`` that
wraps ``langchain_ollama.ChatOllama`` for locally-hosted Llama models.
No API key is required.
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


class LlamaLLM(BaseLLM):
    """Local Llama integration using ``langchain_ollama.ChatOllama``.

    Connects to a locally-running Ollama server.  No API key is needed.
    Supports synchronous generation with token-usage reporting and
    token-by-token streaming.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        """Initialise the local Llama chat model via Ollama.

        Args:
            model:       Ollama model tag (e.g. ``llama3.1:8b``).
            base_url:    Base URL of the Ollama server.
            temperature: Sampling temperature.
            max_tokens:  Maximum tokens to generate.
        """
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)

        from langchain_ollama import ChatOllama

        settings = get_settings()
        resolved_url = base_url or settings.ollama_base_url

        self._llm = ChatOllama(
            model=model,
            base_url=resolved_url,
            temperature=temperature,
            num_predict=max_tokens,
        )
        self._base_url = resolved_url
        logger.info("llama_llm_ready", model=model, base_url=resolved_url)

    # ── Public interface ────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        context: list[Document] | None = None,
    ) -> GenerationResponse:
        """Generate a complete answer using a local Llama model.

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

        # Token usage — Ollama may or may not provide this
        usage_meta = getattr(response, "usage_metadata", None) or {}
        tokens_used = {
            "prompt_tokens": usage_meta.get("input_tokens", 0),
            "completion_tokens": usage_meta.get("output_tokens", 0),
            "total_tokens": usage_meta.get("total_tokens", 0),
        }

        confidence = self._calculate_confidence(answer, context or [])

        logger.info(
            "llama_generate_complete",
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
        """Stream tokens from the local Llama model.

        Args:
            prompt:  User query.
            context: Optional grounding documents.

        Yields:
            Individual text chunks.
        """
        messages = self._build_messages(prompt, context)
        lc_messages = _dicts_to_lc_messages(messages)

        logger.debug("llama_stream_start", model=self.model)
        for chunk in self._llm.stream(lc_messages):
            token = chunk.content
            if token:
                yield token

    def get_langchain_llm(self) -> Any:
        """Return the underlying ``ChatOllama`` instance.

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
