"""
RAGX LLM Package — Factory for creating LLM instances.
"""

from __future__ import annotations

from ragx.config.logging_config import get_logger
from ragx.config.settings import get_settings
from ragx.generation.llm.base import BaseLLM

logger = get_logger(__name__)


def get_llm(provider: str | None = None, model: str | None = None, **kwargs) -> BaseLLM:
    """
    Factory function to create LLM instances.

    Args:
        provider: LLM provider ('openai', 'anthropic', 'gemini', 'ollama').
                  Uses settings default if None.
        model: Model name override. Uses provider default if None.
        **kwargs: Additional keyword arguments passed to the LLM constructor.

    Returns:
        Configured BaseLLM instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    settings = get_settings()
    prov = provider or settings.llm_provider.value

    default_kwargs = {
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }
    default_kwargs.update(kwargs)

    if prov == "openai":
        from ragx.generation.llm.openai_llm import OpenAILLM
        return OpenAILLM(
            model=model or settings.llm_model,
            api_key=settings.openai_api_key,
            **default_kwargs,
        )
    elif prov == "anthropic":
        from ragx.generation.llm.anthropic_llm import AnthropicLLM
        return AnthropicLLM(
            model=model or "claude-sonnet-4-20250514",
            api_key=settings.anthropic_api_key,
            **default_kwargs,
        )
    elif prov == "gemini":
        from ragx.generation.llm.gemini_llm import GeminiLLM
        return GeminiLLM(
            model=model or "gemini-2.0-flash",
            api_key=settings.google_api_key,
            **default_kwargs,
        )
    elif prov == "ollama":
        from ragx.generation.llm.llama_llm import LlamaLLM
        return LlamaLLM(
            model=model or settings.ollama_model,
            base_url=settings.ollama_base_url,
            **default_kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{prov}'. "
            f"Supported: openai, anthropic, gemini, ollama"
        )


__all__ = ["get_llm", "BaseLLM"]
