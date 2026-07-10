"""
RAGX Settings — Centralized configuration management.

All settings are loaded from environment variables and .env file
using Pydantic BaseSettings for validation and type coercion.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    OPENAI = "openai"
    BGE = "bge"
    SENTENCE_TRANSFORMER = "sentence-transformer"


class VectorStoreType(str, Enum):
    """Supported vector store backends."""
    CHROMA = "chroma"
    FAISS = "faiss"


class ChunkingStrategy(str, Enum):
    """Supported chunking strategies."""
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class RetrievalStrategy(str, Enum):
    """Supported retrieval strategies."""
    SIMILARITY = "similarity"
    BM25 = "bm25"
    HYBRID = "hybrid"


class RerankerType(str, Enum):
    """Supported reranker types."""
    CROSS_ENCODER = "cross-encoder"
    COHERE = "cohere"
    NONE = "none"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="RAGX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── General ──────────────────────────────────────────────────────────────
    env: Environment = Environment.DEVELOPMENT
    debug: bool = True
    log_level: str = "INFO"

    # ── API Keys (no prefix — standard env var names) ────────────────────────
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    cohere_api_key: Optional[str] = Field(default=None, alias="COHERE_API_KEY")

    # ── Embedding Configuration ──────────────────────────────────────────────
    embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMER
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── Vector Store ─────────────────────────────────────────────────────────
    vectorstore_type: VectorStoreType = VectorStoreType.CHROMA
    vectorstore_path: Path = Path("./data/vectorstore")
    chroma_collection: str = "ragx_default"

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 100
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieval_top_k: int = 5
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    reranker: RerankerType = RerankerType.CROSS_ENCODER
    reranker_top_n: int = 5

    # ── LLM Configuration ───────────────────────────────────────────────────
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    llm_model: str = "llama3.2:3b"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    llm_streaming: bool = True

    # ── Ollama (Local LLM) ──────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # ── API Server ──────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_cors_origins: list[str] = ["http://localhost:3000"]

    # ── Data Paths ──────────────────────────────────────────────────────────
    data_raw_path: Path = Path("./data/raw")
    data_processed_path: Path = Path("./data/processed")
    data_logs_path: Path = Path("./data/logs")

    # ── Monitoring ──────────────────────────────────────────────────────────
    metrics_enabled: bool = True
    metrics_port: int = 9090

    # ── Conversation Memory ─────────────────────────────────────────────────
    memory_window_size: int = 10

    def ensure_directories(self) -> None:
        """Create all required data directories."""
        for path in [
            self.data_raw_path,
            self.data_processed_path,
            self.data_logs_path,
            self.vectorstore_path / "faiss",
            self.vectorstore_path / "chroma",
        ]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    settings = Settings()
    settings.ensure_directories()
    return settings
