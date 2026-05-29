"""
RAGX API Dependencies — Singleton dependency injection for FastAPI.
"""

from __future__ import annotations

from functools import lru_cache

from ragx.config.logging_config import get_logger
from ragx.config.settings import get_settings

logger = get_logger(__name__)

# Module-level singletons
_ingestion_pipeline = None
_embedding_pipeline = None
_retrieval_engine = None
_generation_pipeline = None


def get_ingestion_pipeline():
    """Get or create the ingestion pipeline singleton."""
    global _ingestion_pipeline
    if _ingestion_pipeline is None:
        from ragx.ingestion.pipeline import IngestionPipeline
        _ingestion_pipeline = IngestionPipeline()
        logger.info("ingestion_pipeline_initialized")
    return _ingestion_pipeline


def get_embedding_pipeline():
    """Get or create the embedding pipeline singleton."""
    global _embedding_pipeline
    if _embedding_pipeline is None:
        from ragx.embeddings.pipeline import EmbeddingPipeline
        _embedding_pipeline = EmbeddingPipeline()
        logger.info("embedding_pipeline_initialized")
    return _embedding_pipeline


def get_retrieval_engine():
    """Get or create the retrieval engine singleton."""
    global _retrieval_engine
    if _retrieval_engine is None:
        from ragx.retrieval.engine import RetrievalEngine
        settings = get_settings()
        embedding_pipeline = get_embedding_pipeline()
        _retrieval_engine = RetrievalEngine(
            settings=settings,
            vectorstore=embedding_pipeline.get_vectorstore(),
        )
        logger.info("retrieval_engine_initialized")
    return _retrieval_engine


def get_generation_pipeline():
    """Get or create the generation pipeline singleton."""
    global _generation_pipeline
    if _generation_pipeline is None:
        from ragx.generation.pipeline import GenerationPipeline
        retrieval_engine = get_retrieval_engine()
        _generation_pipeline = GenerationPipeline(
            retrieval_engine=retrieval_engine,
        )
        logger.info("generation_pipeline_initialized")
    return _generation_pipeline


def reset_all():
    """Reset all singletons (for testing)."""
    global _ingestion_pipeline, _embedding_pipeline, _retrieval_engine, _generation_pipeline
    _ingestion_pipeline = None
    _embedding_pipeline = None
    _retrieval_engine = None
    _generation_pipeline = None
