"""
RAGX Admin Routes — System management and configuration endpoints.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, BackgroundTasks

from ragx.api.schemas import ConfigResponse, StatsResponse
from ragx.config.logging_config import get_logger
from ragx.config.settings import get_settings

logger = get_logger(__name__)
router = APIRouter()


@router.get("/admin/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    from ragx.api.main import get_start_time

    settings = get_settings()
    start_time = get_start_time()
    uptime = time.time() - start_time if start_time > 0 else 0

    total_docs = 0
    try:
        from ragx.api.dependencies import get_embedding_pipeline
        ep = get_embedding_pipeline()
        vs = ep.get_vectorstore()
        if vs and hasattr(vs, "count"):
            total_docs = vs.count()
    except Exception:
        pass

    active_sessions = 0
    try:
        from ragx.generation.memory import ConversationMemory
        active_sessions = len(ConversationMemory.list_sessions())
    except Exception:
        pass

    return StatsResponse(
        total_documents=total_docs,
        total_chunks=total_docs,
        vectorstore_type=settings.vectorstore_type.value,
        embedding_model=settings.embedding_model,
        llm_provider=settings.llm_provider.value,
        llm_model=settings.llm_model,
        active_sessions=active_sessions,
        uptime_seconds=round(uptime, 2),
    )


@router.post("/admin/reindex")
async def reindex(background_tasks: BackgroundTasks):
    """Trigger re-indexing of all documents."""
    background_tasks.add_task(_reindex_all)
    return {"status": "reindexing", "message": "Re-indexing started in background."}


@router.get("/admin/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration (sanitized, no API keys)."""
    settings = get_settings()
    return ConfigResponse(
        environment=settings.env.value,
        embedding_provider=settings.embedding_provider.value,
        embedding_model=settings.embedding_model,
        vectorstore_type=settings.vectorstore_type.value,
        chunking_strategy=settings.chunking_strategy.value,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        retrieval_strategy=settings.retrieval_strategy.value,
        reranker=settings.reranker.value,
        llm_provider=settings.llm_provider.value,
        llm_model=settings.llm_model,
    )


def _reindex_all() -> None:
    """Background task to re-index all processed documents."""
    import json
    from pathlib import Path
    from ragx.api.dependencies import get_embedding_pipeline
    from ragx.ingestion.loaders.base import Document

    settings = get_settings()
    processed_dir = Path(settings.data_processed_path)

    try:
        all_docs: list[Document] = []
        for json_file in processed_dir.glob("*.json"):
            with open(json_file, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    all_docs.append(Document(
                        content=item.get("content", ""),
                        metadata=item.get("metadata", {}),
                        source_path=item.get("source_path", ""),
                    ))

        if all_docs:
            ep = get_embedding_pipeline()
            ep.process(all_docs)
            logger.info("reindexing_complete", total_docs=len(all_docs))
        else:
            logger.info("reindexing_no_documents")
    except Exception as e:
        logger.error("reindexing_failed", error=str(e))
