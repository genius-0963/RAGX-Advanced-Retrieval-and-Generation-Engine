"""
RAGX Query Routes — Query and chat endpoints.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from ragx.api.dependencies import get_generation_pipeline
from ragx.api.schemas import ChatRequest, ChatResponse, QueryRequest, QueryResponse
from ragx.config.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with retrieval and generation.

    Returns an answer with source citations and confidence score.
    """
    start = time.perf_counter()

    try:
        pipeline = get_generation_pipeline()
        response = pipeline.generate(
            query=request.query,
            session_id=request.session_id,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        return QueryResponse(
            answer=response.answer,
            sources=response.sources,
            confidence_score=response.confidence_score,
            model=response.model,
            latency_ms=round(latency_ms, 2),
        )
    except Exception as e:
        logger.error("query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Conversational chat with memory.

    Maintains conversation context across messages within the same session.
    """
    start = time.perf_counter()

    try:
        pipeline = get_generation_pipeline()
        response = pipeline.generate(
            query=request.message,
            use_memory=True,
            session_id=request.session_id,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        return ChatResponse(
            answer=response.answer,
            sources=response.sources,
            confidence_score=response.confidence_score,
            model=response.model,
            latency_ms=round(latency_ms, 2),
            conversation_id=request.session_id,
        )
    except Exception as e:
        logger.error("chat_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


@router.get("/query/history")
async def query_history(n: int = 50):
    """Get recent query history from logs."""
    try:
        from ragx.monitoring.logger import QueryLogger
        query_logger = QueryLogger()
        recent = query_logger.get_recent_queries(n=n)
        return {"queries": recent, "total": len(recent)}
    except Exception as e:
        logger.warning("query_history_failed", error=str(e))
        return {"queries": [], "total": 0}
