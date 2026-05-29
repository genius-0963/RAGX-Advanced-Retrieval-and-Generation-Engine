"""
RAGX API Schemas — Pydantic request/response models for all API endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Ingestion ────────────────────────────────────────────────────────────────


class IngestURLRequest(BaseModel):
    """Request to ingest a document from a URL."""
    url: str = Field(..., description="URL to ingest")


class IngestBatchRequest(BaseModel):
    """Request to ingest multiple URLs."""
    urls: list[str] = Field(..., description="List of URLs to ingest")


# ── Query ────────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request to query the RAG system."""
    query: str = Field(..., description="User question")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to retrieve")
    strategy: str = Field(default="hybrid", description="Retrieval strategy")
    use_reranker: bool = Field(default=True, description="Whether to apply reranking")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation")


class QueryResponse(BaseModel):
    """Response from a query."""
    answer: str = Field(..., description="Generated answer")
    sources: list[dict[str, Any]] = Field(default_factory=list, description="Source citations")
    confidence_score: float = Field(..., description="Answer confidence (0-1)")
    model: str = Field(..., description="LLM model used")
    latency_ms: float = Field(..., description="Total processing time in ms")


# ── Chat ─────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Request for conversational chat."""
    message: str = Field(..., description="User message")
    session_id: str = Field(default="default", description="Conversation session ID")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to retrieve")


class ChatResponse(QueryResponse):
    """Response from a chat message."""
    conversation_id: str = Field(..., description="Conversation/session ID")


# ── Documents ────────────────────────────────────────────────────────────────


class DocumentInfo(BaseModel):
    """Information about an ingested document."""
    id: str = Field(..., description="Document ID")
    source: str = Field(..., description="Source file path or URL")
    file_type: str = Field(..., description="File type/extension")
    upload_time: str = Field(..., description="Upload timestamp (ISO 8601)")
    chunk_count: int = Field(default=0, description="Number of chunks")


class DocumentListResponse(BaseModel):
    """Response listing ingested documents."""
    documents: list[DocumentInfo] = Field(default_factory=list)
    total: int = Field(default=0)


# ── Feedback ─────────────────────────────────────────────────────────────────


class FeedbackRequest(BaseModel):
    """Request to submit feedback on a response."""
    query_id: str = Field(..., description="ID of the query/response")
    rating: int = Field(..., ge=1, le=5, description="Rating (1-5)")
    comment: Optional[str] = Field(default=None, description="Optional comment")


class FeedbackResponse(BaseModel):
    """Response confirming feedback submission."""
    status: str = "received"
    feedback_id: str = Field(..., description="Feedback record ID")


# ── Admin ────────────────────────────────────────────────────────────────────


class StatsResponse(BaseModel):
    """System statistics."""
    total_documents: int = 0
    total_chunks: int = 0
    vectorstore_type: str = ""
    embedding_model: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    active_sessions: int = 0
    uptime_seconds: float = 0.0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ConfigResponse(BaseModel):
    """Sanitized configuration (no API keys)."""
    environment: str = ""
    embedding_provider: str = ""
    embedding_model: str = ""
    vectorstore_type: str = ""
    chunking_strategy: str = ""
    chunk_size: int = 0
    chunk_overlap: int = 0
    retrieval_strategy: str = ""
    reranker: str = ""
    llm_provider: str = ""
    llm_model: str = ""
