"""
RAGX Prometheus Metrics — Metric definitions and instrumentation.

Defines all Prometheus metrics for monitoring RAG pipeline performance
and provides a decorator for instrumenting query functions.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from prometheus_client import Counter, Gauge, Histogram

from ragx.config.logging_config import get_logger

logger = get_logger(__name__)

# ── Metric Definitions ───────────────────────────────────────────────────────

QUERY_LATENCY = Histogram(
    "ragx_query_latency_seconds",
    "Total query processing latency",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

RETRIEVAL_LATENCY = Histogram(
    "ragx_retrieval_latency_seconds",
    "Retrieval step latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

GENERATION_LATENCY = Histogram(
    "ragx_generation_latency_seconds",
    "LLM generation step latency",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

REQUEST_COUNT = Counter(
    "ragx_requests_total",
    "Total number of requests",
    ["endpoint", "method", "status"],
)

ERROR_COUNT = Counter(
    "ragx_errors_total",
    "Total number of errors",
    ["endpoint", "error_type"],
)

TOKENS_USED = Counter(
    "ragx_tokens_total",
    "Total tokens consumed",
    ["type"],  # prompt, completion
)

ACTIVE_SESSIONS = Gauge(
    "ragx_active_sessions",
    "Number of active conversation sessions",
)

DOCUMENT_COUNT = Gauge(
    "ragx_documents_total",
    "Total number of documents in the vector store",
)


# ── Helper Functions ─────────────────────────────────────────────────────────


def record_retrieval_latency(seconds: float) -> None:
    """Record retrieval step latency."""
    RETRIEVAL_LATENCY.observe(seconds)


def record_generation_latency(seconds: float) -> None:
    """Record LLM generation step latency."""
    GENERATION_LATENCY.observe(seconds)


def record_tokens(prompt_tokens: int, completion_tokens: int) -> None:
    """Record token usage."""
    TOKENS_USED.labels(type="prompt").inc(prompt_tokens)
    TOKENS_USED.labels(type="completion").inc(completion_tokens)


def update_document_count(count: int) -> None:
    """Update the document count gauge."""
    DOCUMENT_COUNT.set(count)


def update_active_sessions(count: int) -> None:
    """Update the active sessions gauge."""
    ACTIVE_SESSIONS.set(count)


# ── Decorator ────────────────────────────────────────────────────────────────


def track_query(func: Callable) -> Callable:
    """
    Decorator that instruments a query function with latency and error metrics.

    Usage:
        @track_query
        def my_query_handler(query: str) -> dict:
            ...
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            QUERY_LATENCY.observe(elapsed)
            REQUEST_COUNT.labels(
                endpoint=func.__name__, method="POST", status="success"
            ).inc()
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            QUERY_LATENCY.observe(elapsed)
            ERROR_COUNT.labels(
                endpoint=func.__name__, error_type=type(e).__name__
            ).inc()
            REQUEST_COUNT.labels(
                endpoint=func.__name__, method="POST", status="error"
            ).inc()
            raise

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            QUERY_LATENCY.observe(elapsed)
            REQUEST_COUNT.labels(
                endpoint=func.__name__, method="POST", status="success"
            ).inc()
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            QUERY_LATENCY.observe(elapsed)
            ERROR_COUNT.labels(
                endpoint=func.__name__, error_type=type(e).__name__
            ).inc()
            raise

    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return wrapper


# ── FastAPI Instrumentation ──────────────────────────────────────────────────


def instrument_app(app: Any) -> None:
    """
    Add Prometheus instrumentation to a FastAPI application.

    Args:
        app: FastAPI application instance.
    """
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            excluded_handlers=["/health", "/metrics"],
        ).instrument(app).expose(app, endpoint="/metrics")

        logger.info("prometheus_instrumentation_enabled")
    except ImportError:
        # Fallback: mount basic prometheus metrics endpoint
        from prometheus_client import make_asgi_app

        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)
        logger.info("prometheus_basic_metrics_enabled")
