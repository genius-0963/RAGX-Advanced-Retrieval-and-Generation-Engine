"""
RAGX API Middleware — Request logging, CORS, and request ID injection.
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from ragx.config.logging_config import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs request method, path, status code, and latency."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())[:8]

        response = await call_next(request)

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "http_request",
            method=request.method,
            path=str(request.url.path),
            status=response.status_code,
            latency_ms=round(latency_ms, 2),
            request_id=request_id,
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(round(latency_ms, 2))
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Injects a unique request ID into each request."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def configure_cors(app) -> None:
    """Configure CORS middleware on the FastAPI app."""
    from fastapi.middleware.cors import CORSMiddleware
    from ragx.config.settings import get_settings

    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
