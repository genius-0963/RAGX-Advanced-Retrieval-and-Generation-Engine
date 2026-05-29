"""
RAGX API — FastAPI application entry point.

Production-ready API server with lifespan management,
routing, middleware, and health checks.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import os

from ragx.api.middleware import RequestLoggingMiddleware, configure_cors
from ragx.api.routes import admin, feedback, ingest, query
from ragx.api.schemas import HealthResponse
from ragx.config.logging_config import get_logger, setup_logging
from ragx.config.settings import get_settings

logger = get_logger(__name__)

# Track startup time for uptime calculation
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize on startup, cleanup on shutdown."""
    global _start_time

    settings = get_settings()
    setup_logging(log_level=settings.log_level, json_format=settings.env.value == "production")
    settings.ensure_directories()

    _start_time = time.time()
    logger.info(
        "ragx_starting",
        environment=settings.env.value,
        llm_provider=settings.llm_provider.value,
        vectorstore=settings.vectorstore_type.value,
    )

    # Lazy initialization happens on first request via dependencies
    yield

    logger.info("ragx_shutting_down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="RAGX API",
        version="1.0.0",
        description=(
            "RAGX — Advanced Retrieval and Generation Engine. "
            "A production-grade RAG platform with multi-format ingestion, "
            "hybrid retrieval, multi-LLM generation, and evaluation."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Middleware
    configure_cors(app)
    app.add_middleware(RequestLoggingMiddleware)

    # Prometheus metrics (if enabled)
    if settings.metrics_enabled:
        try:
            from ragx.monitoring.prometheus_metrics import instrument_app
            instrument_app(app)
        except ImportError:
            logger.warning("prometheus_instrumentator_not_available")

    # Routes
    app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
    app.include_router(query.router, prefix="/api/v1", tags=["Query"])
    app.include_router(admin.router, prefix="/api/v1", tags=["Admin"])
    app.include_router(feedback.router, prefix="/api/v1", tags=["Feedback"])

    # Root & health endpoints
    @app.get("/", include_in_schema=False)
    async def root():
        """Redirect root to UI."""
        return RedirectResponse(url="/ui/index.html")

    # Mount static files for the frontend
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)
    app.mount("/ui", StaticFiles(directory=static_dir), name="ui")

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Health check endpoint."""
        return HealthResponse()

    return app


# Create the application instance
app = create_app()


def get_start_time() -> float:
    """Get the application start timestamp."""
    return _start_time


def run() -> None:
    """Run the API server directly."""
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "ragx.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
