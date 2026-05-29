"""
RAGX Query Logger — JSONL-based query/response logging for analysis.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ragx.config.logging_config import get_logger
from ragx.config.settings import get_settings

logger = get_logger(__name__)


class QueryLogger:
    """Logs queries, responses, and retrieved chunks to JSONL files."""

    def __init__(self, log_dir: str | None = None) -> None:
        """
        Initialize query logger.

        Args:
            log_dir: Directory for log files. Uses settings default if None.
        """
        settings = get_settings()
        self.log_dir = Path(log_dir or settings.data_logs_path)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "queries.jsonl"

    def log_query(
        self,
        query: str,
        response: dict[str, Any],
        retrieved_chunks: list[dict[str, Any]] | None = None,
        latency_ms: float = 0.0,
        session_id: str | None = None,
    ) -> None:
        """
        Log a query and its response.

        Args:
            query: User query.
            response: Response dict (answer, sources, confidence, etc.).
            retrieved_chunks: List of retrieved chunk metadata.
            latency_ms: Total processing latency in milliseconds.
            session_id: Session ID if in a conversation.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "answer": response.get("answer", ""),
            "confidence_score": response.get("confidence_score", 0.0),
            "model": response.get("model", ""),
            "latency_ms": round(latency_ms, 2),
            "num_sources": len(response.get("sources", [])),
            "sources": response.get("sources", []),
            "num_chunks_retrieved": len(retrieved_chunks) if retrieved_chunks else 0,
            "session_id": session_id,
        }

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.error("query_logging_failed", error=str(e))

    def get_recent_queries(self, n: int = 50) -> list[dict[str, Any]]:
        """
        Read the most recent N queries from the log.

        Args:
            n: Number of recent queries to return.

        Returns:
            List of query log records (most recent last).
        """
        if not self.log_file.exists():
            return []

        records: list[dict[str, Any]] = []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error("query_log_read_failed", error=str(e))

        return records[-n:]

    def get_query_stats(self) -> dict[str, Any]:
        """
        Aggregate query statistics from logs.

        Returns:
            Dict with total_queries, avg_latency_ms, avg_confidence,
            queries_per_model, etc.
        """
        records = self.get_recent_queries(n=10000)  # Read all

        if not records:
            return {
                "total_queries": 0,
                "avg_latency_ms": 0.0,
                "avg_confidence": 0.0,
                "queries_per_model": {},
            }

        latencies = [r.get("latency_ms", 0) for r in records]
        confidences = [r.get("confidence_score", 0) for r in records]

        model_counts: dict[str, int] = {}
        for r in records:
            model = r.get("model", "unknown")
            model_counts[model] = model_counts.get(model, 0) + 1

        return {
            "total_queries": len(records),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "avg_confidence": round(sum(confidences) / len(confidences), 4),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "queries_per_model": model_counts,
        }
