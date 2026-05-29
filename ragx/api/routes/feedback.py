"""
RAGX Feedback Routes — User feedback collection and reporting.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

from ragx.api.schemas import FeedbackRequest, FeedbackResponse
from ragx.config.logging_config import get_logger
from ragx.config.settings import get_settings

logger = get_logger(__name__)
router = APIRouter()

FEEDBACK_FILE = "feedback.jsonl"


def _get_feedback_path() -> Path:
    """Get the feedback log file path."""
    settings = get_settings()
    log_dir = Path(settings.data_logs_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / FEEDBACK_FILE


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback on a query response."""
    feedback_id = str(uuid.uuid4())

    record = {
        "feedback_id": feedback_id,
        "query_id": request.query_id,
        "rating": request.rating,
        "comment": request.comment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    feedback_path = _get_feedback_path()
    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.info("feedback_received", feedback_id=feedback_id, rating=request.rating)
    return FeedbackResponse(feedback_id=feedback_id)


@router.get("/feedback/report")
async def feedback_report():
    """Get aggregated feedback metrics."""
    feedback_path = _get_feedback_path()

    if not feedback_path.exists():
        return {
            "total_feedback": 0,
            "average_rating": 0.0,
            "rating_distribution": {},
        }

    records: list[dict] = []
    with open(feedback_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return {
            "total_feedback": 0,
            "average_rating": 0.0,
            "rating_distribution": {},
        }

    ratings = [r.get("rating", 0) for r in records]
    distribution = {}
    for r in ratings:
        distribution[str(r)] = distribution.get(str(r), 0) + 1

    return {
        "total_feedback": len(records),
        "average_rating": round(sum(ratings) / len(ratings), 2),
        "rating_distribution": distribution,
        "recent_comments": [
            {"rating": r["rating"], "comment": r.get("comment", "")}
            for r in records[-10:]
            if r.get("comment")
        ],
    }
