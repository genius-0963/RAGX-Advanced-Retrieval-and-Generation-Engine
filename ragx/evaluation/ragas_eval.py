"""
RAGX RAGAS Evaluation — Evaluate RAG pipeline quality using RAGAS metrics.

Supports faithfulness, answer relevancy, context precision, and context recall.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragx.config.logging_config import get_logger

logger = get_logger(__name__)


class RAGASEvaluator:
    """Evaluator using RAGAS framework for RAG quality metrics."""

    def __init__(self, llm: Any = None, embeddings: Any = None) -> None:
        """
        Initialize RAGAS evaluator.

        Args:
            llm: LangChain LLM for evaluation (uses default if None).
            embeddings: LangChain embeddings for evaluation (uses default if None).
        """
        self.llm = llm
        self.embeddings = embeddings
        self._metrics = None

    def _get_metrics(self) -> list:
        """Lazily load RAGAS metrics to avoid import overhead."""
        if self._metrics is None:
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
            self._metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        return self._metrics

    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str] | None = None,
    ) -> dict:
        """
        Evaluate RAG pipeline using RAGAS metrics.

        Args:
            questions: List of input queries.
            answers: List of generated answers.
            contexts: List of retrieved context lists (one list per question).
            ground_truths: Optional list of reference answers.

        Returns:
            Dictionary with aggregate scores and per-question breakdown.
        """
        from datasets import Dataset
        from ragas import evaluate

        logger.info(
            "starting_ragas_evaluation",
            num_questions=len(questions),
        )

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths is not None:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)
        metrics = self._get_metrics()

        eval_kwargs: dict[str, Any] = {"dataset": dataset, "metrics": metrics}
        if self.llm is not None:
            eval_kwargs["llm"] = self.llm
        if self.embeddings is not None:
            eval_kwargs["embeddings"] = self.embeddings

        try:
            result = evaluate(**eval_kwargs)
        except Exception as e:
            logger.error("ragas_evaluation_failed", error=str(e))
            raise

        scores = {
            "aggregate": {
                "faithfulness": float(result.get("faithfulness", 0.0)),
                "answer_relevancy": float(result.get("answer_relevancy", 0.0)),
                "context_precision": float(result.get("context_precision", 0.0)),
                "context_recall": float(result.get("context_recall", 0.0)),
            },
            "per_question": [],
        }

        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            for _, row in df.iterrows():
                scores["per_question"].append(
                    {
                        "question": row.get("question", ""),
                        "faithfulness": float(row.get("faithfulness", 0.0)),
                        "answer_relevancy": float(row.get("answer_relevancy", 0.0)),
                        "context_precision": float(row.get("context_precision", 0.0)),
                        "context_recall": float(row.get("context_recall", 0.0)),
                    }
                )

        logger.info("ragas_evaluation_complete", scores=scores["aggregate"])
        return scores

    def export_results(
        self, results: dict, output_path: str, format: str = "json"
    ) -> None:
        """
        Export evaluation results to file.

        Args:
            results: Evaluation results dictionary.
            output_path: Output file path.
            format: Output format ('json' or 'csv').
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
        elif format == "csv":
            import csv

            per_q = results.get("per_question", [])
            if per_q:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=per_q[0].keys())
                    writer.writeheader()
                    writer.writerows(per_q)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")

        logger.info("results_exported", path=str(path), format=format)
