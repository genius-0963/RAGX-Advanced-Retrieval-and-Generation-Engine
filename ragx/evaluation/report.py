"""
RAGX Evaluation Report — Generates formatted evaluation reports.

Aggregates results from RAGAS, DeepEval, and custom metrics into
a comprehensive markdown report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ragx.config.logging_config import get_logger

logger = get_logger(__name__)


class EvaluationReport:
    """Generates and saves evaluation reports from multiple evaluators."""

    def generate_report(self, results: dict) -> str:
        """
        Generate a markdown-formatted evaluation report.

        Args:
            results: Dictionary containing evaluation results. Expected keys:
                - ragas (dict, optional): RAGAS evaluation scores
                - deepeval (dict, optional): DeepEval evaluation scores
                - custom (dict, optional): Custom metric scores
                - metadata (dict, optional): Additional metadata

        Returns:
            Formatted markdown report string.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        lines = [
            "# RAGX Evaluation Report",
            "",
            f"**Generated**: {timestamp}",
            "",
            "---",
            "",
        ]

        # RAGAS Results
        if "ragas" in results:
            ragas = results["ragas"]
            lines.extend([
                "## RAGAS Metrics",
                "",
                "| Metric | Score |",
                "|--------|-------|",
            ])
            aggregate = ragas.get("aggregate", {})
            for metric, score in aggregate.items():
                emoji = "✅" if score >= 0.7 else "⚠️" if score >= 0.5 else "❌"
                lines.append(f"| {metric} | {emoji} {score:.4f} |")
            lines.extend(["", ""])

            per_q = ragas.get("per_question", [])
            if per_q:
                lines.extend([
                    "### Per-Question Breakdown",
                    "",
                    "| # | Question | Faithfulness | Relevancy | Context Prec. | Context Recall |",
                    "|---|----------|-------------|-----------|---------------|----------------|",
                ])
                for i, q in enumerate(per_q):
                    question = q.get("question", "")[:50]
                    lines.append(
                        f"| {i + 1} | {question} | "
                        f"{q.get('faithfulness', 0):.3f} | "
                        f"{q.get('answer_relevancy', 0):.3f} | "
                        f"{q.get('context_precision', 0):.3f} | "
                        f"{q.get('context_recall', 0):.3f} |"
                    )
                lines.extend(["", ""])

        # DeepEval Results
        if "deepeval" in results:
            deepeval = results["deepeval"]
            lines.extend(["## DeepEval Metrics", ""])

            aggregate = deepeval.get("aggregate", {})
            if aggregate:
                lines.extend([
                    "| Metric | Value |",
                    "|--------|-------|",
                    f"| Total Cases | {aggregate.get('total_cases', 0)} |",
                    f"| Passed Cases | {aggregate.get('passed_cases', 0)} |",
                    f"| Pass Rate | {aggregate.get('pass_rate', 0):.2%} |",
                    f"| Avg Faithfulness | {aggregate.get('avg_faithfulness', 0):.4f} |",
                    f"| Avg Relevance | {aggregate.get('avg_relevance', 0):.4f} |",
                    "", "",
                ])

        # Custom Metrics
        if "custom" in results:
            custom = results["custom"]
            lines.extend([
                "## Custom Metrics",
                "",
                "| Metric | Score |",
                "|--------|-------|",
            ])
            for metric, score in custom.items():
                if isinstance(score, (int, float)):
                    lines.append(f"| {metric} | {score:.4f} |")
            lines.extend(["", ""])

        # Metadata
        if "metadata" in results:
            meta = results["metadata"]
            lines.extend([
                "## Evaluation Metadata",
                "",
                "| Key | Value |",
                "|-----|-------|",
            ])
            for key, value in meta.items():
                lines.append(f"| {key} | {value} |")
            lines.extend(["", ""])

        # Summary
        lines.extend([
            "---",
            "",
            "## Summary",
            "",
        ])

        all_scores: list[float] = []
        if "ragas" in results:
            all_scores.extend(results["ragas"].get("aggregate", {}).values())
        if "custom" in results:
            all_scores.extend(
                v for v in results["custom"].values() if isinstance(v, (int, float))
            )

        if all_scores:
            avg = sum(all_scores) / len(all_scores)
            if avg >= 0.8:
                lines.append("**Overall Assessment**: 🟢 Excellent — RAG pipeline performing well.")
            elif avg >= 0.6:
                lines.append(
                    "**Overall Assessment**: 🟡 Good — Some areas need improvement."
                )
            else:
                lines.append(
                    "**Overall Assessment**: 🔴 Needs Improvement — Significant quality issues."
                )
        else:
            lines.append("**Overall Assessment**: No scores available for summary.")

        lines.append("")
        return "\n".join(lines)

    def save_report(self, results: dict, output_path: str) -> None:
        """
        Save evaluation report to file.

        Args:
            results: Evaluation results dictionary.
            output_path: Output file path (.md for markdown, .json for raw data).
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
        else:
            report = self.generate_report(results)
            with open(path, "w", encoding="utf-8") as f:
                f.write(report)

        logger.info("evaluation_report_saved", path=str(path))
