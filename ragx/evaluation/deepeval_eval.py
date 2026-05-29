"""
RAGX DeepEval Evaluation — Evaluate RAG pipeline using DeepEval metrics.

Supports faithfulness, answer relevance, and batch evaluation with test cases.
"""

from __future__ import annotations

from typing import Any

from ragx.config.logging_config import get_logger

logger = get_logger(__name__)


class DeepEvalEvaluator:
    """Evaluator using DeepEval framework for RAG quality metrics."""

    def evaluate_faithfulness(
        self,
        query: str,
        answer: str,
        context: list[str],
        threshold: float = 0.7,
    ) -> dict:
        """
        Evaluate faithfulness of an answer against provided context.

        Args:
            query: The input query.
            answer: The generated answer.
            context: List of context strings used to generate the answer.
            threshold: Minimum acceptable faithfulness score.

        Returns:
            Dict with score, passed (bool), and details.
        """
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=context,
        )

        metric = FaithfulnessMetric(threshold=threshold)

        try:
            metric.measure(test_case)
            result = {
                "metric": "faithfulness",
                "score": float(metric.score) if metric.score is not None else 0.0,
                "passed": metric.is_successful(),
                "threshold": threshold,
                "reason": getattr(metric, "reason", ""),
            }
        except Exception as e:
            logger.error("deepeval_faithfulness_failed", error=str(e))
            result = {
                "metric": "faithfulness",
                "score": 0.0,
                "passed": False,
                "threshold": threshold,
                "reason": f"Evaluation failed: {e}",
            }

        logger.info("faithfulness_evaluated", score=result["score"], passed=result["passed"])
        return result

    def evaluate_relevance(
        self,
        query: str,
        answer: str,
        threshold: float = 0.7,
    ) -> dict:
        """
        Evaluate relevance of an answer to the query.

        Args:
            query: The input query.
            answer: The generated answer.
            threshold: Minimum acceptable relevance score.

        Returns:
            Dict with score, passed (bool), and details.
        """
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
        )

        metric = AnswerRelevancyMetric(threshold=threshold)

        try:
            metric.measure(test_case)
            result = {
                "metric": "answer_relevancy",
                "score": float(metric.score) if metric.score is not None else 0.0,
                "passed": metric.is_successful(),
                "threshold": threshold,
                "reason": getattr(metric, "reason", ""),
            }
        except Exception as e:
            logger.error("deepeval_relevance_failed", error=str(e))
            result = {
                "metric": "answer_relevancy",
                "score": 0.0,
                "passed": False,
                "threshold": threshold,
                "reason": f"Evaluation failed: {e}",
            }

        logger.info("relevance_evaluated", score=result["score"], passed=result["passed"])
        return result

    def evaluate_batch(self, test_cases: list[dict]) -> dict:
        """
        Evaluate multiple test cases in batch.

        Args:
            test_cases: List of dicts, each with keys:
                - query (str): Input query
                - answer (str): Generated answer
                - context (list[str]): Retrieved context
                - threshold (float, optional): Score threshold

        Returns:
            Dict with aggregate scores and per-case breakdown.
        """
        logger.info("starting_batch_evaluation", num_cases=len(test_cases))

        results = []
        total_faithfulness = 0.0
        total_relevance = 0.0
        passed_count = 0

        for i, tc in enumerate(test_cases):
            threshold = tc.get("threshold", 0.7)

            faith_result = self.evaluate_faithfulness(
                query=tc["query"],
                answer=tc["answer"],
                context=tc.get("context", []),
                threshold=threshold,
            )

            rel_result = self.evaluate_relevance(
                query=tc["query"],
                answer=tc["answer"],
                threshold=threshold,
            )

            case_passed = faith_result["passed"] and rel_result["passed"]
            if case_passed:
                passed_count += 1

            total_faithfulness += faith_result["score"]
            total_relevance += rel_result["score"]

            results.append(
                {
                    "case_index": i,
                    "query": tc["query"],
                    "faithfulness": faith_result,
                    "answer_relevancy": rel_result,
                    "overall_passed": case_passed,
                }
            )

        n = len(test_cases) if test_cases else 1
        aggregate = {
            "total_cases": len(test_cases),
            "passed_cases": passed_count,
            "pass_rate": passed_count / n,
            "avg_faithfulness": total_faithfulness / n,
            "avg_relevance": total_relevance / n,
        }

        logger.info("batch_evaluation_complete", **aggregate)
        return {"aggregate": aggregate, "per_case": results}
